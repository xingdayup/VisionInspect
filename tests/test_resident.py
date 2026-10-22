"""Process lifecycle fixtures only: these do not measure visual model accuracy."""
import json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import psutil,pytest
from visioninspect.common import ROOT,read_json,write_json
from visioninspect.resident import ResidentWorker
from visioninspect.model_cache import ModelCache


FAKE_WORKER='''
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from visioninspect.common import write_json,sha256
for sequence,line in enumerate(sys.stdin,1):
    command=json.loads(line);directory=Path(command['directory'])
    request=json.loads((directory/'request.json').read_text('utf8'));mode=request['fixture']
    if mode=='crash':sys.exit(17)
    if mode=='busy':time.sleep(.8)
    if mode=='hang':time.sleep(10)
    if mode=='memory':allocation=bytearray(128*1024**2);time.sleep(10)
    report={'fixture':'UNIT TEST ONLY','sequence':sequence,'pid':os.getpid()}
    write_json(directory/'report.json',report)
    write_json(directory/'worker-completion.json',dict(token='wrong' if mode=='bad_token' else command['token'],
        success=True,report_sha256=sha256(directory/'report.json'),pid=os.getpid(),cache={}))
'''


@pytest.fixture
def runner(tmp_path):
    fixture=tmp_path/'fixture_worker.py';fixture.write_text(FAKE_WORKER,encoding='utf8')
    instance=ResidentWorker(tmp_path/'runtime',lambda _: [sys.executable,'-B',str(fixture),str(ROOT)],ttl=1)
    instance.poll_seconds=.025
    yield instance
    instance.close()


def job(runner,index,fixture='normal'):
    directory=runner.directory.parent/f'{index:032x}';directory.mkdir()
    write_json(directory/'request.json',dict(fixture=fixture))
    return directory


def wait_until(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.025)
    raise AssertionError('Lifecycle condition not reached')


def test_process_is_reused_and_tasks_have_separate_reports(runner):
    first=job(runner,1);second=job(runner,2)
    one=runner.run(first);two=runner.run(second)
    assert one['pid']==two['pid']
    assert read_json(first/'report.json')['sequence']==1
    assert read_json(second/'report.json')['sequence']==2
    assert not read_json(first/'execution.json')['process_reused']
    assert read_json(second/'execution.json')['process_reused']
    assert read_json(second/'execution.json')['task_exit_code']==0
    assert read_json(second/'execution.json')['exit_code'] is None  # Resident process is still alive.


def test_idle_expiry_releases_process_and_next_task_starts_new_generation(runner):
    first=job(runner,1);runner.run(first);generation=runner.generation
    inference_pid=runner.status()['inference_pid']
    wait_until(lambda:runner.status()['state']=='stopped')
    assert runner.status()['last_stop_reason']=='idle_ttl'
    assert not psutil.pid_exists(inference_pid)
    runner.run(job(runner,2))
    assert runner.generation!=generation
    assert not read_json(runner.directory.parent/f'{2:032x}'/'execution.json')['process_reused']


def test_busy_task_outlives_idle_ttl_and_resets_it_on_completion(runner):
    runner.ttl=.2
    directory=job(runner,1,'busy');runner.run(directory)
    assert runner.status()['state']=='idle'
    assert runner.status()['expires_in_seconds']>.1
    assert read_json(directory/'report.json')['sequence']==1


@pytest.mark.parametrize('fixture,expected',[('crash','worker_failure'),('bad_token','protocol_failure'),
                                          ('hang','timeout'),('memory','memory_limit')])
def test_failure_discards_worker_and_next_task_recovers(runner,fixture,expected):
    if fixture=='hang':runner.task_timeout_seconds=.3
    if fixture=='memory':runner.memory_limit_bytes=96*1024**2
    directory=job(runner,1,fixture)
    with pytest.raises(RuntimeError,match=expected):runner.run(directory)
    generation=runner.generation
    assert runner.status()['state']=='stopped' and read_json(directory/'execution.json')['failure']
    assert read_json(directory/'execution.json')['task_exit_code']==1
    runner.task_timeout_seconds=180;runner.memory_limit_bytes=12*1024**3
    runner.run(job(runner,2))
    assert runner.generation!=generation


def test_close_interrupts_active_work_and_releases_process(runner):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(runner.run,job(runner,1,'hang'))
        wait_until(lambda:runner.status()['state']=='running')
        pid=runner.status()['pid'];runner.close()
        with pytest.raises(RuntimeError,match='interrupted'):future.result(timeout=5)
        assert not psutil.pid_exists(pid)
    assert not runner.monitor.is_alive()


def test_dead_launcher_does_not_leave_a_busy_interpreter_child(runner):
    first=runner.run(job(runner,1));inference_pid=first['pid']
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(runner.run,job(runner,2,'hang'))
        wait_until(lambda:runner.status()['state']=='running')
        runner.process.kill()
        with pytest.raises(RuntimeError,match='worker_failure'):future.result(timeout=10)
    wait_until(lambda:not psutil.pid_exists(inference_pid))
    assert runner.status()['state']=='stopped'


def test_detector_cache_reuses_slot_and_invalidates_changed_assets(tmp_path):
    checkpoint=tmp_path/'fit/model.ckpt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'FIXTURE')
    registration=dict(fit_run=str(checkpoint.parent),profile=str(tmp_path/'profile.json'),references=str(tmp_path/'refs.npz'))
    cache=ModelCache();cache.bind_detector(registration,{'digest':'fixture'})
    detector=object();references=object();cache.detector=detector;cache.references=references
    cache.bind_detector(registration,{'digest':'fixture'})
    assert cache.detector is detector and cache.references is references
    checkpoint.write_bytes(b'CHANGED FIXTURE')
    cache.bind_detector(registration,{'digest':'fixture'})
    assert cache.detector is None and cache.references is None
    cache.detector=detector;cache.bind_detector(registration,{'digest':'different profile'})
    assert cache.detector is None
