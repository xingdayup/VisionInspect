"""Finite isolated real-model cache/TTL regression; never a formal test-set evaluation."""
import argparse,math,sys,time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,load_config,read_json,write_json,sha256
configure_environment()
from fastapi.testclient import TestClient
from visioninspect.api import create_app
from visioninspect.data import load_manifest


def verify(skip_vlm=False):
    directory=ROOT/'outputs'/f'{datetime.now():%Y%m%d-%H%M%S}-resident-verification-{uuid4().hex[:6]}'
    directory.mkdir()
    manifest,validation=load_manifest(load_config(ROOT/'configs/patchcore_cpu.yaml'),('validation',))
    sample=next(record for record in validation if record['path']=='train/good/002.png')
    source=Path(manifest['root'])/sample['path'];body=source.read_bytes();tasks=[]
    with TestClient(create_app(directory/'service')) as client:
        runtime=client.app.state.store.runtime
        configured_ttl=runtime.ttl
        def submit(name,vlm):
            response=client.post('/api/tasks',files={'file':(name,body,'image/png')},
                data=dict(category='bottle',model='patchcore',mode='fixed' if vlm else 'baseline',vlm=str(vlm).lower()))
            assert response.status_code==202,response.text
            task_id=response.json()['id'];deadline=time.monotonic()+180
            while True:
                task=client.get(f'/api/tasks/{task_id}').json()
                if task['status'] in {'completed','failed'}:break
                assert time.monotonic()<deadline,'Real-model regression timed out'
                time.sleep(.1)
            assert task['status']=='completed' and not task['report']['errors'],task
            assert client.get(f'/api/tasks/{task_id}/report').status_code==200
            assert client.get(f'/api/tasks/{task_id}/artifacts/inspection_overlay.png').status_code==200
            task['execution']=read_json(directory/'service'/task_id/'execution.json');tasks.append(task)
            print(dict(task=task_id,seconds=task['seconds'],initialization=task['report']['initialization'],
                       runtime=task['report']['runtime']),flush=True)
            return task
        first=submit('validation-002-cold.png',not skip_vlm)
        second=submit('validation-002-warm.png',not skip_vlm)
        a=first['report'];b=second['report']
        assert a['runtime']['worker_pid']==b['runtime']['worker_pid']
        assert a['runtime']['generation']==b['runtime']['generation']
        assert not a['initialization']['timings']['cache_hit'] and b['initialization']['timings']['cache_hit']
        assert a['runtime']['task_sequence']==1 and b['runtime']['task_sequence']==2
        assert first['execution']['process_reused'] is False and second['execution']['process_reused'] is True
        if not skip_vlm:
            assert not a['runtime']['vlm_loaded_before_task'] and b['runtime']['vlm_loaded_before_task']
            assert a['vlm_calls']==b['vlm_calls']==1
            for task in (first,second):
                audit=read_json(directory/'service'/task['id']/'vlm_input/input.json')
                assert audit['pixel_values_shape']==[1,2,3,512,512]
        cached_state=client.get('/api/runtime').json()
        assert cached_state['state']=='idle' and cached_state['cache']['detector_loaded']
        # Shortened in this verification process only; configs/service.yaml stays at 600 seconds.
        runtime.ttl=2.;deadline=time.monotonic()+10
        while client.get('/api/runtime').json()['state']!='stopped':
            assert time.monotonic()<deadline,'Idle TTL did not stop the worker'
            time.sleep(.1)
        expired=client.get('/api/runtime').json()
        assert expired['last_stop_reason']=='idle_ttl' and not expired['cache']
        import psutil
        assert not psutil.pid_exists(b['runtime']['worker_pid'])
        runtime.ttl=configured_ttl
        third=submit('validation-002-after-expiry.png',False);c=third['report']
        assert c['runtime']['generation']!=b['runtime']['generation']
        assert not c['initialization']['timings']['cache_hit']
        assert c['vlm_calls']==0 and not c['runtime']['model_cache']['vlm_loaded']
        import numpy as np
        original_map=np.load(directory/'service'/first['id']/'evidence/inspection.npy')
        assert original_map.shape==(256,256) and np.isfinite(original_map).all()
        for task in (second,third):
            score=task['report']['outputs']['detection']['score']
            assert math.isfinite(score) and math.isclose(score,a['outputs']['detection']['score'],rel_tol=1e-5,abs_tol=1e-6)
            actual=np.load(directory/'service'/task['id']/'evidence/inspection.npy')
            assert actual.shape==(256,256) and np.isfinite(actual).all()
            np.testing.assert_allclose(actual,original_map,rtol=1e-5,atol=1e-6)
            assert task['report']['result']['state']==a['result']['state']
        pid_before_close=c['runtime']['worker_pid']
    assert not psutil.pid_exists(pid_before_close),'Service shutdown left an inference process running'
    result=dict(passed=True,source=sample['path'],split=sample['split'],source_sha256=sha256(source),
        test_set_used=False,formal_metrics=False,semantic_accuracy_evaluated=False,isolated_state=True,
        configured_idle_ttl_seconds=configured_ttl,verification_idle_ttl_seconds=2.,expired=expired,
        vlm_exercised=not skip_vlm,cached_status=cached_state,shutdown_released_worker=True,
        tasks=[dict(id=t['id'],seconds=t['seconds'],initialization=t['report']['initialization'],
            runtime=t['report']['runtime'],execution=t['execution'],score=t['report']['outputs']['detection']['score']) for t in tasks])
    write_json(directory/'verification.json',result)
    print('OUTPUT',directory,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-vlm',action='store_true')
    verify(parser.parse_args().skip_vlm)
