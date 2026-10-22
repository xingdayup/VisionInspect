from io import BytesIO
import json,sys,time
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from visioninspect.api import create_app
from visioninspect.service import TaskStore,timestamp
from visioninspect.common import read_json,write_json

def wait_task(store,task_id):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        task=store.get(task_id)
        if task['status'] in {'completed','failed'}:return task
        time.sleep(.05)
    raise AssertionError('Task did not terminate')

def png():
    stream=BytesIO();Image.new('RGB',(256,256),'gray').save(stream,format='PNG');return stream.getvalue()

def test_upload_validation_and_unregistered_model(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post('/api/tasks',files={'file':('bad.png',b'not an image','image/png')}).status_code==422
        assert client.post('/api/tasks',files={'file':('huge.png',b'x'*(10*1024**2+1),'image/png')}).status_code==413
        assert client.post('/api/tasks',files={'file':('oversized.png',b'x'*(11*1024**2),'image/png')}).status_code==413
        assert client.post('/api/tasks',files={'file':('a.png',png(),'image/png')},data={'category':'unknown'}).status_code==422
        assert client.get('/api/tasks/not-a-task').status_code==404
        assert client.get('/api/tasks?limit=0').status_code==422

def test_real_quality_failure_is_reviewable_and_persistent(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/tasks',files={'file':('../flat.png',png(),'image/png')})
        assert response.status_code==202;task=response.json();assert task['filename']=='flat.png'
        task_id=task['id'];deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            task=client.get('/api/tasks/'+task_id).json()
            if task['status'] in {'completed','failed'}:break
            time.sleep(.1)
        assert task['status']=='completed' and task['state']=='manual_review'
        assert task['report']['outputs']['detection'] is None if 'detection' in task['report']['outputs'] else True
        assert client.get(f'/api/tasks/{task_id}/artifacts/worker.log').status_code==404
        before=task['report']['result']
        review=client.post(f'/api/tasks/{task_id}/review',json={'decision':'manual_review','note':'单元测试复核记录'})
        assert review.status_code==200
        report=client.get(f'/api/tasks/{task_id}/report').json()
        assert report['result']==before and report['human_review']['decision']=='manual_review'
        assert client.get('/api/tasks?q=flat').json()['total']==1
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/tasks/'+task_id).json()['review']['note']=='单元测试复核记录'

def test_missing_model_is_a_real_service_error(tmp_path,monkeypatch):
    monkeypatch.setattr('visioninspect.api.model_health',lambda:[dict(category='bottle',model='patchcore',available=False,reason='模型资产缺失')])
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/tasks',files={'file':('image.png',png(),'image/png')})
        assert response.status_code==503 and response.json()['detail']=='模型资产缺失'
        assert client.get('/api/tasks').json()['total']==0
        assert client.get('/ready').status_code==503

@pytest.mark.parametrize('exit_code,state',[(17,'normal_candidate'),(0,'released')])
def test_failed_output_is_never_accepted_and_next_job_recovers(tmp_path,record_property,exit_code,state):
    class FaultThenRealWorker(TaskStore):
        first=True
        def worker_command(self,directory):
            if not self.first:return super().worker_command(directory)
            self.first=False
            # Explicit fault-injection output, never an experiment or a user result.
            report=dict(schema_version=1,result=dict(state=state,reasons=['UNIT TEST INJECTION'],
                        evidence_paths=[],rule_version='unit-test'),outputs={},trace=[],notice='UNIT TEST ONLY')
            code='from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2],encoding="utf8"); sys.exit(int(sys.argv[3]))'
            return [sys.executable,'-B','-c',code,str(directory/'report.json'),json.dumps(report),str(exit_code)]
    store=FaultThenRealWorker(tmp_path)
    try:
        failed=store.create(png(),'fault-fixture.png','bottle','patchcore','dynamic')
        recovered=store.create(png(),'flat-fixture.png','bottle','patchcore','dynamic')
        failed=wait_task(store,failed['id']);recovered=wait_task(store,recovered['id'])
        assert failed['status']=='failed' and failed['state']=='manual_review'
        assert failed['report']['result']['state']=='manual_review'
        assert not failed['report']['supervisor']['worker_report_accepted']
        directory=store.directory(failed['id'])
        assert read_json(directory/'worker-report-unaccepted.json')['result']['state']==state
        assert (directory/'supervisor-error.log').is_file()
        assert recovered['status']=='completed' and recovered['state']=='manual_review'
        assert recovered['report']['outputs']['quality']['valid'] is False
        record_property('controlled_recovery_case','true')
        record_property('vision_accuracy_evaluation','false')
    finally:store.close()

def test_interrupted_task_report_is_invalidated_on_restart(tmp_path):
    store=TaskStore(tmp_path);task_id='a'*32;directory=store.directory(task_id);directory.mkdir()
    write_json(directory/'report.json',dict(result=dict(state='normal_candidate',evidence_paths=[]),trace=[]))
    write_json(directory/'progress.json',dict(stage='detect_anomaly',trace=[]))
    with store.connect() as db:
        db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (task_id,timestamp(),timestamp(),'running','bottle','patchcore','dynamic','interrupted-fixture.png',None,None,None,None))
    store.close();restarted=TaskStore(tmp_path)
    try:
        task=restarted.get(task_id)
        assert task['status']=='failed' and task['report']['result']['state']=='manual_review'
        assert task['report']['stop_reason']=='supervisor_failure'
        assert read_json(directory/'worker-report-unaccepted.json')['result']['state']=='normal_candidate'
    finally:restarted.close()

def test_missing_or_corrupt_report_has_readable_http_error(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        task=client.post('/api/tasks',files={'file':('flat.png',png(),'image/png')}).json()
        task=wait_task(client.app.state.store,task['id'])
        path=client.app.state.store.directory(task['id'])/'report.json';path.unlink()
        assert client.get(f"/api/tasks/{task['id']}/report").status_code==410
        path.write_text('{invalid',encoding='utf8')
        response=client.get(f"/api/tasks/{task['id']}")
        assert response.status_code==409 and '损坏' in response.json()['detail']
