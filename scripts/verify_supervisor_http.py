"""Real-image HTTP regression through the strengthened subprocess supervisor."""
from pathlib import Path
from datetime import datetime
import sys,time
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.api import create_app
from visioninspect.common import read_json,write_json

directory=ROOT/'outputs'/f'{datetime.now():%Y%m%d-%H%M%S}-supervisor-http'
with TestClient(create_app(directory)) as client:
    source=ROOT/'data/mvtec_ad/bottle/test/broken_large/000.png'
    response=client.post('/api/tasks',files={'file':(source.name,source.read_bytes(),'image/png')})
    assert response.status_code==202;task=response.json();deadline=time.monotonic()+180
    while task['status'] in {'queued','running'} and time.monotonic()<deadline:
        time.sleep(.25);task=client.get('/api/tasks/'+task['id']).json()
    assert task['status']=='completed' and task['state']=='anomaly_candidate',task.get('error')
    expected=read_json(ROOT/'outputs/20261021-232247-evaluate-5225f8/predictions.json')[0]['score']
    assert abs(task['report']['outputs']['detection']['score']-expected)<1e-6
    assert client.get(f"/api/tasks/{task['id']}/report").json()['result']['state']=='anomaly_candidate'
    assert client.get(f"/api/tasks/{task['id']}/artifacts/inspection_overlay.png").status_code==200
    result=dict(passed=True,task_id=task['id'],state=task['state'],score=expected,seconds=task['seconds'],
                trace_tools=[step['tool'] for step in task['report']['trace']],output=str(directory))
    write_json(ROOT/'logs/supervisor-http-regression.json',result);print(result,flush=True)
