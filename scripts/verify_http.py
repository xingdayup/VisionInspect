"""Exercise a real local server using a real bottle image, never substitute model outputs."""
import json,sys,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import write_json

with httpx.Client(base_url='http://127.0.0.1:8000',timeout=15) as client:
    source=ROOT/'data/mvtec_ad/bottle/test/broken_large/000.png'
    response=client.post('/api/tasks',files={'file':(source.name,source.read_bytes(),'image/png')})
    print('create',response.status_code,flush=True);response.raise_for_status();task=response.json()
    (ROOT/'logs/integration-task-id.txt').write_text(task['id'])
    deadline=time.monotonic()+180
    while task['status'] in {'queued','running'} and time.monotonic()<deadline:
        time.sleep(.5);task=client.get('/api/tasks/'+task['id']).json()
    write_json(ROOT/'logs/integration-task.json',task)
    print(task['id'],task['status'],task['state'],task['seconds'],flush=True)
    assert task['status']=='completed',task.get('error')
    assert task['report']['outputs']['detection']['score']>0
    for filename in ('original.png','inspection_heatmap.png','inspection_overlay.png','inspection.npy'):
        response=client.get(f"/api/tasks/{task['id']}/artifacts/{filename}")
        print(filename,response.status_code,len(response.content),flush=True);assert response.status_code==200
    response=client.get(f"/api/tasks/{task['id']}/report")
    assert response.status_code==200 and response.json()['task_id']==task['id']
    print('trace',[item['tool'] for item in task['report']['trace']],flush=True)
    print('HTTP_INTEGRATION_PASSED',flush=True)
