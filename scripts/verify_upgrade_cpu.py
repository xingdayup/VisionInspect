"""Real CPU parity plus unavailable-GPU recovery, using normal validation images only."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,load_config,offline,project_path,read_json,sha256,write_json
from visioninspect.data import load_manifest
from visioninspect.model import Detector,setup
from visioninspect.service import TaskStore,registry


def main():
    configure_environment();output=ROOT/'outputs'/f'{datetime.now():%Y%m%d-%H%M%S}-upgrade-cpu-verification-{uuid4().hex[:6]}'
    output.mkdir();config=load_config(ROOT/'configs/patchcore_cpu.yaml');registration=registry()['bottle']['patchcore']
    profile=read_json(project_path(registration['profile']));checkpoint=project_path(registration['fit_run'])/'model.ckpt'
    assert sha256(checkpoint)==profile['checkpoint_sha256']
    selected=None;examined=[];scores={};maps={}
    with offline():
        manifest,validation=load_manifest(config,('validation',))
        detector,metadata=Detector.load(checkpoint);setup(config)
        for record in validation:
            source=project_path(config['dataset']['root'])/record['path']
            with Image.open(source) as image:score,heat,_=detector.predict(np.asarray(image.convert('RGB')))
            examined.append(dict(path=record['path'],score=score))
            if record['path']=='train/good/002.png':scores['clear']=score;maps['clear']=heat.copy()
            if selected is None and profile['review_lower']<=score<=profile['review_upper']:
                selected=record;scores['uncertain']=score;maps['uncertain']=heat.copy()
            if selected and 'clear' in scores:break
    assert selected is not None,'No eligible normal validation fixture; do not change the frozen review band'
    store=TaskStore(output/'service');results=[]
    try:
        # This process deliberately tests unavailable GPU behavior, never a replacement VLM.
        assert not store.gpu.config['enabled'],'Run this regression with GPU disabled'
        for key,path in [('clear','train/good/002.png'),('uncertain',selected['path'])]:
            source=project_path(config['dataset']['root'])/path
            task=store.create(source.read_bytes(),f'validation-{key}.png','bottle','patchcore','agent_review')
            deadline=time.monotonic()+180
            while task['status'] in {'queued','running'}:
                if time.monotonic()>deadline:raise TimeoutError('CPU regression timeout')
                time.sleep(.1);task=store.get(task['id'])
            actual=task['report']['outputs']['detection']
            np.testing.assert_allclose(actual['score'],scores[key],rtol=1e-5,atol=1e-6)
            heat=np.load(store.directory(task['id'])/actual['anomaly_map_path'],allow_pickle=False)
            np.testing.assert_allclose(heat,maps[key],rtol=1e-5,atol=1e-6)
            if key=='clear':assert task['status']=='completed' and task['report']['vlm_calls']==0
            else:
                assert task['status']=='failed' and task['state']=='manual_review'
                assert task['report']['agent_review']['status']=='failed'
                retry=store.retry(task['id']);assert retry['retry_of']==task['id']
                while store.get(retry['id'])['status'] in {'queued','running'}:
                    if time.monotonic()>deadline:raise TimeoutError('Retry regression timeout')
                    time.sleep(.1)
                retried=store.get(retry['id'])
                assert retried['report']['runtime']['worker_pid']==task['report']['runtime']['worker_pid']
                assert retried['report']['initialization']['timings']['cache_hit']
            results.append(dict(case=key,task=task,heatmap_max_abs_delta=float(np.max(np.abs(heat-maps[key])))))
        write_json(output/'verification.json',dict(passed=True,scope='Normal validation only; engineering parity and GPU-unavailable failure handling',
            examined=examined,results=results,gpu_model_inference_verified=False,semantic_accuracy_evaluated=False))
        print(json.dumps(dict(passed=True,output=str(output),cases=[dict(case=r['case'],status=r['task']['status'],score=r['task']['report']['outputs']['detection']['score'],heatmap_max_abs_delta=r['heatmap_max_abs_delta']) for r in results]),ensure_ascii=False,indent=2))
    except Exception as error:
        import traceback
        write_json(output/'failure.json',dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
            gpu_model_inference_verified=False,semantic_accuracy_evaluated=False))
        raise
    finally:store.close()


if __name__=='__main__':main()
