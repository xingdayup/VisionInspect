"""Architecture/throughput smoke only. Synthetic tensors are NOT a formal B experiment."""
import sys,time
from datetime import datetime
from uuid import uuid4
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,load_config,offline,write_json,sha256,identity
from visioninspect.model import setup
from visioninspect.efficientad import EfficientAD

configure_environment();config=load_config(ROOT/'configs/efficientad_cpu.yaml');setup(config)
directory=ROOT/'outputs'/f'{datetime.now():%Y%m%d-%H%M%S}-efficientad-smoke-{uuid4().hex[:6]}';directory.mkdir()
write_json(directory/'config.json',config)
with offline():
    detector=EfficientAD(config);teacher_sha=detector.load_teacher();optimizer=detector.optimizer()
    teacher_before={key:value.clone() for key,value in detector.model.teacher.state_dict().items()}
    batch=torch.rand((1,3,256,256));regularizer=torch.rand((1,3,256,256));times=[];losses=[]
    for index in range(7):
        start=time.perf_counter();losses.append(detector.training_step(batch,regularizer,optimizer))
        if index>=2:times.append(time.perf_counter()-start)
    assert all(torch.equal(value,teacher_before[key]) for key,value in detector.model.teacher.state_dict().items())
    array=(batch[0].permute(1,2,0).numpy()*255).astype(np.uint8);score,heatmap,_=detector.predict(array)
    checkpoint=directory/'synthetic-smoke.ckpt';detector.save(checkpoint,dict(purpose='synthetic architecture smoke; not trained benchmark'))
    restored,_=EfficientAD.load(checkpoint);score_after,heatmap_after,_=restored.predict(array)
    np.testing.assert_allclose(score,score_after,rtol=1e-5,atol=1e-6);np.testing.assert_allclose(heatmap,heatmap_after,rtol=1e-5,atol=1e-6)
    p95=float(np.quantile(times,.95));estimate=p95*config['training']['steps']
    result=dict(formal_training=False,formal_test_metrics=False,inputs='synthetic tensors, no dataset-performance claim',
                cpu_threads=config['runtime']['threads'],config_digest=identity(config),
                teacher_sha256=teacher_sha,checkpoint_sha256=sha256(checkpoint),teacher_frozen=True,finite_losses=True,
                checkpoint_roundtrip=True,optimizer_steps=7,timed_steps=5,step_p50_seconds=float(np.median(times)),step_p95_seconds=p95,
                configured_steps=config['training']['steps'],estimated_compute_seconds=estimate,fit_budget_seconds=config['runtime']['fit_timeout_seconds'],
                estimated_within_budget=estimate<config['runtime']['fit_timeout_seconds'],
                estimate_limitation='Synthetic compute only; real image decoding/statistics/validation add time; not a measured full fit')
    write_json(directory/'smoke.json',result);print(result,flush=True);print('OUTPUT',directory,flush=True)
