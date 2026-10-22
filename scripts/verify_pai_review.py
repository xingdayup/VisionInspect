"""Finite real GPU protocol/parity/TTL verification; never an accuracy benchmark."""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,offline,read_json,write_json,sha256
from visioninspect.service import TaskStore,registry
from visioninspect.tools import ToolKit
from visioninspect.vllm_client import VLLMClient


def verify(image,ttl_smoke=False):
    configure_environment()
    output=ROOT/'outputs'/f'{datetime.now():%Y%m%d-%H%M%S}-gpu-verification-{uuid4().hex[:6]}'
    store=TaskStore(output/'service');started=time.monotonic();report=None
    try:
        policy=__import__('yaml').safe_load((ROOT/'configs/agent.yaml').read_text('utf8'))
        tools=ToolKit(registry()['bottle']['patchcore'],policy,output/'direct')
        request=dict(image_path=str(image.resolve()),category='bottle')
        with offline():
            quality=tools.invoke('inspect_image_quality',request)
            if not quality['valid']:raise ValueError('Use a genuine readable bottle image with passing image quality')
            detection=tools.invoke('detect_anomaly',request)
            references=tools.invoke('retrieve_normal_reference',request)
        config=store.gpu.config
        if not config['enabled']:raise ValueError('Set VISIONINSPECT_GPU_ENABLED=1 explicitly for this verification')
        original=output/'direct/evidence/original.png'
        reference=output/'direct'/references['reference_paths'][0]
        # A real-image central crop tests the image protocol only; it is not an anomaly ROI.
        crops=[]
        for name,path in [('query_crop',original),('reference_crop',reference)]:
            with Image.open(path) as source:
                width,height=source.size;destination=output/'direct'/f'{name}.png'
                source.crop((width//4,height//4,3*width//4,3*height//4)).save(destination);crops.append(destination)
        client=VLLMClient(config);probes=[]
        with store.gpu.acquire() as startup:
            with offline(allow_loopback_ports=(8001,)):
                for index,images in enumerate([
                    [dict(id='original',path=original),dict(id='reference:1',path=reference)],
                    [dict(id='original',path=original),dict(id='region:R1',path=crops[0]),dict(id='matched:R1:1',path=crops[1])]],1):
                    context=dict(category='bottle',engineering_protocol_smoke=True,engineering_crop_only=index==2,
                        available_evidence=[i['id'] for i in images],region_ids=['R1'],reference_ids=[1],calls_remaining_after_this=0,
                        instruction='仅验证真实图像和 JSON 接口。中间裁剪不是已检测缺陷，不推断产品合格或工业放行。')
                    value,metrics=client.decide(context,images,output/f'round_{index}')
                    probes.append(dict(image_count=len(images),decision=value.model_dump(),metrics=metrics))
        task=store.create(image.read_bytes(),image.name,'bottle','patchcore','agent_review')
        deadline=time.monotonic()+200
        while task['status'] in ('queued','running'):
            if time.monotonic()>deadline:raise TimeoutError('Application task exceeded verification timeout')
            time.sleep(.1);task=store.get(task['id'])
        if task['status']!='completed':raise RuntimeError(task['error'])
        actual=task['report']['outputs']['detection'];task_directory=store.directory(task['id'])
        np.testing.assert_allclose(actual['score'],detection['score'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(np.load(task_directory/actual['anomaly_map_path'],allow_pickle=False),
                                   np.load(output/'direct'/detection['anomaly_map_path'],allow_pickle=False),rtol=1e-5,atol=1e-6)
        gpu_before_ttl=store.gpu.status();ttl_verified=False
        if ttl_smoke:
            store.gpu.config=dict(store.gpu.config,idle_ttl_seconds=2)
            deadline=time.monotonic()+30
            while store.gpu.status()['pid'] is not None:
                if time.monotonic()>deadline:raise TimeoutError('GPU idle TTL did not reap the process tree')
                time.sleep(.2)
            ttl_verified=True
        report=dict(passed=True,image_sha256=sha256(image),seconds=time.monotonic()-started,probes=probes,
            task_id=task['id'],task_report=task['report'],parity_passed=True,gpu_runtime=gpu_before_ttl,
            shortened_ttl_verified=ttl_verified,semantic_accuracy_evaluated=False,
            full_agent_loop_exercised=task['report']['vlm_calls']>1,
            notice='真实 GPU 工程验证，中心裁剪仅用于输入协议；不产生准确率指标或工业放行结论。')
        write_json(output/'verification.json',report)
        print(json.dumps(dict(passed=True,output=str(output),gpu=gpu_before_ttl,ttl_verified=ttl_verified,
                             full_agent_loop_exercised=report['full_agent_loop_exercised']),ensure_ascii=False,indent=2))
    except Exception as error:
        write_json(output/'failure.json',dict(error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-started,
                                             semantic_accuracy_evaluated=False))
        raise
    finally:store.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path,required=True);parser.add_argument('--ttl-smoke',action='store_true')
    args=parser.parse_args()
    if sys.platform!='linux':parser.error('Run real GPU verification on PAI Linux')
    verify(args.image,args.ttl_smoke)
