"""Offline VLM smoke check on a normal validation image, without testing prompt alternatives."""
from pathlib import Path
import sys,time
import psutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,load_config,offline,write_json
from visioninspect.data import load_manifest
from visioninspect.vlm import LocalVLM

configure_environment();started=time.perf_counter()
with offline():
    config=load_config(ROOT/'configs/patchcore_cpu.yaml')
    manifest,validation=load_manifest(config,('validation',));_,training=load_manifest(config,('train',))
    root=Path(manifest['root'])
    model=LocalVLM(ROOT/'weights/smolvlm-500m',48)
    model.load();loaded=time.perf_counter()
    with Image.open(root/validation[0]['path']) as image,Image.open(root/training[0]['path']) as reference:
        result=model.describe(image.convert('RGB'),reference.convert('RGB'))
    record=dict(validation_image=validation[0]['path'],reference_image=training[0]['path'],result=result,
                loading_seconds=loaded-started,generation_seconds=time.perf_counter()-loaded,total_seconds=time.perf_counter()-started,
                peak_process_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
                network='Python socket connections disabled',device='CPU FP32',test_images_used=False)
    write_json(ROOT/'logs/vlm-smoke.json',record);print(record,flush=True)
