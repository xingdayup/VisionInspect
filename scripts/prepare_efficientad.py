"""Prepare only the pinned official EfficientAD teacher archive, never a dataset."""
import json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_assets import fetch
from visioninspect.common import write_json,sha256

URL='https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip'
ARCHIVE_SHA='c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a'
directory=ROOT/'weights/efficientad';archive=directory/'efficientad_pretrained_weights.zip'
preflight=dict(url=URL,archive_bytes=39960466,archive_sha256=ARCHIVE_SHA,license='Apache-2.0',
               disk_budget_bytes=80*1024**2,device='CPU FP32',source='Official Anomalib release; hash pinned by anomalib 2.7.0')
write_json(ROOT/'logs/efficientad-asset-preflight.json',preflight)
record=fetch(URL,archive,39960466,ARCHIVE_SHA)
with zipfile.ZipFile(archive) as bundle:
    matches=[entry for entry in bundle.infolist() if Path(entry.filename).name=='pretrained_teacher_small.pth']
    if len(matches)!=1 or matches[0].file_size>20*1024**2:raise ValueError('Unexpected teacher member or size')
    content=bundle.read(matches[0]);target=directory/'pretrained_teacher_small.pth'
    if target.exists() and target.read_bytes()!=content:raise ValueError('Existing teacher differs; refusing overwrite')
    if not target.exists():target.write_bytes(content)
write_json(directory/'source.json',dict(**preflight,archive=record,teacher_path=target.relative_to(ROOT).as_posix(),
           teacher_bytes=target.stat().st_size,teacher_sha256=sha256(target),archive_member=matches[0].filename))
print(json.dumps(dict(teacher_bytes=target.stat().st_size,teacher_sha256=sha256(target),archive_verified=True)))
