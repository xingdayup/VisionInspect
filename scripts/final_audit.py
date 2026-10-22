"""Read-only delivery snapshot. Logical file sizes can double-count hardlinks."""
import importlib.metadata,json,os,platform,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import sha256,write_json

def size(directory):
    total=0;count=0
    for parent,_,names in os.walk(directory):
        for name in names:
            try:total+=(Path(parent)/name).stat().st_size;count+=1
            except FileNotFoundError:pass
    return dict(bytes=total,files=count)

result=dict(timestamp=datetime.now(timezone.utc).isoformat(),root=str(ROOT),python=platform.python_version(),
            executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
            disk_free_bytes=shutil.disk_usage(ROOT).free,
            logical_directories={name:size(ROOT/name) for name in ('.venv','.cache','.tmp','data','weights','outputs','logs','frontend/node_modules','frontend/dist')},
            versions=sorted([dict(name=item.metadata['Name'],version=item.version) for item in importlib.metadata.distributions()],key=lambda v:v['name'].lower()),
            hashes={str(path.relative_to(ROOT)):sha256(path) for path in (ROOT/'requirements.txt',ROOT/'requirements-linux.txt',
                     ROOT/'frontend/package-lock.json',ROOT/'VisionInspect-Agent_交接文档_含前端要求.md',ROOT/'weights/smolvlm-500m/model.safetensors',
                     ROOT/'outputs/20261021-232056-fit-ae72d8/model.ckpt')},
            git_repository=(ROOT/'.git').exists(),paid_api=False,gpu_used=False,
            note='Logical sizes, may double-count shared hardlinks; not physical disk increment or measured network transfer')
write_json(ROOT/'logs/final-delivery-audit.json',result)
print(json.dumps({key:value for key,value in result.items() if key not in ('versions','hashes')},ensure_ascii=False,indent=2))
