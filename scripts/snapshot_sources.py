"""Archive current experiment sources only when every byte matches the pre-recorded SHA."""
import shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import project_path,read_json,sha256
for value in sys.argv[1:]:
    directory=project_path(value);frozen=read_json(directory/'frozen.json');target=directory/'source/visioninspect'
    target.mkdir(parents=True,exist_ok=True)
    for name in ('agent.py','tools.py','vlm.py'):
        original=ROOT/'visioninspect'/name
        if sha256(original)!=frozen['code'][name]:raise ValueError(f'Cannot snapshot changed experiment source: {name}')
        shutil.copy2(original,target/name)
    print(directory.name,'source snapshot matches frozen SHA256',flush=True)
