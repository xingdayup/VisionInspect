"""Read-only, bounded local dataset search; folder existence never establishes a license."""
import os, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visioninspect.common import write_json

names = ('imagenet', 'ImageNet', 'imagenet1k', 'ILSVRC2012', 'imagenette', 'imagenette2', 'imagenette2-320', 'imagenette2-160')
parents = [ROOT/'data', ROOT/'datasets', ROOT/'.cache', ROOT/'.cache/huggingface/datasets',
           Path('E:/data'), Path('E:/datasets'), Path('D:/data'), Path('D:/datasets'),
           Path.home()/'data', Path.home()/'datasets']
candidates = {str((parent/name).resolve()) for parent in parents for name in names}
configured = {}
for key in ('IMAGENET_ROOT', 'IMAGENETTE_ROOT', 'IMAGENET_TRAIN_PATH'):
    if os.environ.get(key):
        configured[key] = os.environ[key]
        candidates.add(str(Path(os.environ[key]).resolve()))
records = []
for value in sorted(candidates):
    path = Path(value)
    record = dict(path=value, exists=path.exists(), authorized=False)
    if path.is_dir():
        images = [p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        record.update(image_files=len(images), sample=[str(p.relative_to(path)) for p in images[:5]],
                      completeness='Not established: no authorized manifest supplied')
    records.append(record)
result = dict(timestamp=datetime.now(timezone.utc).isoformat(), scope='Named dataset locations only; no whole-disk or unrelated-project crawl',
              inspected_parents=[dict(path=str(p), exists=p.exists()) for p in parents],
              configured_dataset_paths=configured, candidates=records,
              existing_candidates=[r for r in records if r['exists']],
              authorized_regularizer_manifest=(ROOT/'data/manifests/imagenette-authorized.json').exists(),
              conclusion='No authorized regularization dataset established by this bounded check. User supplied no authorized path. No dataset downloaded.')
write_json(ROOT/'logs/regularization-data-audit.json', result)
print({k:v for k,v in result.items() if k != 'candidates'})
