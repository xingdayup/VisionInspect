"""Inspect newly resolved frontend packages before installation, without fetching tarballs."""
import concurrent.futures,json,sys,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import read_json,write_json
before=read_json(ROOT/'logs/frontend-lock-before-radix.json')['packages']
after=read_json(ROOT/'frontend/package-lock.json')['packages']
changed=[name for name,item in before.items() if name and name in after and item.get('version')!=after[name].get('version')]
if changed:raise ValueError(f'Existing dependency versions changed: {changed}')
new=[(name,item) for name,item in after.items() if name and name not in before]
def inspect(pair):
    path,item=pair;name=path.split('node_modules/')[-1];version=item['version']
    url=f'https://registry.npmjs.org/{urllib.parse.quote(name,safe="@")}'+f'/{version}'
    with urllib.request.urlopen(url,timeout=20) as response:metadata=json.load(response)
    size=metadata['dist']['unpackedSize']
    if not item.get('integrity'):raise ValueError('Missing npm lock integrity')
    return dict(name=name,version=version,license=metadata.get('license'),unpacked_bytes=size)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:packages=list(pool.map(inspect,new))
total=sum(item['unpacked_bytes'] for item in packages)
result=dict(new_packages=packages,new_unpacked_bytes=total,limit_bytes=3*1024**2,
            existing_versions_changed=changed,within_budget=total<=3*1024**2,
            note='Registry unpacked size, not measured network transfer')
write_json(ROOT/'logs/frontend-dependency-budget.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
if not result['within_budget']:raise SystemExit(1)
