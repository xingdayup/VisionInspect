"""Metadata and license text only; never downloads MVTec images or expands the data directory."""
import json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import write_json
REVISION='c75b39616f84db43677bcc8228caaafaf5096d7f'
result=[]
for category in ('metal_nut','capsule','pill'):
    url=f'https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/{REVISION}/{category}?recursive=true&limit=1000'
    with urllib.request.urlopen(url,timeout=25) as response:
        if response.headers.get('Link'):raise ValueError('Pagination requires further enumeration')
        entries=json.load(response)
    files=[entry for entry in entries if entry['type']=='file']
    if any(not item['path'].startswith(category+'/') for item in files):raise ValueError('Category path mismatch')
    with urllib.request.urlopen(f'https://huggingface.co/datasets/foersben/mvtec-ad/raw/{REVISION}/{category}/license.txt',timeout=25) as response:
        license_text=response.read(128*1024).decode('utf8')
    if 'Attribution-NonCommercial-ShareAlike' not in license_text:raise ValueError('Category license mismatch')
    counts={}
    for item in files:
        if item['path'].endswith('.png'):
            group='/'.join(item['path'].split('/')[1:3]);counts[group]=counts.get(group,0)+1
    record=dict(category=category,files=len(files),bytes=sum(item['size'] for item in files),counts=counts,
                license='CC-BY-NC-SA-4.0',revision=REVISION,image_downloaded=False)
    result.append(record);print(record,flush=True)
write_json(ROOT/'logs/category-extension-preflight.json',dict(metadata_only=True,categories=result,total_bytes=sum(item['bytes'] for item in result),
           permission_status='User authorized sequential expansion on 2026-10-22; each invocation still verifies class license and resource budget'))
