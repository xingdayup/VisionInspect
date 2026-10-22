"""One explicitly authorized MVTec class per invocation; fixed revision, size and hash checks."""
import argparse, concurrent.futures, hashlib, json, shutil, sys, urllib.request
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import yaml
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visioninspect.common import read_json, write_json, sha256
from prepare_assets import fetch

REVISION = 'c75b39616f84db43677bcc8228caaafaf5096d7f'
ALLOWED = ('metal_nut', 'capsule', 'pill')

def prepare(category):
    prior = read_json(ROOT/'logs/category-extension-preflight.json')
    expected = next(item for item in prior['categories'] if item['category'] == category)
    url = f'https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/{REVISION}/{category}?recursive=true&limit=1000'
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.headers.get('Link'):
            raise ValueError('Unenumerated pagination')
        entries = json.load(response)
    files = [item for item in entries if item['type'] == 'file']
    size = sum(item['size'] for item in files)
    if (len(files),size) != (expected['files'],expected['bytes']) or size > 1_000_000_000:
        raise ValueError('Class budget or file count changed; review required')
    for item in files:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or path.parts[0] != category or '..' in path.parts or '\\' in item['path']:
            raise ValueError('Unsafe class path')
    license_url = f'https://huggingface.co/datasets/foersben/mvtec-ad/raw/{REVISION}/{category}/license.txt'
    with urllib.request.urlopen(license_url, timeout=30) as response:
        license_text = response.read(128*1024).decode('utf8')
    if 'Attribution-NonCommercial-ShareAlike' not in license_text or '4.0' not in license_text:
        raise ValueError('Class license mismatch')
    if shutil.disk_usage(ROOT).free < 15*1024**3:
        raise OSError('Less than 15 GiB free on project disk')
    preflight = dict(category=category, repository='foersben/mvtec-ad', revision=REVISION,
        license='CC-BY-NC-SA-4.0', license_url=license_url, license_sha256=hashlib.sha256(license_text.encode()).hexdigest(),
        timestamp=datetime.now(timezone.utc).isoformat(), provider='User-authorized third-party mirror; not officially hosted',
        official_source='https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
        user_authorization='2026-10-22: allowed metal_nut, capsule, pill sequentially after license/resource checks',
        total_files=len(files), total_bytes=size, free_bytes=shutil.disk_usage(ROOT).free,
        estimated_increment_bytes=size+250*1024**2, runtime_memory_budget_gib=12, gpu='not applicable: CPU', tree=entries)
    write_json(ROOT/f'logs/{category}-download-preflight.json', preflight)
    print(f'PREFLIGHT {category} files={len(files)} bytes={size} license=CC-BY-NC-SA-4.0', flush=True)
    def download(entry):
        path = ROOT/'data/mvtec_ad'/entry['path']
        url = f"https://huggingface.co/datasets/foersben/mvtec-ad/resolve/{REVISION}/{entry['path']}"
        checksum = entry.get('lfs',{}).get('oid')
        record = fetch(url, path, entry['size'], checksum)
        if not checksum:
            body = path.read_bytes()
            if hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest() != entry['oid']:
                raise ValueError(f'Git blob hash mismatch: {path}')
        return dict(source_path=entry['path'], git_blob_oid=entry.get('oid'), **record)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for index, item in enumerate(pool.map(download, files),1):
            records.append(item)
            if index % 25 == 0: print(f'downloaded/verified {index}/{len(files)}',flush=True)
    # Inspect actual images, not guessed category dimensions. Full decode/mask checks follow in check-data.
    dimensions, modes = set(), set()
    counts = {}
    for item in records:
        if item['source_path'].endswith('.png'):
            group = '/'.join(item['source_path'].split('/')[1:3])
            counts[group] = counts.get(group,0)+1
            with Image.open(ROOT/item['path']) as image:
                image.load(); dimensions.add(image.size)
                if not group.startswith('ground_truth/'): modes.add(image.mode)
    if counts != expected['counts'] or len(dimensions) != 1 or modes != {'RGB'}:
        raise ValueError(f'Unexpected image structure: {counts}, {dimensions}, {modes}')
    width,height = next(iter(dimensions))
    if width != height: raise ValueError('Current strict dataset protocol requires square original images')
    config = yaml.safe_load((ROOT/'configs/patchcore_cpu.yaml').read_text(encoding='utf8'))
    defects = sorted(key.split('/')[1] for key in counts if key.startswith('test/') and key != 'test/good')
    config['dataset'].update(category=category, root=f'data/mvtec_ad/{category}', manifest=f'data/manifests/{category}.json',
        image_size=width, expected_train=counts['train/good'], expected_test_good=counts['test/good'],
        expected_test_anomaly=sum(counts[f'test/{key}'] for key in defects), defects=defects)
    target = ROOT/f'configs/patchcore_cpu_{category}.yaml'
    if target.exists() and yaml.safe_load(target.read_text(encoding='utf8')) != config:
        raise ValueError('Existing class config differs; will not overwrite')
    target.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf8')
    provenance = {k:v for k,v in preflight.items() if k != 'tree'}
    provenance.update(files=records, actual_image_size=[width,height], actual_counts=counts, fully_decoded=True)
    write_json(ROOT/f'data/mvtec_ad/{category}-source.json',provenance)
    print(f'CLASS_READY {category} bytes={size} dimensions={width} counts={counts}',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('category',choices=ALLOWED)
    prepare(parser.parse_args().category)
