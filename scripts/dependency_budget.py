"""Inspect exact locked Windows distributions before installation. Downloads metadata only."""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
TAGS = list(sys_tags())
RANK = {tag: index for index, tag in enumerate(TAGS)}
CPU = {
    'torch': ('https://download-r2.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-cp312-cp312-win_amd64.whl', 122900744),
    'torchvision': ('https://download-r2.pytorch.org/whl/cpu/torchvision-0.27.0%2Bcpu-cp312-cp312-win_amd64.whl', 4021192),
}

def read_metadata(url, destination):
    if destination.exists():
        return json.loads(destination.read_text(encoding='utf-8'))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                result = json.load(response)
            destination.write_text(json.dumps(result), encoding='utf-8')
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

def inspect(item):
    name, version = item
    if name in CPU:
        url, size = CPU[name]
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            actual = int(response.headers['Content-Length'])
        if size != actual:
            raise ValueError(f'CPU wheel size changed: {name}')
        return dict(name=name, version=version, url=url, size=size, kind='wheel')
    metadata = read_metadata(f'https://pypi.org/pypi/{name}/{version}/json',
                             ROOT/'.cache'/f'pypi-{name}-{version}.json')
    candidates = []
    for distribution in metadata['urls']:
        if distribution['packagetype'] != 'bdist_wheel':
            continue
        tags = parse_wheel_filename(distribution['filename'])[3]
        matching = [RANK[tag] for tag in tags if tag in RANK]
        if matching:
            candidates.append((min(matching), distribution))
    if candidates:
        distribution = min(candidates, key=lambda x: x[0])[1]
        kind = 'wheel'
    else:
        sources = [d for d in metadata['urls'] if d['packagetype'] == 'sdist']
        if not sources:
            raise ValueError(f'No compatible distribution: {name} {version}')
        distribution = sources[0]
        kind = 'sdist_requires_review'
    return dict(name=name, version=version, url=distribution['url'], size=distribution['size'], kind=kind)

if __name__ == '__main__':
    lock=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'requirements.txt'
    entries = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', lock.read_text(), re.M)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        distributions = list(pool.map(inspect, entries))
    total = sum(d['size'] for d in distributions)
    result = dict(packages=len(distributions), selected_distribution_bytes=total, limit_bytes=1500000000,
                  within_budget=total <= 1500000000, distributions=distributions)
    (ROOT/'logs'/'dependency-budget.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'distributions'}, indent=2))
    print('source_distributions', [d for d in distributions if d['kind'] != 'wheel'])
    if not result['within_budget']:
        raise SystemExit('Dependency budget exceeded; installation stopped')
