"""Inspect exact new wheels before installing the Agent dependency lock."""
import concurrent.futures
import importlib.metadata
import json
import re
import urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def main():
    pins = dict(re.findall(r'^([\w-]+)==([^\s\\]+)', (ROOT/'requirements-agent.txt').read_text(), re.M))
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}
    changed = {n:v for n,v in pins.items() if n in installed and installed[n] != v}
    if changed:
        raise RuntimeError(f'Existing versions would change: {changed}')
    tags = set(sys_tags())
    def inspect(item):
        name, version = item
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as response:
            metadata = json.load(response)
        wheels = []
        for file in metadata['urls']:
            if file['filename'].endswith('.whl') and parse_wheel_filename(file['filename'])[3] & tags:
                wheels.append(file)
        if not wheels:
            raise RuntimeError(f'No compatible wheel: {name} {version}')
        file = min(wheels, key=lambda f:f['size'])
        return dict(name=name, version=version, filename=file['filename'], bytes=file['size'], sha256=file['digests']['sha256'])
    additions = [(n,v) for n,v in pins.items() if n not in installed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(inspect, additions))
    report = dict(existing_packages_changed=changed, files=files, download_bytes=sum(f['bytes'] for f in files), maximum_bytes=100*1024**2)
    (ROOT/'logs/agent-dependency-budget.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}, indent=2))
    if report['download_bytes'] > report['maximum_bytes']:
        raise RuntimeError('100 MiB incremental budget exceeded')


if __name__=='__main__':main()
