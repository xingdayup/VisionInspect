"""Read-only Linux/Python wheel and resource budget; no package installation."""
import concurrent.futures
import html.parser
import json
import platform
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]


def main():
    if sys.platform!='linux' or sys.version_info[:2]!=(3,12) or platform.machine()!='x86_64':
        raise RuntimeError('PAI application requires Linux x86_64 and existing Python 3.12')
    tags={tag:index for index,tag in enumerate(sys_tags())}
    class Links(html.parser.HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':
                href=dict(attrs).get('href')
                if href:self.links.append(href)
    def inspect(item):
        name,version=item;candidates=[]
        if name in {'torch','torchvision'}:
            with urllib.request.urlopen(f'https://download.pytorch.org/whl/cpu/{name}/',timeout=30) as response:
                parser=Links();parser.feed(response.read(8*1024**2).decode())
            files=[dict(filename=urllib.parse.unquote(url.split('/')[-1].split('#')[0]),url=url) for url in parser.links]
        else:
            with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as response:files=json.load(response)['urls']
        for file in files:
            if not file['filename'].endswith('.whl'):continue
            parsed=parse_wheel_filename(file['filename'])
            if str(parsed[1])!=version:continue
            matching=[tags[t] for t in parsed[3] if t in tags]
            if matching:candidates.append((min(matching),file))
        if not candidates:raise RuntimeError(f'No compatible binary wheel: {name} {version}')
        file=min(candidates,key=lambda item:item[0])[1]
        size=file.get('size')
        if size is None:
            with urllib.request.urlopen(urllib.request.Request(file['url'],method='HEAD'),timeout=30) as response:
                size=int(response.headers['Content-Length'])
        return dict(name=name,version=version,filename=file['filename'],bytes=size,url=file['url'])
    entries=re.findall(r'^([\w-]+)==([^\s\\]+)',(ROOT/'requirements-agent-linux.txt').read_text(),re.M)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(inspect,entries))
    report=dict(python=sys.version,executable=sys.executable,files=files,download_bytes=sum(f['bytes'] for f in files),
                limit_bytes=1_500_000_000,free_disk_bytes=shutil.disk_usage(ROOT).free,minimum_free_disk_bytes=15*1024**3,
                packages_installed=False)
    report['passed']=report['download_bytes']<=report['limit_bytes'] and report['free_disk_bytes']>=report['minimum_free_disk_bytes']
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'logs/pai-app-preflight.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    if not report['passed']:raise RuntimeError('Application resource budget exceeded; installation stopped')


if __name__=='__main__':main()
