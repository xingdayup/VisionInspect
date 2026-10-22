"""Read-only Linux/Python wheel and resource budget; no package installation."""
import argparse
import concurrent.futures
from http.client import IncompleteRead
import html.parser
import json
import platform
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]


def fetch_metadata(url,method='GET'):
    """Retry transient connection AND response-body failures, keeping TLS checks."""
    label=urllib.parse.urlsplit(url)
    for attempt in range(1,4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method=method),timeout=90) as response:
                if method=='HEAD':return int(response.headers['Content-Length'])
                body=response.read(8*1024**2+1)
                if len(body)>8*1024**2:raise ValueError('Metadata response exceeds 8 MiB')
                return body
        except urllib.error.HTTPError as failure:
            failure.close()
            if failure.code not in {408,429,500,502,503,504}:raise
            error=failure
        except urllib.error.URLError as failure:
            if isinstance(failure.reason,ssl.SSLCertVerificationError):raise
            error=failure
        except (TimeoutError,ConnectionError,IncompleteRead) as failure:
            error=failure
        if attempt==3:
            raise RuntimeError(f'Metadata request failed after 3 attempts: {method} {label.hostname}{label.path}') from error
        delay=2**attempt
        print(f'[retry {attempt}/3] {method} {label.hostname}{label.path}: '
              f'{type(error).__name__}; retrying in {delay}s',flush=True)
        time.sleep(delay)


def main(argv=None):
    arguments=argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('--workers',type=int,choices=range(1,5),default=2)
    args=arguments.parse_args(argv)
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
            parser=Links();parser.feed(fetch_metadata(f'https://download.pytorch.org/whl/cpu/{name}/').decode())
            files=[dict(filename=urllib.parse.unquote(url.split('/')[-1].split('#')[0]),url=url) for url in parser.links]
        else:
            files=json.loads(fetch_metadata(f'https://pypi.org/pypi/{name}/{version}/json'))['urls']
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
            size=fetch_metadata(file['url'],method='HEAD')
        return dict(name=name,version=version,filename=file['filename'],bytes=size,url=file['url'])
    entries=re.findall(r'^([\w-]+)==([^\s\\]+)',(ROOT/'requirements-agent-linux.txt').read_text(),re.M)
    if not entries:raise RuntimeError('No pinned dependencies found; installation stopped')
    print(f'Checking {len(entries)} pinned packages with {args.workers} workers; '
          'network timeout 90s, at most 3 attempts per request',flush=True)
    results={};pool=concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures={pool.submit(inspect,item):index for index,item in enumerate(entries)}
    try:
        for future in concurrent.futures.as_completed(futures):
            index=futures[future];name,version=entries[index]
            try:results[index]=future.result()
            except Exception as failure:
                raise RuntimeError(f'Preflight failed for {name}=={version}; no packages installed') from failure
            print(f'[{len(results)}/{len(entries)}] {name}=={version}: {results[index]["bytes"]} bytes',flush=True)
    finally:
        # Do not keep starting queued HTTP requests after a package has failed.
        pool.shutdown(wait=True,cancel_futures=True)
    files=[results[index] for index in range(len(entries))]
    report=dict(python=sys.version,executable=sys.executable,files=files,download_bytes=sum(f['bytes'] for f in files),
                limit_bytes=1_500_000_000,free_disk_bytes=shutil.disk_usage(ROOT).free,minimum_free_disk_bytes=15*1024**3,
                packages_installed=False)
    report['passed']=report['download_bytes']<=report['limit_bytes'] and report['free_disk_bytes']>=report['minimum_free_disk_bytes']
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'logs/pai-app-preflight.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    if not report['passed']:raise RuntimeError('Application resource budget exceeded; installation stopped')


if __name__=='__main__':main()
