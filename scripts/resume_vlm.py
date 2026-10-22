"""Resume the same pinned weight with bounded verified HTTP ranges. No alternative source."""
import concurrent.futures,json,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import sha256,read_json,write_json

size=1015025832;expected='d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2'
directory=ROOT/'weights/smolvlm-500m';partial=directory/'model.safetensors.part';destination=directory/'model.safetensors'
url='https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/resolve/a7da5b986cb59b408707209984f360a5f4ad7e47/model.safetensors'
state_path=directory/'ranges.json'
if destination.is_file():
    assert sha256(destination)==expected;print('Verified existing weight');raise SystemExit(0)
state=read_json(state_path) if state_path.exists() else dict(prefix_bytes=partial.stat().st_size if partial.exists() else 0,completed=[])
prefix=state['prefix_bytes']
if prefix>=size:raise ValueError('Unexpected prefix size')
with partial.open('a+b') as stream:stream.truncate(size)
write_json(state_path,state)
chunk=8*1024**2;parts=[(start,min(size-1,start+chunk-1)) for start in range(prefix,size,chunk)]
parts=[part for part in parts if list(part) not in state['completed']];started=time.monotonic()
def download(part):
    start,end=part;position=start
    with partial.open('r+b',buffering=0) as target:
        for attempt in range(4):
            try:
                if time.monotonic()-started>1800:raise TimeoutError('VLM continuation exceeded 30 minutes')
                request=urllib.request.Request(url,headers={'Range':f'bytes={position}-{end}'})
                with urllib.request.urlopen(request,timeout=20) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {position}-{end}/{size}':raise ValueError('HTTP range mismatch')
                    target.seek(position)
                    while position<=end:
                        block=response.read(min(64*1024,end-position+1))
                        if not block:raise OSError('Truncated range')
                        target.write(block);position+=len(block)
                return part
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    raise RuntimeError('range failed')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for part in pool.map(download,parts):
        state['completed'].append(list(part));write_json(state_path,state)
        done=prefix+sum(end-start+1 for start,end in state['completed'])
        print(f'verified transport ranges {done}/{size} bytes',flush=True)
if sha256(partial)!=expected:raise ValueError('Final weight SHA256 mismatch; unverified partial retained')
partial.replace(destination);print('VLM_WEIGHT_SHA256_VERIFIED',expected,flush=True)
