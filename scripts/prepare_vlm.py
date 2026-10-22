"""Explicit pinned local-only VLM preparation, with a preflight cap and file verification."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_assets import fetch
from visioninspect.common import ROOT,read_json,write_json

REVISION='a7da5b986cb59b408707209984f360a5f4ad7e47'
NAMES={'README.md','added_tokens.json','chat_template.json','config.json','generation_config.json','merges.txt',
       'model.safetensors','preprocessor_config.json','processor_config.json','special_tokens_map.json',
       'tokenizer.json','tokenizer_config.json','vocab.json'}

def inspect(metadata):
    if metadata['sha']!=REVISION or metadata['id']!='HuggingFaceTB/SmolVLM-500M-Instruct':raise ValueError('Model revision mismatch')
    if metadata['cardData']['license']!='apache-2.0' or metadata['gated']:raise ValueError('License or access changed')
    files=[item for item in metadata['siblings'] if item['rfilename'] in NAMES]
    if {item['rfilename'] for item in files}!=NAMES:raise ValueError('Required model files missing')
    total=sum(item['size'] for item in files)
    if total>1050000000:raise ValueError('VLM asset download budget exceeded')
    write_json(ROOT/'logs/vlm-download-preflight.json',dict(bytes=total,ram_estimate_gib=[3,5],maximum_process_gib=12,
               revision=metadata['sha'],license='Apache-2.0',files=files,device='CPU FP32',paid_api=False))
    return files,total

def main(preflight=False):
    if preflight:
        url=f'https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-500M-Instruct/revision/{REVISION}?blobs=true'
        with urllib.request.urlopen(url,timeout=30) as response:body=response.read(1024**2+1)
        if len(body)>1024**2:raise ValueError('Metadata too large')
        metadata=json.loads(body);files,total=inspect(metadata)
        write_json(ROOT/'logs/smolvlm-meta.json',metadata)
        print('PREFLIGHT_ONLY',len(files),total,'bytes; Apache-2.0; no model downloaded');return
    metadata_path=ROOT/'logs/smolvlm-meta.json'
    if not metadata_path.is_file():raise FileNotFoundError('Run prepare_vlm.py --preflight and review metadata first')
    metadata=read_json(metadata_path);files,total=inspect(metadata);results=[]
    for item in files:
        name=item['rfilename'];url=f"https://huggingface.co/{metadata['id']}/resolve/{metadata['sha']}/{name}"
        result=fetch(url,ROOT/'weights/smolvlm-500m'/name,item['size'],item.get('lfs',{}).get('sha256'))
        if 'lfs' not in item:
            body=(ROOT/'weights/smolvlm-500m'/name).read_bytes()
            if hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()!=item['blobId']:raise ValueError('Git blob checksum mismatch')
        results.append(result);print(name,item['size'],'verified',flush=True)
    write_json(ROOT/'weights/smolvlm-500m/source.json',dict(model=metadata['id'],revision=metadata['sha'],license='Apache-2.0',files=results))
    print('VLM_ASSETS_READY',total,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();main(args.preflight)
