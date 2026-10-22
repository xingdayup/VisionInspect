"""Check public image manifests without Docker, image-layer downloads or local login."""
import hashlib,json,re,sys,urllib.error,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import write_json
ACCEPT=', '.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json',
                 'application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json'])
SOURCES=[('node:24.22.0-bookworm-slim','https://registry-1.docker.io','library/node','24.22.0-bookworm-slim'),
         ('python:3.12.12-slim-bookworm','https://registry-1.docker.io','library/python','3.12.12-slim-bookworm'),
         ('ghcr.io/astral-sh/uv:0.11.7','https://ghcr.io','astral-sh/uv','0.11.7')]
def metadata(url,headers=None):
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=20) as response:
        body=response.read(2*1024**2+1);digest=response.headers.get('Docker-Content-Digest')
    if len(body)>2*1024**2:raise ValueError('Unexpected metadata size')
    if digest and digest!='sha256:'+hashlib.sha256(body).hexdigest():raise ValueError('Manifest checksum mismatch')
    return json.loads(body),digest
result=[]
dockerfile=(ROOT/'Dockerfile').read_text(encoding='utf8')
for image,host,repository,tag in SOURCES:
    source_match=re.search(r'^FROM '+re.escape(image)+r'(?:@(sha256:[a-f0-9]{64}))?(?:\s|$)',dockerfile,re.M)
    if not source_match:raise ValueError('Dockerfile differs from source audit')
    headers={'Accept':ACCEPT};url=f'{host}/v2/{repository}/manifests/{tag}'
    try:
        try:manifest,digest=metadata(url,headers)
        except urllib.error.HTTPError as error:
            if error.code!=401:raise
            challenge=error.headers.get('WWW-Authenticate','')
            values=dict(re.findall(r'(\w+)="([^"]+)"',challenge));realm=values['realm']
            if not any(realm.startswith(allowed) for allowed in ('https://auth.docker.io/','https://ghcr.io/')):raise ValueError('Unexpected registry auth host')
            token_url=realm+'?'+urllib.parse.urlencode({key:values[key] for key in ('service','scope') if key in values})
            token,_=metadata(token_url)
            headers['Authorization']='Bearer '+(token.get('token') or token['access_token'])
            manifest,digest=metadata(url,headers)
        index_digest=digest
        if source_match.group(1) and source_match.group(1)!=index_digest:raise ValueError('Dockerfile digest differs from registry metadata')
        if 'manifests' in manifest:
            candidates=[item for item in manifest['manifests'] if item.get('platform',{}).get('os')=='linux' and item.get('platform',{}).get('architecture')=='amd64']
            if len(candidates)!=1:raise ValueError('Ambiguous linux/amd64 image')
            manifest,digest=metadata(f"{host}/v2/{repository}/manifests/{candidates[0]['digest']}",headers)
        layers=manifest.get('layers',[])
        if not layers:raise ValueError('Image layers missing from manifest')
        record=dict(image=image,verified=True,index_digest=index_digest,linux_amd64_digest=digest,
                    compressed_layer_bytes=sum(layer['size'] for layer in layers),layers=len(layers))
    except Exception as error:record=dict(image=image,verified=False,error=f'{type(error).__name__}: {error}')
    result.append(record);print(record,flush=True)
write_json(ROOT/'logs/container-source-preflight.json',dict(metadata_only=True,images=result,
           note='No layers pulled; compressed sizes do not prove installed disk use or a successful Docker build'))
if not all(item['verified'] for item in result):raise SystemExit(1)
