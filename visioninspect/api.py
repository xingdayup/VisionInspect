"""REST API for the local research workstation. Start one server worker only."""
from contextlib import asynccontextmanager
from io import BytesIO
import re
from pathlib import Path
from fastapi import FastAPI,File,Form,HTTPException,Query,Request,UploadFile
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field,field_validator
from PIL import Image,UnidentifiedImageError
from .common import ROOT,project_path,read_json,configure_environment
from .schemas import TOOL_CONTRACTS
from .service import TaskStore,model_health,registry

class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    decision: str=Field(pattern=r'^(normal_candidate|anomaly_candidate|manual_review)$')
    note: str=Field(min_length=1,max_length=2000)
    @field_validator('note')
    @classmethod
    def meaningful_note(cls,value):
        if not value.strip():raise ValueError('复核说明不能为空')
        return value.strip()

class UploadBodyLimit:
    """Cap multipart bodies before parser spooling; all temp files remain in the project."""
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method']!='POST' or scope['path']!='/api/tasks':
            return await self.app(scope,receive,send)
        limit=10*1024**2+64*1024;chunks=[];total=0
        headers=dict(scope.get('headers',[]))
        try:length=int(headers.get(b'content-length',b'0'))
        except ValueError:length=0
        if length>limit:return await JSONResponse({'detail':'上传请求超过 10 MiB 图像上限'},status_code=413)(scope,receive,send)
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            total+=len(message.get('body',b''))
            if total>limit:return await JSONResponse({'detail':'上传请求超过 10 MiB 图像上限'},status_code=413)(scope,receive,send)
            chunks.append(message.get('body',b''))
            if not message.get('more_body',False):break
        body=b''.join(chunks);del chunks;delivered=False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered=True;return dict(type='http.request',body=body,more_body=False)
            return await receive()
        return await self.app(scope,replay,send)

def create_app(state_directory:Path|None=None):
    configure_environment()
    @asynccontextmanager
    async def lifespan(application):
        application.state.store=TaskStore(state_directory)
        try:yield
        finally:application.state.store.close()
    application=FastAPI(title='VisionInspect-Agent',version='0.3.0',lifespan=lifespan)
    application.add_middleware(UploadBodyLimit)
    def store(request):
        if not hasattr(request.app.state,'store'):request.app.state.store=TaskStore(state_directory)
        return request.app.state.store
    def get_task(request,task_id):
        try:return store(request).get(task_id)
        except KeyError:raise HTTPException(404,detail='任务不存在')
        except (OSError,ValueError):raise HTTPException(409,detail='任务记录损坏或无法读取，请检查日志或重新提交')

    @application.get('/health')
    def health():
        models=model_health()
        return dict(status='ok',phase='integrated',inference_service=any(m['available'] for m in models),
                    models=models,device='CPU',precision='FP32',worker_concurrency=1,paid_api=False,
                    vlm_available=(ROOT/'weights/smolvlm-500m/source.json').is_file(),
                    weights_present=(ROOT/'weights/resnet18.a1_in1k/model.safetensors').is_file(),
                    dataset_manifest_present=(ROOT/'data/manifests/bottle.json').is_file())

    @application.get('/ready')
    def ready():
        value=health()
        return JSONResponse(value,status_code=200 if value['inference_service'] else 503)

    @application.get('/api/models')
    def models():return model_health()

    @application.get('/api/runtime')
    def runtime(request:Request):return store(request).runtime.status()

    @application.get('/api/tools/schemas')
    def schemas():
        return {name:dict(input=a.model_json_schema(),output=b.model_json_schema(),service_implemented=True)
                for name,(a,b) in TOOL_CONTRACTS.items()}

    @application.post('/api/tasks',status_code=202)
    async def create_task(request:Request,file:UploadFile=File(...),category:str=Form('bottle'),
                          model:str=Form('patchcore'),mode:str=Form('dynamic'),vlm:bool=Form(False)):
        if mode not in {'baseline','fixed','dynamic'}:raise HTTPException(422,detail='不支持的工作流')
        if mode=='baseline':vlm=False
        if category not in registry() or model not in registry()[category] or model=='label':
            raise HTTPException(422,detail='未注册的产品类别或模型')
        # Stream with an explicit cap; reject before model inference and do not use client paths.
        content=bytearray()
        while block:=await file.read(256*1024):
            content.extend(block)
            if len(content)>10*1024**2:raise HTTPException(413,detail='图像超过 10 MiB 上限')
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format not in {'PNG','JPEG','WEBP'}:raise ValueError('只支持 PNG、JPEG、WebP')
                if image.width*image.height>16777216:raise ValueError('图像超过 1600 万像素上限')
                image.verify()
        except (OSError,ValueError,Image.DecompressionBombError) as error:
            raise HTTPException(422,detail=f'图像不可用：{error}')
        selected=next(m for m in model_health() if m['category']==category and m['model']==model)
        if not selected['available']:raise HTTPException(503,detail=selected['reason'])
        if vlm and not (ROOT/'weights/smolvlm-500m/source.json').is_file():raise HTTPException(503,detail='本地 VLM 权重缺失，请关闭候选解释或准备资产')
        filename=(file.filename or 'image').replace('\\','/').split('/')[-1][:160]
        try:return store(request).create(bytes(content),filename,category,model,mode,vlm)
        except OverflowError as error:raise HTTPException(429,detail=str(error))

    @application.get('/api/tasks')
    def tasks(request:Request,status:str|None=None,state:str|None=None,q:str='',limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0)):
        return store(request).list(status,state,q[:160],limit,offset)

    @application.get('/api/tasks/{task_id}')
    def task(request:Request,task_id:str):return get_task(request,task_id)

    @application.get('/api/tasks/{task_id}/report')
    def report(request:Request,task_id:str):
        value=get_task(request,task_id)
        if value['status'] not in {'completed','failed'}:raise HTTPException(409,detail='报告尚未生成')
        if not isinstance(value['report'],dict):raise HTTPException(410,detail='报告文件已缺失，无法导出；请重新检测')
        result=dict(value['report'],human_review=value['review'])
        return JSONResponse(result,headers={'Content-Disposition':f'attachment; filename="inspection-{task_id}.json"'})

    @application.post('/api/tasks/{task_id}/review')
    def review(request:Request,task_id:str,value:ReviewRequest):
        get_task(request,task_id)
        try:return store(request).review(task_id,value.decision,value.note.strip())
        except ValueError as error:raise HTTPException(409,detail=str(error))

    @application.get('/api/tasks/{task_id}/artifacts/{filename}')
    def artifact(request:Request,task_id:str,filename:str):
        get_task(request,task_id)
        if not re.fullmatch(r'(original|inspection_heatmap|inspection_overlay|reference_[12]|region_[1-4])\.png|inspection\.npy',filename):
            raise HTTPException(404,detail='证据文件不存在')
        path=store(request).directory(task_id)/'evidence'/filename
        if not path.is_file():raise HTTPException(404,detail='证据文件尚未生成或已缺失')
        return FileResponse(path,media_type='image/png' if filename.endswith('.png') else 'application/octet-stream')

    @application.get('/api/runs')
    def runs():
        result=[]
        for directory in sorted((ROOT/'outputs').iterdir(),reverse=True):
            if not (directory/'status.json').is_file():continue
            status=read_json(directory/'status.json')
            result.append(dict(id=directory.name,command=status['command'],exit_code=status['exit_code'],seconds=status['seconds'],metrics_available=(directory/'metrics.json').is_file()))
        return result

    @application.get('/api/runs/{run_id}/metrics')
    def metrics(run_id:str):
        if not re.fullmatch(r'\d{8}-\d{6}-[a-z-]+-[a-f0-9]{6}',run_id):raise HTTPException(400,detail='Invalid run ID')
        path=ROOT/'outputs'/run_id/'metrics.json'
        if not path.is_file():raise HTTPException(404,detail='No completed metrics for this run')
        return read_json(path)

    dist=ROOT/'frontend/dist'
    if dist.is_dir():application.mount('/',StaticFiles(directory=dist,html=True),name='workstation')
    return application

app=create_app()
