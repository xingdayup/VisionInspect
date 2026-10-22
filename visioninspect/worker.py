"""Offline inspections with request-scoped tools and process-scoped model assets."""
import argparse,contextlib,json,os,platform,sys,threading,traceback
from pathlib import Path
import yaml
from .common import ROOT,configure_environment,offline,project_path,read_json,write_json,identity,sha256
from .agent import InspectionAgent
from .service import registry
from .tools import ToolKit
from .model_cache import ModelCache

def main(directory,cache=None,worker_runtime=None):
    request=read_json(directory/'request.json')
    snapshot=request.get('snapshot') or {}
    policy=snapshot.get('policy') or yaml.safe_load((ROOT/'configs/agent.yaml').read_text(encoding='utf8'))
    policy['vlm']['enabled']=request['vlm']
    registration=snapshot.get('registration') or registry()[request['category']][request['model']]
    agent_review=request['mode']=='agent_review'
    configure_environment()
    os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    with offline(allow_loopback_ports=(8001,) if agent_review else ()):
        vlm=None;vlm_loaded_before_task=False
        if request['vlm'] and not agent_review:
            if cache is not None:
                vlm=cache.get_vlm(project_path(policy['vlm']['directory']),policy['vlm']['max_new_tokens'])
            else:
                from .vlm import LocalVLM
                vlm=LocalVLM(project_path(policy['vlm']['directory']),policy['vlm']['max_new_tokens'])
            vlm_loaded_before_task=vlm.model is not None
        tools=ToolKit(registration,policy,directory,vlm,cache=cache)
        if snapshot.get('profile_digest') and tools.profile['digest']!=snapshot['profile_digest']:
            raise ValueError('模型策略与提交任务时的快照不一致')
        progress=lambda value:write_json(directory/'progress.json',value)
        if agent_review:
            from .review_agent import ReviewAgent
            from .vllm_client import VLLMClient
            from .vllm_runtime import gpu_config
            gpu=snapshot.get('gpu') or gpu_config()
            report=ReviewAgent(tools,policy,VLLMClient(gpu,request.get('gpu_startup_error')),progress,gpu['max_model_calls']).run(
                request['image_path'],request['category'])
            from .review_contracts import REVIEW_PROMPT
            report['review_policy']=dict(version='agent-review-v1',max_model_calls=gpu['max_model_calls'],
                request_timeout_seconds=gpu['request_timeout_seconds'],model=gpu['model'],
                prompt_digest=identity(dict(prompt=REVIEW_PROMPT)),
                graph_source_sha256=sha256(ROOT/'visioninspect/review_agent.py'),
                gpu_startup_error=request.get('gpu_startup_error'))
        else:
            report=InspectionAgent(tools,policy,progress).run(request['image_path'],request['category'],request['mode'])
        report.update(task_id=request['id'],filename=request['filename'],created_at=request['created_at'],
                      runtime=dict(python=platform.python_version(),platform=platform.platform(),device='CPU',precision='FP32'),
                      policy=policy,policy_digest=identity(policy),checkpoint_sha256=tools.profile['checkpoint_sha256'],
                      manifest_digest=tools.profile['manifest_digest'])
        if worker_runtime is not None:
            report['runtime'].update(worker_runtime,model_cache=cache.status(),
                                     vlm_loaded_before_task=vlm_loaded_before_task)
        write_json(directory/'report.json',report)
        return report


def resident(root,parent_pid=None):
    configure_environment();root=root.resolve();cache=ModelCache();sequence=0
    # EOF also ends the loop. This watchdog bounds orphan lifetime if the API dies mid-task.
    if parent_pid is not None:
        import psutil
        try:parent=psutil.Process(parent_pid);created=parent.create_time()
        except psutil.NoSuchProcess:return
        def watch_parent():
            while True:
                threading.Event().wait(1)
                try:
                    if not parent.is_running() or parent.create_time()!=created:os._exit(1)
                except psutil.NoSuchProcess:os._exit(1)
        threading.Thread(target=watch_parent,name='api-parent-watch',daemon=True).start()
    sys.stdin.reconfigure(encoding='utf8')
    for line in sys.stdin:
        message=json.loads(line);directory=Path(message['directory']).resolve()
        if directory.parent!=root or len(directory.name)!=32 or any(c not in '0123456789abcdef' for c in directory.name):
            raise ValueError('Worker task directory is outside its state root')
        sequence+=1;result=None
        with (directory/'worker.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                print(f'Resident inference PID {os.getpid()}, task sequence {sequence}',flush=True)
                report=main(directory,cache,dict(worker_pid=os.getpid(),generation=message['generation'],
                                               task_sequence=sequence,resident=True))
                isolated_review_failure=bool(report.get('mode')=='agent_review' and
                    (report.get('agent_review') or {}).get('status')=='failed' and
                    (report.get('outputs') or {}).get('detection'))
                result=dict(token=message['token'],success=True,report_sha256=sha256(directory/'report.json'),
                            pid=os.getpid(),cache=cache.status(),recycle=bool(report['errors']) and not isolated_review_failure)
            except Exception as error:
                traceback.print_exc()
                result=dict(token=message['token'],success=False,error=f'{type(error).__name__}: {error}',pid=os.getpid())
        write_json(directory/'worker-completion.json',result)
        if not result['success']:return


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--resident',action='store_true');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.resident:resident(args.directory,args.parent_pid)
    else:main(args.directory.resolve())
