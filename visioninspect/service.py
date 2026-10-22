"""Local queue, SQLite history and one supervised resident inference worker."""
from __future__ import annotations
import json,os,shutil,sqlite3,sys,time,traceback,uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import yaml
from .common import ROOT,project_path,read_json,write_json,sha256,identity
from .resident import ResidentWorker
from .schemas import FinalResult

def timestamp():return datetime.now(timezone.utc).isoformat()

def registry():
    return yaml.safe_load((ROOT/'configs/models.yaml').read_text(encoding='utf8'))['models']

def model_health():
    result=[]
    for category,item in registry().items():
        for model,config in item.items():
            if model=='label':continue
            missing=[str(path) for path in (project_path(config['fit_run'])/'model.ckpt',project_path(config['profile']),project_path(config['references'])) if not path.is_file()]
            result.append(dict(category=category,label=item['label'],model=model,available=not missing,
                               reason='模型资产完整' if not missing else '模型、校准策略或参考索引缺失'))
    return result

class TaskStore:
    def __init__(self,directory:Path|None=None):
        self.root=directory or project_path(os.environ.get('VISIONINSPECT_STATE_DIR','outputs/service'))
        self.root.mkdir(parents=True,exist_ok=True);self.database=self.root/'history.sqlite3'
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='inspection')
        self.closed=False
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, status TEXT, category TEXT, model TEXT, mode TEXT, filename TEXT, state TEXT, error TEXT, seconds REAL, review TEXT)')
            interrupted=[row['id'] for row in db.execute("SELECT id FROM tasks WHERE status IN ('queued','running')")]
        for task_id in interrupted:self.fail(task_id,'interrupted: 服务中断；请重新提交',None)
        self.runtime=ResidentWorker(self.root/'runtime',self.worker_command)

    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.database,timeout=15);connection.row_factory=sqlite3.Row
        try:
            with connection:yield connection
        finally:connection.close()

    def directory(self,task_id):
        if not isinstance(task_id,str) or len(task_id)!=32 or any(c not in '0123456789abcdef' for c in task_id):raise KeyError(task_id)
        return self.root/task_id

    def get(self,task_id,details=True):
        directory=self.directory(task_id)
        with self.connect() as db:row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
        if row is None:raise KeyError(task_id)
        result=dict(row);result['review']=json.loads(result['review']) if result['review'] else None
        if details:
            for name in ('progress','report'):
                path=directory/f'{name}.json'
                result[name]=read_json(path) if path.is_file() else None
        return result

    def list(self,status=None,state=None,query='',limit=50,offset=0):
        conditions=[];arguments=[]
        for name,value in [('status',status),('state',state)]:
            if value:conditions.append(f'{name}=?');arguments.append(value)
        if query:conditions.append('(filename LIKE ? OR id LIKE ?)');arguments.extend([f'%{query}%',f'%{query}%'])
        where=' WHERE '+' AND '.join(conditions) if conditions else ''
        with self.connect() as db:
            total=db.execute('SELECT COUNT(*) FROM tasks'+where,arguments).fetchone()[0]
            rows=db.execute('SELECT * FROM tasks'+where+' ORDER BY created_at DESC LIMIT ? OFFSET ?',arguments+[limit,offset]).fetchall()
        items=[dict(row) for row in rows]
        for item in items:item['review']=json.loads(item['review']) if item['review'] else None
        return dict(items=items,total=total,limit=limit,offset=offset)

    def create(self,body,filename,category,model,mode,vlm=False):
        if self.closed:raise RuntimeError('服务正在关闭')
        with self.connect() as db:
            pending=db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]
        if pending>=8:raise OverflowError('任务队列已满，请稍后再试')
        task_id=uuid.uuid4().hex;directory=self.directory(task_id);directory.mkdir()
        (directory/'input.image').write_bytes(body)
        request=dict(id=task_id,image_path=str(directory/'input.image'),category=category,model=model,mode=mode,vlm=vlm,
                     filename=filename,created_at=timestamp())
        write_json(directory/'request.json',request)
        write_json(directory/'progress.json',dict(stage='queued',trace=[]))
        with self.connect() as db:
            db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (task_id,request['created_at'],timestamp(),'queued',category,model,mode,filename,None,None,None,None))
        self.executor.submit(self.execute,task_id)
        return self.get(task_id)

    def update(self,task_id,**values):
        values['updated_at']=timestamp()
        with self.connect() as db:
            db.execute('UPDATE tasks SET '+','.join(f'{key}=?' for key in values)+' WHERE id=?',list(values.values())+[task_id])

    def worker_command(self,directory):
        return [sys.executable,'-B','-m','visioninspect.worker',str(self.root.resolve()),
                '--resident','--parent-pid',str(os.getpid())]

    def fail(self,task_id,failure,seconds):
        """Publish an authoritative manual-review report before exposing terminal failure."""
        directory=self.directory(task_id);report_path=directory/'report.json';original={}
        if report_path.is_file():
            backup=directory/'worker-report-unaccepted.json'
            if not backup.exists():shutil.copy2(report_path,backup)
            try:original=read_json(report_path)
            except (OSError,ValueError):original={}
            if not isinstance(original,dict):original={}
        try:progress=read_json(directory/'progress.json')
        except (OSError,ValueError):progress={}
        if not isinstance(progress,dict):progress={}
        with self.connect() as db:
            metadata=dict(db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone())
        paths=original.get('result',{}).get('evidence_paths',[]) if isinstance(original.get('result'),dict) else []
        if not isinstance(paths,list):paths=[]
        result=FinalResult(state='manual_review',evidence_paths=[p for p in paths if isinstance(p,str)],
                           rule_version='supervisor-failure-v1',reasons=[failure]).model_dump()
        trace=original.get('trace') or progress.get('trace',[])
        if not isinstance(trace,list):trace=[]
        report=dict(original,schema_version=1,task_id=task_id,filename=metadata['filename'],category=metadata['category'],
                    mode=metadata['mode'],created_at=metadata['created_at'],result=result,
                    errors=[failure],outputs=original.get('outputs',{}) if isinstance(original.get('outputs',{}),dict) else {},seconds=seconds,
                    tool_calls=len(trace),vlm_calls=original.get('vlm_calls',0),monetary_cost=0,trace=trace,
                    stop_reason='supervisor_failure',supervisor=dict(worker_report_accepted=False,failure=failure),
                    notice='监督器未接受本次推理结果，必须人工复核；原始工作进程输出仅供排查。')
        write_json(report_path,report)
        self.update(task_id,status='failed',state='manual_review',error=failure,seconds=seconds)

    def execute(self,task_id):
        directory=self.directory(task_id);started=time.perf_counter()
        self.update(task_id,status='running')
        try:
            reply=self.runtime.run(directory)
            report=read_json(directory/'report.json')
            result=FinalResult.model_validate(report['result'])
            if reply.get('recycle'):self.runtime.recycle('task_tool_failure')
            execution=read_json(directory/'execution.json');execution['worker_report_accepted']=True
            write_json(directory/'execution.json',execution)
            self.update(task_id,status='completed',state=result.state,seconds=time.perf_counter()-started)
        except Exception as error:
            failure=f'{type(error).__name__}: {error}'
            (directory/'supervisor-error.log').write_text(traceback.format_exc(),encoding='utf8')
            self.runtime.recycle('task_failure')
            execution_path=directory/'execution.json'
            if execution_path.is_file():
                execution=read_json(execution_path)
                execution.update(failure=failure,worker_report_accepted=False,task_exit_code=1)
                write_json(execution_path,execution)
            self.fail(task_id,failure,time.perf_counter()-started)

    def review(self,task_id,decision,note):
        task=self.get(task_id)
        if task['status'] not in ('completed','failed'):raise ValueError('任务结束后才能保存复核')
        # Separate append-only review events; never rewrite the original model report.
        value=dict(decision=decision,note=note,recorded_at=timestamp(),scope='研究复核记录，不构成产线放行授权')
        directory=self.directory(task_id)
        with (directory/'reviews.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        self.update(task_id,review=json.dumps(value,ensure_ascii=False));return value

    def close(self):
        self.closed=True
        self.runtime.close()
        self.executor.shutdown(wait=True,cancel_futures=True)
        with self.connect() as db:
            pending=[row['id'] for row in db.execute("SELECT id FROM tasks WHERE status IN ('queued','running')")]
        for task_id in pending:self.fail(task_id,'interrupted: 服务关闭；请重新提交',None)
