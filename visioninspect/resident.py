"""Single resident subprocess with idle expiry and per-task hard supervision."""
from __future__ import annotations
import json,math,subprocess,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import psutil,yaml
from .common import ROOT,read_json,write_json,sha256
from .resources import resident_tree_memory,kill_tree


def idle_ttl():
    value=yaml.safe_load((ROOT/'configs/service.yaml').read_text('utf8'))
    if value.get('schema_version')!=1:raise ValueError('Unsupported service config schema')
    ttl=value['idle_ttl_seconds']
    if isinstance(ttl,bool) or not isinstance(ttl,(int,float)) or not math.isfinite(ttl) or ttl<=0:
        raise ValueError('idle_ttl_seconds must be finite and greater than zero')
    return float(ttl)


class ResidentWorker:
    task_timeout_seconds=180
    memory_limit_bytes=12*1024**3
    poll_seconds=.2

    def __init__(self,directory:Path,command,ttl=None):
        self.directory=directory;self.directory.mkdir(parents=True,exist_ok=True)
        self.command_factory=command;self.ttl=idle_ttl() if ttl is None else float(ttl)
        if not math.isfinite(self.ttl) or self.ttl<=0:raise ValueError('Idle TTL must be positive')
        self.lock=threading.Lock();self.shutdown=threading.Event()
        self.process=None;self.log=None;self.generation=None;self.command=None;self.last_idle=None
        self.tracked={}
        self._state=dict(state='not_started',pid=None,inference_pid=None,generation=None,cache={},last_stop_reason=None)
        self.monitor=threading.Thread(target=self._monitor,name='inference-idle-monitor',daemon=True)
        self.monitor.start()

    def _publish(self,**values):
        self._state=dict(self._state,**values)

    def _event(self,event,**values):
        with (self.directory/'events.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),event=event,
                generation=self.generation,**values),ensure_ascii=False)+'\n')

    def status(self):
        state=dict(self._state)
        idle=max(0.,time.monotonic()-state['idle_since']) if state.get('idle_since') is not None else 0.
        return dict(state,idle_ttl_seconds=self.ttl,idle_seconds=idle,
                    expires_in_seconds=max(0.,self.ttl-idle) if state['state']=='idle' else None,
                    task_timeout_seconds=self.task_timeout_seconds,memory_limit_bytes=self.memory_limit_bytes)

    def _start(self,directory):
        self.generation=uuid.uuid4().hex
        self.tracked={}
        self.command=self.command_factory(directory)
        self.log=(self.directory/f'worker-{self.generation}.log').open('ab')
        try:
            self.process=subprocess.Popen(self.command,cwd=ROOT,stdin=subprocess.PIPE,stdout=self.log,
                stderr=subprocess.STDOUT,text=True,encoding='utf8')
        except BaseException:
            self.log.close();self.log=None;raise
        self._publish(state='running',pid=self.process.pid,inference_pid=None,generation=self.generation,
                      idle_since=None,cache={})
        self._event('started',pid=self.process.pid)

    def _stop(self,reason):
        process=self.process
        if process is not None:
            if process.poll() is None:
                try:kill_tree(psutil.Process(process.pid))
                except psutil.NoSuchProcess:pass
            # Retain sampled child identities so descendants are reaped even if
            # their parent exits before this supervisor observes the failure.
            for member in self.tracked.values():
                try:
                    if member.is_running():kill_tree(member)
                except psutil.NoSuchProcess:pass
            process.wait(timeout=10)
            if process.stdin:
                try:process.stdin.close()
                except OSError:pass
            if self.log:self.log.close()
            self._event('stopped',reason=reason,pid=process.pid,exit_code=process.returncode)
        self.process=None;self.log=None;self.last_idle=None;self.tracked={}
        self._publish(state='stopped',pid=None,inference_pid=None,idle_since=None,task_id=None,cache={},last_stop_reason=reason)

    def recycle(self,reason):
        with self.lock:self._stop(reason)

    def _monitor(self):
        while not self.shutdown.wait(min(self.poll_seconds,self.ttl/4)):
            if not self.lock.acquire(blocking=False):continue
            try:
                if self.process is None:continue
                if self.process.poll() is not None:self._stop('worker_exited');continue
                if resident_tree_memory(psutil.Process(self.process.pid),self.tracked)>self.memory_limit_bytes:
                    self._stop('idle_memory_limit');continue
                if self.last_idle is not None and time.monotonic()-self.last_idle>=self.ttl:
                    self._stop('idle_ttl')
            except psutil.NoSuchProcess:self._stop('worker_exited')
            finally:self.lock.release()

    def run(self,directory:Path):
        with self.lock:
            started=time.monotonic();peak=0;failure=None;reply=None;process=None;reused=False
            token=uuid.uuid4().hex
            try:
                if self.shutdown.is_set():raise RuntimeError('interrupted: 服务关闭')
                if self.process is not None and (self.process.poll() is not None or
                        self.last_idle is not None and time.monotonic()-self.last_idle>=self.ttl):
                    self._stop('worker_exited' if self.process.poll() is not None else 'idle_ttl')
                reused=self.process is not None
                if not reused:self._start(directory)
                process=self.process;monitored=psutil.Process(process.pid)
                self.last_idle=None;self._publish(state='running',idle_since=None,task_id=directory.name)
                process.stdin.write(json.dumps(dict(directory=str(directory.resolve()),token=token,
                                                    generation=self.generation))+'\n')
                process.stdin.flush()
                completion=directory/'worker-completion.json'
                while True:
                    peak=max(peak,resident_tree_memory(monitored,self.tracked))
                    if self.shutdown.is_set():raise RuntimeError('interrupted: 服务关闭')
                    if time.monotonic()-started>self.task_timeout_seconds:
                        raise RuntimeError(f'timeout: 任务超过 {self.task_timeout_seconds:g} 秒')
                    if peak>self.memory_limit_bytes:raise RuntimeError('memory_limit: 进程树内存超过限制')
                    if process.poll() is not None:
                        raise RuntimeError(f'worker_failure: 推理子进程退出码 {process.returncode}；详见运行日志')
                    if completion.is_file():
                        reply=read_json(completion)
                        if reply.get('token')!=token:raise RuntimeError('protocol_failure: 任务响应标识不匹配')
                        if not reply.get('success'):raise RuntimeError(reply.get('error','worker_failure'))
                        if reply.get('report_sha256')!=sha256(directory/'report.json'):
                            raise RuntimeError('protocol_failure: 报告校验不匹配')
                        break
                    time.sleep(self.poll_seconds)
                self.last_idle=time.monotonic()
                self._publish(state='idle',idle_since=self.last_idle,task_id=None,
                              inference_pid=reply.get('pid'),cache=reply.get('cache',{}))
                self._event('task_completed',task_id=directory.name,reused=reused)
                return reply
            except BaseException as error:
                failure=f'{type(error).__name__}: {error}'
                self._stop('task_failure')
                raise
            finally:
                write_json(directory/'execution.json',dict(command=self.command,seconds=time.monotonic()-started,
                    peak_process_tree_bytes=peak,memory_measurement='sampled process-tree RSS per task',
                    failure=failure,gpu_memory='See report.outputs.detection.peak_gpu_allocated_bytes and GPU runtime sampling',exit_code=process.returncode if process else None,
                    worker_resident=True,worker_pid=process.pid if process else None,
                    inference_pid=reply.get('pid') if reply else None,generation=self.generation,
                    process_reused=reused,task_protocol_completed=bool(reply and reply.get('success')),
                    task_exit_code=0 if reply and reply.get('success') and failure is None else 1,
                    worker_log=str(self.directory/f'worker-{self.generation}.log')))

    def close(self):
        self.shutdown.set()
        self.monitor.join(timeout=2)
        with self.lock:self._stop('service_shutdown')
