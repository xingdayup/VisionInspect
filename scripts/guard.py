"""Run a finite local command with a wall-clock and full-process-tree memory cap."""
import argparse,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import write_json
from visioninspect.resources import tree_memory,kill_tree
parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,required=True);parser.add_argument('--record',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
if not command:parser.error('command missing')
started=time.perf_counter();peaks={};failure=None
child_log=args.record.with_suffix('.child.log');child_log.parent.mkdir(parents=True,exist_ok=True)
with child_log.open('wb') as log:
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
print('CHILD_LOG',child_log,flush=True)
while process.poll() is None:
    memory=tree_memory(psutil.Process(process.pid),peaks)
    if time.perf_counter()-started>args.seconds:failure='wall-clock timeout'
    if memory>12*1024**3:failure='12 GiB process-tree memory budget exceeded'
    if failure:kill_tree(psutil.Process(process.pid));break
    time.sleep(.2)
code=process.wait()
write_json(args.record,dict(command=command,exit_code=code,seconds=time.perf_counter()-started,
                           peak_process_tree_bytes=sum(peaks.values()),failure=failure,gpu_memory='not applicable',child_log=str(child_log)))
if failure:print('SUPERVISOR_TERMINATED',failure,flush=True)
raise SystemExit(124 if failure else code)
