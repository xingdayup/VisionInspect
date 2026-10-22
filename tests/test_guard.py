import subprocess,sys,json,time
from visioninspect.common import ROOT

def test_supervisor_terminates_real_child_and_keeps_failure_log(tmp_path):
    record=tmp_path/'guard.json';started=time.monotonic()
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/guard.py'),'--seconds','1.5','--record',str(record),
                           '--',sys.executable,'-u','-c','import time; print("real test child started",flush=True); time.sleep(30)'],
                          cwd=ROOT,capture_output=True,timeout=15)
    assert result.returncode==124 and time.monotonic()-started<15
    execution=json.loads(record.read_text(encoding='utf8'))
    assert execution['failure']=='wall-clock timeout' and execution['peak_process_tree_bytes']>0
    assert 'real test child started' in record.with_suffix('.child.log').read_text()
