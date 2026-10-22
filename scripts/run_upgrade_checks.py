"""Finite commands with exact argv, exit codes, logs, packages and elapsed time."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    commands=[[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--junitxml=logs/upgrade-tests.xml'],
              [str(Path(os.environ.get('ProgramFiles','C:/Program Files'))/'nodejs/npm.cmd'),'--prefix','frontend','run','build'] if sys.platform=='win32' else ['npm','--prefix','frontend','run','build'],
              [sys.executable,'-B','scripts/check_git_content.py']]
    results=[]
    for index,command in enumerate(commands,1):
        log=logs/f'upgrade-check-{index}.log';started=time.monotonic()
        with log.open('w',encoding='utf8') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        result=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,log=str(log.relative_to(ROOT)))
        results.append(result);print(json.dumps(result),flush=True)
        if process.returncode:break
    report=dict(commands=results,passed=len(results)==len(commands) and all(r['exit_code']==0 for r in results),
        packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']),
        gpu_model_inference_verified=False,semantic_accuracy_evaluated=False)
    (logs/'upgrade-checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
