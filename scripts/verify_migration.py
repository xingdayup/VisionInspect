"""Clone a bundle into a fresh local directory and run real relocated CPU checks."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args();package=args.package.resolve()
    manifest=json.loads((package/'transfer-manifest.json').read_text('utf8'))
    checkout=ROOT/'.cache/migration-qa'/f'{time.strftime("%Y%m%d-%H%M%S")}-{manifest["commit"][:8]}'
    checkout.parent.mkdir(parents=True,exist_ok=True)
    commands=[(['git','clone',str(package/'visioninspect.bundle'),str(checkout)],ROOT),
              (['git','bundle','verify',str(package/'visioninspect.bundle')],checkout),
              ([sys.executable,'-B','scripts/install_assets.py','--package',str(package)],checkout),
              ([sys.executable,'-B','-c','from pathlib import Path; from visioninspect.common import ROOT; assert ROOT.resolve()==Path.cwd().resolve(); print(ROOT)'],checkout),
              ([sys.executable,'-B','-m','visioninspect.baseline','check-data','--config','configs/patchcore_cpu.yaml'],checkout),
              ([sys.executable,'-B','scripts/verify_upgrade_cpu.py'],checkout)]
    results=[];env=dict(os.environ,VISIONINSPECT_RELOCATED_ASSETS='1',VISIONINSPECT_GPU_ENABLED='0',PYTHONDONTWRITEBYTECODE='1')
    for index,(command,cwd) in enumerate(commands,1):
        log=ROOT/'logs'/f'migration-check-{index}.log';started=time.monotonic()
        with log.open('w',encoding='utf8') as stream:
            process=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        result=dict(command=command,working_directory=str(cwd),exit_code=process.returncode,seconds=time.monotonic()-started,log=str(log))
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        if process.returncode:break
    report=dict(passed=len(results)==len(commands) and all(r['exit_code']==0 for r in results),checkout=str(checkout),
        commit=manifest['commit'],commands=results,scope='Fresh Windows checkout with relocated assets and existing project interpreter; not a Linux installation or GPU test')
    (ROOT/'logs/migration-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
