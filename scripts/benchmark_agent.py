"""Frozen A/D or C evaluation. Manual review is abstention, never counted as a correct prediction."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse,time,sys,shutil,importlib.metadata,platform
from uuid import uuid4
import numpy as np,yaml,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import configure_environment,load_config,offline,project_path,read_json,write_json,sha256,identity
from visioninspect.model import Detector,setup
from visioninspect.data import load_manifest
from visioninspect.agent import InspectionAgent
from visioninspect.tools import ToolKit

def run(mode,enable_vlm,category='bottle'):
    policy=yaml.safe_load((ROOT/'configs/agent.yaml').read_text(encoding='utf8'));policy['vlm']['enabled']=enable_vlm
    registration=yaml.safe_load((ROOT/'configs/models.yaml').read_text(encoding='utf8'))['models'][category]['patchcore']
    config=load_config(project_path(registration['config']));setup(config)
    profile=read_json(project_path(registration['profile']))
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    directory=ROOT/'outputs'/f'{stamp}-{category}-agent-{mode}-{uuid4().hex[:6]}';directory.mkdir()
    fit=project_path(registration['fit_run'])
    evaluation=read_json(fit/'evaluation.json')
    visual=project_path(evaluation['output'])
    if sha256(visual/'metrics.json')!=evaluation['metrics_sha256']:raise ValueError('Baseline metrics changed')
    if sha256(fit/'model.ckpt')!=profile['checkpoint_sha256']:raise ValueError('Checkpoint/profile mismatch')
    frozen=dict(category=category,mode=mode,policy=policy,policy_digest=identity(policy),profile_digest=profile['digest'],
                checkpoint_sha256=profile['checkpoint_sha256'],manifest_digest=profile['manifest_digest'],
                visual_metrics_source=(visual/'metrics.json').relative_to(ROOT).as_posix(),
                code={name:sha256(ROOT/'visioninspect'/name) for name in ('agent.py','tools.py','vlm.py')},
                note='Frozen before opening evaluation test images; no thresholds or prompts fitted on test labels')
    write_json(directory/'frozen.json',frozen)
    write_json(directory/'config.json',config)
    write_json(directory/'environment.json',dict(python=sys.version,os=platform.platform(),
               packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    write_json(directory/'command.json',dict(argv=[sys.executable,*sys.argv],cwd=str(ROOT)))
    sources=directory/'source/visioninspect';sources.mkdir(parents=True)
    for name,expected in frozen['code'].items():
        shutil.copy2(ROOT/'visioninspect'/name,sources/name)
        if sha256(sources/name)!=expected:raise ValueError('Experiment source changed while freezing')
    detector,_=Detector.load(project_path(registration['fit_run'])/'model.ckpt')
    vlm=None
    if enable_vlm:
        from visioninspect.vlm import LocalVLM
        vlm=LocalVLM(project_path(policy['vlm']['directory']),policy['vlm']['max_new_tokens']);vlm.load()
    manifest,records=load_manifest(config,('test',));root=(ROOT/config['dataset']['root']);rows=[];start=time.perf_counter()
    for i,record in enumerate(records):
        output=directory/f'{i:03d}';tools=ToolKit(registration,policy,output,vlm);tools._detector=detector
        report=InspectionAgent(tools,policy).run(str(root/record['path']),category,mode)
        write_json(output/'report.json',report)
        rows.append(dict(path=record['path'],label=record['label'],state=report['result']['state'],
                         seconds=report['seconds'],tool_calls=report['tool_calls'],vlm_calls=report['vlm_calls'],
                         completed=not report['errors'],attempts=len(report['trace']),
                         successful=sum(item['status']=='completed' for item in report['trace']),
                         invalid=sum(item.get('error',{}).get('code')=='schema_error' for item in report['trace'])))
        print(f"{i+1}/{len(records)} {record['path']} {rows[-1]['state']} {rows[-1]['seconds']:.3f}s",flush=True)
        if psutil.Process().memory_info().rss>12*1024**3:raise MemoryError('12 GiB budget exceeded')
        if time.perf_counter()-start>15*60:raise TimeoutError('15 minute comparison budget exceeded')
    states=Counter(row['state'] for row in rows);accepted=[row for row in rows if row['state']!='manual_review']
    attempted=sum(row['attempts'] for row in rows)
    result=dict(group=('C: fixed VLM' if enable_vlm else 'fixed tools only (not C)') if mode=='fixed' else 'D: dynamic tools',count=len(rows),
                states=dict(states),coverage=len(accepted)/len(rows),manual_review=len(rows)-len(accepted),
                accepted_false_positive=sum(row['state']=='anomaly_candidate' and row['label']==0 for row in accepted),
                accepted_false_negative=sum(row['state']=='normal_candidate' and row['label']==1 for row in accepted),
                manual_review_normal=sum(row['state']=='manual_review' and row['label']==0 for row in rows),
                manual_review_anomaly=sum(row['state']=='manual_review' and row['label']==1 for row in rows),
                tool_success_rate=sum(row['successful'] for row in rows)/attempted,
                invalid_call_rate=sum(row['invalid'] for row in rows)/attempted,
                task_completion_rate=sum(row['completed'] for row in rows)/len(rows),
                mean_tool_calls=float(np.mean([row['tool_calls'] for row in rows])),
                vlm_calls=sum(row['vlm_calls'] for row in rows),monetary_cost=0,
                warm_end_to_end_p50_seconds=float(np.quantile([row['seconds'] for row in rows],.5)),
                warm_end_to_end_p95_seconds=float(np.quantile([row['seconds'] for row in rows],.95)),
                comparison_seconds=time.perf_counter()-start,peak_process_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
                interpretation='Manual review is abstention. Accepted FP/FN excludes abstentions and cannot imply overall accuracy improvement.',
                category=category,visual_metrics_source=frozen['visual_metrics_source'],
                visual_scores_unchanged_by_agent=True,policy_digest=identity(policy),profile_digest=profile['digest'])
    write_json(directory/'predictions.json',rows);write_json(directory/'metrics.json',result)
    print(result,flush=True);print('OUTPUT',directory,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['dynamic','fixed'],default='dynamic');parser.add_argument('--vlm',action='store_true')
    parser.add_argument('--category',default='bottle',choices=['bottle','metal_nut','capsule','pill']);args=parser.parse_args()
    configure_environment()
    with offline():run(args.mode,args.vlm,args.category)
