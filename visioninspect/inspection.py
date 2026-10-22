"""CLI for the same supervised workflow used by the workstation."""
import argparse,time
from pathlib import Path
from .common import configure_environment,ROOT
from .service import TaskStore,registry

def main():
    parser=argparse.ArgumentParser(description='GPU 视觉质检工作台命令行入口')
    parser.add_argument('--image',type=Path,required=True);parser.add_argument('--category',default='bottle')
    parser.add_argument('--model',default='patchcore');parser.add_argument('--mode',choices=['baseline','agent_review'],default='agent_review')
    args=parser.parse_args()
    if args.category not in registry() or args.model not in registry()[args.category]:parser.error('未注册的类别或模型')
    if not args.image.is_file():parser.error('输入图像不存在')
    if args.image.stat().st_size>10*1024**2:parser.error('图像超过 10 MiB')
    configure_environment();store=TaskStore(ROOT/'outputs/cli')
    try:
        task=store.create(args.image.read_bytes(),args.image.name,args.category,args.model,args.mode)
        print('TASK',task['id'],flush=True)
        while task['status'] in {'queued','running'}:
            time.sleep(.2);task=store.get(task['id'])
        print('STATUS',task['status'],task['state'],task.get('error') or '',flush=True)
        print('REPORT',store.directory(task['id'])/'report.json',flush=True)
        return 0 if task['status']=='completed' else 1
    finally:store.close()

if __name__=='__main__':raise SystemExit(main())
