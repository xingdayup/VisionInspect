"""CPU baseline CLI. Asset download is a separate, explicitly invoked script."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from .common import ROOT, configure_environment, identity, load_config, offline, project_path, read_json, sha256, write_json


def source_run(args, config):
    if not args.run:
        raise ValueError('--run must identify the completed formal fit directory')
    run = Path(args.run).resolve()
    checkpoint = run/'model.ckpt'
    summary = read_json(run/'fit.json')
    if summary['config_digest'] != identity(config) or summary['purpose'] != 'formal':
        raise ValueError('Source fit config mismatch or smoke checkpoint used as formal model')
    if sha256(checkpoint) != summary['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since fit')
    return run, summary


def predictions(detector, root, records, output, warmup_record=None, warmup=0):
    import numpy as np
    from PIL import Image
    from .artifacts import save_evidence
    if warmup_record is not None:
        with Image.open(root/warmup_record['path']) as image:
            array = np.asarray(image.convert('RGB'))
        for _ in range(warmup):
            detector.predict(array)
    results, maps, latencies = [], [], []
    for index, record in enumerate(records):
        with Image.open(root/record['path']) as image:
            array = np.asarray(image.convert('RGB'))
        score, anomaly_map, elapsed = detector.predict(array)
        key = record['path'].replace('/', '_').removesuffix('.png')
        evidence = save_evidence(output/'evidence', key, array, anomaly_map)
        results.append(dict(**record, score=score, latency_seconds=elapsed, evidence=evidence))
        maps.append(anomaly_map)
        latencies.append(elapsed)
        print(f'predicted {index+1}/{len(records)} {record["path"]}', flush=True)
    write_json(output/'predictions.json', results)
    import csv
    with (output/'scores.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path','split','label','score','latency_seconds'])
        writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in results)
    write_json(output/'latency.json', dict(warmup=warmup, warmup_split='validation' if warmup else None,
               timed='preprocess + model, excludes file read/write', count=len(latencies),
               p50_seconds=float(np.percentile(latencies,50)), p95_seconds=float(np.percentile(latencies,95))))
    return results, maps


def worker(args, config, output):
    from .data import check_data, load_manifest
    from .model import Detector, setup
    setup(config)
    started = time.perf_counter()
    if args.command == 'check-data':
        manifest = check_data(config)
        counts = {s:sum(r['split'] == s for r in manifest['records']) for s in ['train','validation','test']}
        write_json(output/'data-check.json', dict(counts=counts, manifest_digest=manifest['digest']))
        print(json.dumps(counts), flush=True)
        return
    if args.command in {'smoke','fit'}:
        manifest, train = load_manifest(config, ('train',))
        smoke = args.command == 'smoke'
        if smoke:
            train = train[:16]
            if len(train) != 16:
                raise ValueError('Smoke needs 16 training images')
        detector = Detector(config, size=128 if smoke else None)
        backbone_hash = detector.load_backbone()
        detector.fit(project_path(config['dataset']['root']), train, lambda message: print(message, flush=True))
        metadata = dict(manifest_digest=manifest['digest'], backbone_sha256=backbone_hash,
                        config_digest=identity(config), purpose='smoke' if smoke else 'formal')
        checkpoint = output/'model.ckpt'
        detector.save(checkpoint, metadata)
        fit_result = dict(**metadata, checkpoint_sha256=sha256(checkpoint), training_images=len(train),
                          memory_bank_shape=list(detector.model.memory_bank.shape), fit_seconds=time.perf_counter()-started)
        write_json(output/'split.json', manifest)
        write_json(output/'fit.json', fit_result)
        if smoke:
            import numpy as np
            from PIL import Image
            _, validation = load_manifest(config, ('validation',))
            validation = validation[:4]
            if len(validation) != 4:
                raise ValueError('Smoke needs 4 validation images')
            results, maps = predictions(detector, project_path(config['dataset']['root']), validation, output)
            reloaded, _ = Detector.load(checkpoint)
            with Image.open(project_path(config['dataset']['root'])/validation[0]['path']) as image:
                score, anomaly_map, _ = reloaded.predict(np.asarray(image.convert('RGB')))
            np.testing.assert_allclose(score, results[0]['score'], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(anomaly_map, maps[0], rtol=1e-5, atol=1e-6)
            write_json(output/'smoke.json', dict(passed=True, train=16, validation=4, image_size=128,
                      checkpoint_roundtrip=True, network_disabled=True, formal_metrics=False))
        return
    run, fit_result = source_run(args, config)
    detector, checkpoint_metadata = Detector.load(run/'model.ckpt')
    if checkpoint_metadata['manifest_digest'] != fit_result['manifest_digest']:
        raise ValueError('Checkpoint manifest identity mismatch')
    if args.command in {'calibrate','evaluate'}:
        manifest, records = load_manifest(config, ('validation',) if args.command == 'calibrate' else ('test',))
        if manifest['digest'] != fit_result['manifest_digest']:
            raise ValueError('Dataset changed since fit')
        calibration_path = run/'calibration.json'
        if args.command == 'calibrate':
            if calibration_path.exists():
                raise FileExistsError('Calibration already frozen for this fit; create a new experiment to change it')
            from .metrics import calibrate
            results, _ = predictions(detector, project_path(config['dataset']['root']), records, output)
            threshold = calibrate(results, config['evaluation']['quantile'])
            calibration = dict(threshold=threshold, quantile=config['evaluation']['quantile'], method='higher',
                               validation_count=len(records), checkpoint_sha256=fit_result['checkpoint_sha256'],
                               manifest_digest=manifest['digest'], config_digest=identity(config), source_run=str(output),
                               seconds=time.perf_counter()-started,
                               limitation='Normal validation only; no guarantee of population false-positive rate')
            write_json(output/'calibration.json', calibration)
            write_json(calibration_path, calibration)
            return
        if (run/'evaluation.json').exists():
            raise FileExistsError('Formal test already evaluated for this experiment; use existing outputs')
        calibration = read_json(calibration_path)
        if calibration['checkpoint_sha256'] != fit_result['checkpoint_sha256'] or calibration['manifest_digest'] != manifest['digest'] or calibration['config_digest'] != identity(config):
            raise ValueError('Calibration identity mismatch')
        calibration_hash = sha256(calibration_path)
        _, validation = load_manifest(config, ('validation',))
        results, maps = predictions(detector, project_path(config['dataset']['root']), records, output,
                                    warmup_record=validation[0], warmup=config['evaluation']['warmup'])
        import numpy as np
        from PIL import Image
        from .metrics import evaluate
        masks = []
        for record in records:
            if record['label']:
                with Image.open(project_path(config['dataset']['root'])/record['mask']) as image:
                    mask = np.asarray(image.resize((detector.size, detector.size), Image.Resampling.NEAREST)) > 0
                if not mask.any():
                    raise ValueError(f'Anomaly mask vanished during resize: {record["mask"]}')
            else:
                mask = np.zeros((detector.size,detector.size),dtype=np.uint8)
            masks.append(mask)
        metrics = evaluate([r['label'] for r in records], [r['score'] for r in results], maps, masks,
                           calibration['threshold'], config['evaluation']['aupro_fpr_limit'])
        metrics.update(threshold=calibration['threshold'], checkpoint_sha256=fit_result['checkpoint_sha256'],
                       calibration_sha256=calibration_hash, manifest_digest=manifest['digest'],
                       test_images=len(records), seconds=time.perf_counter()-started,
                       gpu_memory='not applicable: CPU run', configuration='PatchCore-ResNet18 CPU, not paper default')
        if sha256(calibration_path) != calibration_hash:
            raise ValueError('Calibration mutated during test')
        write_json(output/'metrics.json',metrics)
        write_json(run/'evaluation.json',dict(output=str(output), metrics_sha256=sha256(output/'metrics.json')))
        print(json.dumps(metrics),flush=True)
        return
    if not args.image:
        raise ValueError('predict requires --image')
    calibration = read_json(run/'calibration.json')
    if calibration['checkpoint_sha256'] != fit_result['checkpoint_sha256'] or calibration['config_digest'] != identity(config):
        raise ValueError('Calibration does not belong to this checkpoint/config')
    import numpy as np
    from PIL import Image
    from .artifacts import save_evidence
    image_path = Path(args.image).resolve()
    with Image.open(image_path) as image:
        array = np.asarray(image.convert('RGB'))
    score, anomaly_map, elapsed = detector.predict(array)
    evidence = save_evidence(output/'evidence','input',array,anomaly_map)
    write_json(output/'prediction.json',dict(image=str(image_path), image_sha256=sha256(image_path), score=score,
               threshold=calibration['threshold'], state='anomaly_candidate' if score > calibration['threshold'] else 'normal_candidate',
               latency_seconds=elapsed, checkpoint_sha256=fit_result['checkpoint_sha256'],
               evidence=evidence, network_disabled=True, research_prototype=True))


def main():
    configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check-data','smoke','fit','calibrate','evaluate','predict'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/patchcore_cpu.yaml')
    parser.add_argument('--run', help='Completed formal fit directory')
    parser.add_argument('--image', help='Single image for predict')
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker_output:
        try:
            with offline():
                worker(args, config, args.worker_output)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    output = project_path(config['outputs'])/f'{stamp}-{args.command}-{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable,'-B','-u','-m','visioninspect.baseline',args.command,'--config',str(args.config.resolve()),
               '--worker-output',str(output)]
    if args.run:
        command += ['--run', str(Path(args.run).resolve())]
    if args.image:
        command += ['--image',str(Path(args.image).resolve())]
    write_json(output/'config.json',config)
    write_json(output/'command.json',dict(argv=command, cwd=str(ROOT), purpose='actual execution'))
    write_json(output/'environment.json',dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
               os=platform.platform(), packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    print(f'RUN_DIRECTORY={output}',flush=True)
    import psutil
    from .resources import tree_memory,kill_tree
    seconds_limit=config['runtime'].get(f'{args.command}_timeout_seconds',600)
    started=time.perf_counter()
    peak=0
    process_peaks={}
    reason=None
    with (output/'execution.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        monitored=psutil.Process(process.pid)
        while process.poll() is None:
            peak=max(peak,tree_memory(monitored,process_peaks))
            if peak > config['runtime']['memory_limit_gib']*1024**3:
                reason='memory_limit_exceeded'
            elif time.perf_counter()-started > seconds_limit:
                reason='timeout'
            if reason:
                kill_tree(monitored)
                process.wait()
                break
            time.sleep(0.25)
    status=dict(command=args.command,exit_code=process.returncode,seconds=time.perf_counter()-started,
                peak_process_tree_bytes=peak,process_peak_bytes=process_peaks,
                memory_method='sum of OS peak working set for observed process tree; conservative upper bound',
                stop_reason=reason,gpu_memory='not applicable: CPU run')
    write_json(output/'status.json',status)
    print(json.dumps(status),flush=True)
    if process.returncode:
        print((output/'execution.log').read_text(encoding='utf-8')[-8000:],file=sys.stderr)
    return 0 if process.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
