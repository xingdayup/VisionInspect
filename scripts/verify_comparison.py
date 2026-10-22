"""Post-evaluation parity check against the frozen visual baseline, without fitting anything."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import read_json,write_json

directory=Path(sys.argv[1]);directory=directory if directory.is_absolute() else ROOT/directory
frozen=read_json(directory/'frozen.json')
baseline=ROOT/frozen.get('visual_metrics_source','outputs/20261021-232247-evaluate-5225f8/metrics.json')
baseline=baseline.parent
original={row['path']:row for row in read_json(baseline/'predictions.json')}
rows=read_json(directory/'predictions.json');maximum_score_difference=0.;maximum_map_difference=0.;checked=0;skipped=[]
for index,row in enumerate(rows):
    report=read_json(directory/f'{index:03d}'/'report.json');detection=report['outputs'].get('detection')
    if not detection:skipped.append(row['path']);continue
    expected=original[row['path']];difference=abs(detection['score']-expected['score'])
    assert np.isclose(detection['score'],expected['score'],rtol=1e-5,atol=1e-6),row['path']
    actual_map=np.load(directory/f'{index:03d}'/detection['anomaly_map_path'],allow_pickle=False)
    expected_map=np.load(baseline/'evidence'/expected['evidence']['raw_map'],allow_pickle=False)
    assert np.isfinite(actual_map).all() and actual_map.shape==(256,256)
    assert np.allclose(actual_map,expected_map,rtol=1e-5,atol=1e-6),row['path']
    maximum_score_difference=max(maximum_score_difference,difference)
    maximum_map_difference=max(maximum_map_difference,float(np.max(np.abs(actual_map-expected_map))))
    checked+=1
result=dict(total=len(rows),checked=checked,quality_rejected_or_missing_detection=skipped,passed=True,
            rtol=1e-5,atol=1e-6,maximum_score_absolute_difference=maximum_score_difference,maximum_map_absolute_difference=maximum_map_difference)
write_json(directory/'score-parity.json',result);print(result)
