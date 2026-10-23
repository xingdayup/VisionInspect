#!/usr/bin/env bash
# Run the review-resolution experiment: three arms, sequential, self-describing.
#
# Why sequential: an earlier attempt ran two reviews at once, both arms fought
# over the same GPU and port 8001, and the measurement died halfway. Each arm
# here waits for the previous one to finish writing its run directory.
#
# Arms (only one variable changes per step):
#   base    512 px cap, plain views      -> must reproduce the recorded 3/4 result
#   marked  512 px cap, annotated views  -> does pointing at the boxes help?
#   wide    1024 px cap, annotated views -> is resolution also a bottleneck?
#
# Usage:
#   bash scripts/run_review_resolution_experiment.sh
#   FORCE=24 bash scripts/run_review_resolution_experiment.sh   # 24 images per arm
#   ARMS="base marked" bash scripts/run_review_resolution_experiment.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

if [[ -f scripts/env_autodl.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env_autodl.sh
fi
: "${VISIONINSPECT_GPU_ROOT:?set VISIONINSPECT_GPU_ROOT or source scripts/env_autodl.sh}"
PYTHON="${PYTHON:-python3}"
FORCE="${FORCE:-0}"
ARMS="${ARMS:-base marked wide}"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="logs/review-resolution-$STAMP"
mkdir -p "$REPORT"

note() { printf '\n=== %s\n' "$*"; }

note "preflight"
"$PYTHON" - <<'PY'
import sys, torch
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA here: PatchCore refuses to run, run this on the GPU host")
PY
curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 \
  && echo "vLLM endpoint already up" \
  || echo "vLLM endpoint down: the runtime starts it on demand, first call is slow"

run_arm() {
  local label="$1" side="$2" extra="$3" log
  log="$REPORT/$label.log"
  note "arm $label (evidence side $side) ${extra:-(plain views)}"
  # shellcheck disable=SC2086
  "$PYTHON" -B scripts/experiment_review_comparison.py \
    --evidence-side "$side" ${extra} --force-review "$FORCE" 2>&1 | tee "$log"
  grep -o 'RUN_DIRECTORY=.*' "$log" | tail -1 | cut -d= -f2 > "$REPORT/$label.dir"
}

for arm in $ARMS; do
  case "$arm" in
    base)   run_arm base 512 "" ;;
    marked) run_arm marked 512 "--annotated-view" ;;
    wide)   run_arm wide 1024 "--annotated-view" ;;
    *)      echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
done

note "comparison"
"$PYTHON" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path

report = Path(sys.argv[1])
rows = []
for label in ("base", "marked", "wide"):
    path = report / f"{label}.dir"
    if not path.is_file():
        continue
    run = Path(path.read_text().strip())
    summary = json.loads((run / "comparison.json").read_text("utf-8"))["summary"]
    config = json.loads((run / "config.json").read_text("utf-8"))
    label_correct = summary.get("review_vs_label") or {}
    by_label = summary.get("review_by_label") or {}
    rows.append(
        dict(
            arm=label,
            side=config.get("evidence_side"),
            annotated=config.get("annotated_view"),
            reviewed=summary.get("reviewed"),
            decided=label_correct.get("decided"),
            correct=label_correct.get("correct"),
            manual=label_correct.get("manual_review"),
            false_alarms=(by_label.get("good") or {}).get("false_alarms"),
            missed=(by_label.get("defect") or {}).get("missed"),
            calls=summary.get("model_calls_total"),
            max_calls=summary.get("model_calls_max"),
            seconds=round(summary.get("review_seconds_mean") or 0, 1),
            gates=",".join(summary.get("rejection_gates") or []),
        )
    )

if not rows:
    raise SystemExit("no completed arm found")

columns = [
    ("arm", 7), ("side", 5), ("annotated", 9), ("reviewed", 8), ("decided", 7),
    ("correct", 7), ("manual", 6), ("false_alarms", 12), ("missed", 6),
    ("calls", 5), ("max_calls", 9), ("seconds", 7), ("gates", 40),
]
head = "".join(name.ljust(width) for name, width in columns)
print(head)
print("-" * len(head))
for row in rows:
    print("".join(str(row[name]).ljust(width) for name, width in columns))

print("\nread it in this order:")
print("1. base must show decided>0 and no missed defects, otherwise the host differs from the recorded run")
print("2. marked better than base on correct/manual with false_alarms unchanged -> pointing helps")
print("3. wide better than marked -> resolution is also a bottleneck")
print("4. any arm with false_alarms>0 or missed>0 is not adoptable")
print(f"\nraw runs and logs: {report}")
PY

note "done: $REPORT"
