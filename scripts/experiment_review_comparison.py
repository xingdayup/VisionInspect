"""Compare the baseline and agent_review workflows on the frozen test split.

The experiment changes nothing: it uses the registered checkpoint, profile,
threshold and review band as they are, then measures what the reviewer actually
does on test images —

  * how many images land inside the frozen review band,
  * whether the review changes the candidate state,
  * whether that change agrees with the label,
  * how many model calls and seconds it costs.

Ground-truth labels are read only to score the outcome; they never influence a
decision, and no threshold is recomputed here.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visioninspect.common import (
    configure_environment,
    load_config,
    project_path,
    read_json,
    write_json,
)
from visioninspect.data import load_manifest
from visioninspect.review_agent import ANNOTATED_VIEW_ENV
from visioninspect.service import TaskStore, registry
from visioninspect.vllm_client import EVIDENCE_SIDE_ENV, evidence_side

TERMINAL = {"completed", "failed"}
STATE_TO_LABEL = {"normal_candidate": 0, "anomaly_candidate": 1}


def wait_for_task(store, task, timeout_seconds=900):
    """Block until a task reaches a terminal state, polling the store."""
    deadline = time.monotonic() + timeout_seconds
    while task["status"] not in TERMINAL:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"task {task['id']} did not finish in {timeout_seconds}s"
            )
        time.sleep(0.2)
        task = store.get(task["id"])
    return task


def submit(store, path, category, model, mode):
    return wait_for_task(
        store, store.create(path.read_bytes(), path.name, category, model, mode)
    )


def detection_of(task):
    report = task.get("report") or {}
    return (report.get("outputs") or {}).get("detection") or {}


def result_state(task):
    report = task.get("report") or {}
    return (report.get("result") or {}).get("state") or task.get("state")


def review_of(task):
    return (task.get("report") or {}).get("agent_review") or {}


def score_row(record, baseline, agent, band, threshold):
    """Build one comparable row; agent is None when the review never fired."""
    detection = detection_of(baseline)
    score = detection.get("score")
    row = dict(
        path=record["path"],
        label=record["label"],
        score=score,
        threshold=threshold,
        in_review_band=bool(score is not None and band[0] <= score <= band[1]),
        baseline_state=result_state(baseline),
        baseline_seconds=baseline.get("seconds"),
        agent_state=None,
        agent_seconds=None,
        model_calls=0,
        review_status=None,
        review_reasons=None,
        gates=[],
    )
    if agent is not None:
        review = review_of(agent)
        row.update(
            agent_state=result_state(agent),
            agent_seconds=agent.get("seconds"),
            model_calls=review.get("calls", 0),
            review_status=review.get("status"),
            review_reasons=review.get("reasons"),
            gates=list(review.get("gates") or []),
        )
    return row


def by_label(reviewed):
    """Split review outcomes by ground truth to expose false alarms and misses."""
    table = {}
    for row in reviewed:
        key = "defect" if row["label"] else "good"
        bucket = table.setdefault(key, dict(reviewed=0, normal=0, anomaly=0, manual=0))
        bucket["reviewed"] += 1
        state = row["agent_state"]
        if state == "normal_candidate":
            bucket["normal"] += 1
        elif state == "anomaly_candidate":
            bucket["anomaly"] += 1
        else:
            bucket["manual"] += 1
    if "good" in table:
        table["good"]["false_alarms"] = table["good"]["anomaly"]
    if "defect" in table:
        table["defect"]["missed"] = table["defect"]["normal"]
    return table or None


def summarize(rows, band):
    """Aggregate states, agreement and review effect without hiding misses."""

    def counts(key):
        seen = {}
        for row in rows:
            value = row.get(key)
            if value:
                seen[value] = seen.get(value, 0) + 1
        return seen

    def correctness(key):
        """Count decided rows vs the label, and rows left to a human."""
        decided = correct = manual = 0
        for row in rows:
            expected = STATE_TO_LABEL.get(row.get(key) or "")
            if expected is None:
                manual += 1
                continue
            decided += 1
            correct += expected == row["label"]
        return dict(decided=decided, correct=correct, manual_review=manual)

    reviewed = [row for row in rows if row.get("agent_state")]
    changed = [row for row in reviewed if row["agent_state"] != row["baseline_state"]]
    calls = [row["model_calls"] for row in reviewed]
    seconds = [row["agent_seconds"] for row in reviewed if row.get("agent_seconds")]

    def reviewed_correctness():
        """Score only the images the reviewer actually touched.

        Counting untouched rows as manual_review would overstate the human cost
        and hide whether the review itself was right or wrong.
        """
        decided = correct = 0
        for row in reviewed:
            expected = STATE_TO_LABEL.get(row["agent_state"])
            if expected is None:
                continue
            decided += 1
            correct += expected == row["label"]
        return dict(
            decided=decided, correct=correct, manual_review=len(reviewed) - decided
        )

    return dict(
        test_images=len(rows),
        review_band=list(band),
        in_band=sum(1 for row in rows if row["in_review_band"]),
        baseline_states=counts("baseline_state"),
        baseline_vs_label=correctness("baseline_state"),
        reviewed=len(reviewed),
        review_changed_state=len(changed),
        review_changed_to_manual=sum(
            1 for row in changed if row["agent_state"] == "manual_review"
        ),
        review_vs_label=reviewed_correctness() if reviewed else None,
        model_calls_total=sum(calls),
        model_calls_max=max(calls) if calls else 0,
        review_seconds_mean=(sum(seconds) / len(seconds)) if seconds else None,
        rejection_gates=sorted(
            {code for row in reviewed for code in (row.get("gates") or [])}
        ),
        review_by_label=by_label(reviewed),
        changed_paths=[row["path"] for row in changed],
        label_note="Labels only score the outcome; nothing here tunes a threshold or the band",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model", default="patchcore")
    parser.add_argument("--limit", type=int, default=0, help="First N test images only")
    parser.add_argument("--skip-review", action="store_true", help="Baseline only")
    parser.add_argument(
        "--force-review",
        type=int,
        default=0,
        help="Also review N images outside the band (half hardest normals, half hardest defects)",
    )
    parser.add_argument(
        "--evidence-side",
        type=int,
        default=0,
        help=(
            "Send-side cap for evidence images (default: production 512). "
            "Only the cap changes; regions keep their native crop pixels."
        ),
    )
    parser.add_argument(
        "--annotated-view",
        action="store_true",
        help=(
            "Draw the candidate boxes and the anomaly map onto the two full views. "
            "Evidence ids and the local-binding contract are unchanged."
        ),
    )
    args = parser.parse_args()

    if args.evidence_side:
        # Experiment-only: production never sets this. Recorded below so the
        # run is self-describing and two caps can be compared afterwards.
        os.environ[EVIDENCE_SIDE_ENV] = str(args.evidence_side)
    cap = evidence_side()
    if args.annotated_view:
        os.environ[ANNOTATED_VIEW_ENV] = "1"

    configure_environment()
    registration = registry()[args.category][args.model]
    config = load_config(project_path(registration["config"]))
    profile = read_json(project_path(registration["profile"]))
    band = (profile["review_lower"], profile["review_upper"])
    threshold = profile["image_threshold"]

    _, records = load_manifest(config, ("test",))
    if args.limit:
        records = records[: args.limit]
    root = project_path(config["dataset"]["root"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = (
        project_path(config["outputs"]) / f"{stamp}-review-comparison-{uuid4().hex[:6]}"
    )
    output.mkdir(parents=True)
    write_json(
        output / "config.json",
        dict(
            category=args.category,
            model=args.model,
            checkpoint_sha256=profile["checkpoint_sha256"],
            profile_digest=profile["digest"],
            threshold=threshold,
            review_band=list(band),
            test_images=len(records),
            limit=args.limit or None,
            evidence_side=cap,
            evidence_side_default=not bool(args.evidence_side),
            annotated_view=bool(args.annotated_view),
        ),
    )

    store = TaskStore(output / "service")
    rows = []
    try:
        print(f"RUN_DIRECTORY={output}", flush=True)
        print(f"EVIDENCE_SIDE={cap}", flush=True)
        print(f"ANNOTATED_VIEW={bool(args.annotated_view)}", flush=True)
        print(f"baseline pass over {len(records)} test images", flush=True)
        baselines = {}
        for index, record in enumerate(records, 1):
            baseline = submit(
                store, root / record["path"], args.category, args.model, "baseline"
            )
            baselines[record["path"]] = baseline
            print(
                f"baseline {index}/{len(records)} {record['path']} "
                f"score={detection_of(baseline).get('score')} state={result_state(baseline)}",
                flush=True,
            )

        candidates = [
            record
            for record in records
            if band[0]
            <= (detection_of(baselines[record["path"]]).get("score") or -1)
            <= band[1]
        ]
        print(f"in review band: {len(candidates)}/{len(records)}", flush=True)

        if args.force_review:
            # Judgeability is best measured on the hardest cases: the normal parts
            # that scored highest and the defective ones that scored lowest. This
            # changes no threshold; it only decides which images get reviewed.
            scored = [
                (record, detection_of(baselines[record["path"]]).get("score") or 0.0)
                for record in records
                if record["path"] not in {c["path"] for c in candidates}
            ]
            good = sorted(
                (item for item in scored if item[0]["label"] == 0),
                key=lambda item: item[1],
                reverse=True,
            )
            bad = sorted(
                (item for item in scored if item[0]["label"] != 0),
                key=lambda item: item[1],
            )
            half = max(1, args.force_review // 2)
            for record, _ in good[:half] + bad[:half]:
                candidates.append(record)
            print(
                f"forced review on {len(good[:half])} hard normals and "
                f"{len(bad[:half])} hard defects (total {len(candidates)})",
                flush=True,
            )

        agents = {}
        if candidates and not args.skip_review:
            for index, record in enumerate(candidates, 1):
                agent = submit(
                    store,
                    root / record["path"],
                    args.category,
                    args.model,
                    "agent_review",
                )
                agents[record["path"]] = agent
                print(
                    f"review {index}/{len(candidates)} {record['path']} "
                    f"state={result_state(agent)} calls={review_of(agent).get('calls')}",
                    flush=True,
                )

        for record in records:
            rows.append(
                score_row(
                    record,
                    baselines[record["path"]],
                    agents.get(record["path"]),
                    band,
                    threshold,
                )
            )
    finally:
        store.close()

    summary = summarize(rows, band)
    write_json(output / "comparison.json", dict(summary=summary, rows=rows))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "path",
                "label",
                "score",
                "in_review_band",
                "baseline_state",
                "agent_state",
                "model_calls",
                "baseline_seconds",
                "agent_seconds",
                "gates",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
