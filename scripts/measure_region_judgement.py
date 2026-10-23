"""Measure how the local VLM judges single regions of known-good images.

Two numbers matter for judgeability:

  A. false-positive rate: given a normal part, how often is one region view called
     anomalous when shown alone, with no prior conversation;
  B. anchoring: the same view prefixed by an earlier "anomaly" claim from the same
     model. A large flip rate means the loop's accumulated observations steer the
     verdict, which is a loop problem rather than a model capability limit.

Labels only select known-good images; nothing is tuned and no threshold changes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image

from visioninspect.common import (
    configure_environment,
    load_config,
    offline,
    project_path,
    write_json,
)
from visioninspect.data import load_manifest
from visioninspect.service import registry
from visioninspect.tools import ToolKit
from visioninspect.vllm_client import VLLMClient
from visioninspect.vllm_runtime import gpu_config

MAX_REGIONS_PER_IMAGE = 2


def matched_crop(toolkit, reference_path, region, destination):
    """Crop the reference at the region grid coordinates, like the review loop."""
    with Image.open(toolkit.output / reference_path) as image:
        box = [
            round(value * image.size[index % 2] / 256)
            for index, value in enumerate(region["box_grid"])
        ]
        image.convert("RGB").crop(box).save(destination)
    return destination


def context_for(detection, region, images, prior, region_total):
    return dict(
        category="bottle",
        detection=detection,
        view=["region", region["id"], 1],
        available_evidence=[i["id"] for i in images],
        regions=[
            dict(
                id=region["id"],
                box_grid=region["box_grid"],
                maximum_score=region["maximum_score"],
            )
        ],
        reference_ids=[1],
        reviewed_regions={},
        remaining_regions=[region["id"]],
        coverage=f"0/{region_total}",
        previous_observations=prior,
        calls_remaining_after_this=3,
        abstention_nudge=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--regions", type=int, default=MAX_REGIONS_PER_IMAGE)
    args = parser.parse_args()

    configure_environment()
    registration = registry()[args.category]["patchcore"]
    config = load_config(project_path(registration["config"]))
    policy = yaml.safe_load((ROOT / "configs/agent.yaml").read_text("utf8"))
    _, records = load_manifest(config, ("test",))
    good = [r for r in records if r["label"] == 0][: args.images]
    root = project_path(config["dataset"]["root"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = project_path(config["outputs"]) / f"{stamp}-region-judgement"
    output.mkdir(parents=True)
    client = VLLMClient(gpu_config())
    rows = []

    with offline(allow_loopback_ports=(8001,)):
        for index, record in enumerate(good, 1):
            directory = output / f"image_{index:02d}"
            toolkit = ToolKit(registration, policy, directory)
            request = dict(
                image_path=str(root / record["path"]), category=args.category
            )
            detection = toolkit.invoke("detect_anomaly", request)
            found = toolkit.invoke(
                "analyze_suspicious_region",
                dict(**request, anomaly_map_path=detection["anomaly_map_path"]),
            )
            references = toolkit.invoke("retrieve_normal_reference", request)
            reference_paths = references["reference_paths"]
            total_regions = len(found.get("regions") or [])
            for region in (found.get("regions") or [])[: args.regions]:
                region_id = region["id"]
                destination = directory / "evidence" / f"measure_{region_id}.png"
                matched_crop(toolkit, reference_paths[0], region, destination)
                images = [
                    dict(id="original", path=directory / "evidence/original.png"),
                    dict(
                        id=f"region:{region_id}", path=directory / region["crop_path"]
                    ),
                    dict(id=f"matched:{region_id}:1", path=destination),
                ]
                pair = None
                if len(reference_paths) > 1:
                    # Two references: a real difference should disagree with both.
                    second = directory / "evidence" / f"measure_{region_id}_b.png"
                    matched_crop(toolkit, reference_paths[1], region, second)
                    pair = images + [dict(id=f"matched:{region_id}:2", path=second)]
                paired = None
                if pair is not None:
                    two, _ = client.decide(
                        context_for(detection, region, pair, [], total_regions),
                        pair,
                        directory / "vlm" / region_id / "two_references",
                    )
                    paired = [o.finding for o in two.observations]
                fresh, _ = client.decide(
                    context_for(detection, region, images, [], total_regions),
                    images,
                    directory / "vlm" / region_id / "fresh",
                )
                anchored, _ = client.decide(
                    context_for(
                        detection,
                        region,
                        images,
                        [
                            dict(
                                text="该区域与参考存在纹理差异",
                                evidence_ids=[f"region:{region_id}"],
                                region_id=region_id,
                                finding="anomaly",
                            )
                        ],
                        total_regions,
                    ),
                    images,
                    directory / "vlm" / region_id / "anchored",
                )
                rows.append(
                    dict(
                        image=record["path"],
                        region=region_id,
                        fresh=[o.finding for o in fresh.observations],
                        anchored=[o.finding for o in anchored.observations],
                        two_references=paired,
                    )
                )
                print("MEASURED", record["path"], region_id, rows[-1], flush=True)

    total = len(rows)
    fresh_anomaly = sum(1 for r in rows if "anomaly" in r["fresh"])
    anchored_anomaly = sum(1 for r in rows if "anomaly" in r["anchored"])
    flips = sum(
        1 for r in rows if "anomaly" not in r["fresh"] and "anomaly" in r["anchored"]
    )
    summary = dict(
        regions=total,
        fresh_anomaly=fresh_anomaly,
        fresh_false_positive_rate=round(fresh_anomaly / total, 4) if total else None,
        anchored_anomaly=anchored_anomaly,
        anchoring_flips=flips,
        two_reference_anomaly=sum(
            1
            for r in rows
            if r.get("two_references") and "anomaly" in r["two_references"]
        ),
        two_reference_regions=sum(1 for r in rows if r.get("two_references")),
        anchoring_flip_rate=round(flips / total, 4) if total else None,
        note="Known-good images only; labels select images, never tune anything",
    )
    write_json(output / "measurement.json", dict(summary=summary, rows=rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
