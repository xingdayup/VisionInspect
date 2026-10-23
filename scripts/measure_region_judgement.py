"""Measure how the local VLM judges single regions of known-good images.

Two numbers matter for judgeability:

  A. false-positive rate: given a normal part, how often does one region view get
     called anomalous when shown alone, with no prior conversation;
  B. anchoring: the same region view, prefixed by an earlier "anomaly" claim from
     the same model. A large flip rate means the review loop's accumulated
     observations are steering the verdict, which is a loop problem, not a model
     capability limit.

No thresholds are changed and no labels are used for tuning; labels only select
known-good images. This is a measurement, not an acceptance test.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from visioninspect.agent import InspectionAgent
from visioninspect.common import (
    configure_environment,
    load_config,
    offline,
    project_path,
    read_json,
    write_json,
)
from visioninspect.data import load_manifest
from visioninspect.service import registry
from visioninspect.tools import ToolKit
from visioninspect.vllm_client import VLLMClient
from visioninspect.vllm_runtime import gpu_config

MAX_REGIONS_PER_IMAGE = 2


def region_images(agent, region_id, reference_id):
    """Mirror ReviewAgent.images() for one region view."""
    root = agent.tools.output
    region = agent.regions[region_id]
    reference = root / agent.references[reference_id - 1]
    destination = root / "evidence" / f"measure_{region_id}_ref_{reference_id}.png"
    with Image.open(reference) as image:
        box = [
            round(value * image.size[index % 2] / 256)
            for index, value in enumerate(region["box_grid"])
        ]
        image.convert("RGB").crop(box).save(destination)
    return [
        dict(id="original", path=root / "evidence/original.png"),
        dict(id=f"region:{region_id}", path=root / region["crop_path"]),
        dict(id=f"matched:{region_id}:{reference_id}", path=destination),
    ]


def context_for(agent, region_id, images, prior):
    return dict(
        category=agent.category,
        detection=agent.report["outputs"]["detection"],
        view=["region", region_id, 1],
        available_evidence=[i["id"] for i in images],
        regions=[
            dict(
                id=r["id"],
                box_grid=r["box_grid"],
                maximum_score=r["maximum_score"],
            )
            for r in agent.regions.values()
            if r["id"] == region_id
        ],
        reference_ids=[1],
        reviewed_regions={},
        remaining_regions=[region_id],
        coverage=f"0/{len(agent.regions)}",
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
    policy = read_json(ROOT / "configs/agent.yaml")
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
            agent = InspectionAgent(ToolKit(registration, policy, directory), policy)
            agent.run(root / record["path"], args.category, "baseline")
            regions = list(agent.regions)[: args.regions]
            for region_id in regions:
                images = region_images(agent, region_id, 1)
                view_dir = directory / f"vlm_{region_id}"
                fresh, _ = client.decide(
                    context_for(agent, region_id, images, []),
                    images,
                    view_dir / "fresh",
                )
                anchored, _ = client.decide(
                    context_for(
                        agent,
                        region_id,
                        images,
                        [
                            dict(
                                text="该区域与参考存在纹理差异",
                                evidence_ids=[f"region:{region_id}"],
                                region_id=region_id,
                                finding="anomaly",
                            )
                        ],
                    ),
                    images,
                    view_dir / "anchored",
                )
                rows.append(
                    dict(
                        image=record["path"],
                        region=region_id,
                        fresh=[o.finding for o in fresh.observations],
                        fresh_verdict=fresh.verdict,
                        anchored=[o.finding for o in anchored.observations],
                        anchored_verdict=anchored.verdict,
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
        fresh_false_positive_rate=fresh_anomaly / total if total else None,
        anchored_anomaly=anchored_anomaly,
        anchoring_flips=flips,
        anchoring_flip_rate=flips / total if total else None,
        note="Known-good images only; labels select images, never tune anything",
    )
    write_json(output / "measurement.json", dict(summary=summary, rows=rows))
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
