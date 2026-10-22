"""Explicit synthetic protocol fixtures; these tests measure no model accuracy."""

import pytest
import yaml
from PIL import Image
from visioninspect.common import ROOT, read_json
from visioninspect.review_agent import ReviewAgent
from visioninspect.review_contracts import ReviewDecision
from visioninspect.schemas import FinalizeInput
from visioninspect.tools import ToolKit, ToolFailure


class FixtureTools:
    threshold = 10

    def __init__(self, root, score=10, regions=1):
        self.output = root
        self.score = score
        self.count = regions
        self.policy = yaml.safe_load((ROOT / "configs/agent.yaml").read_text("utf8"))
        self.profile = dict(
            review_lower=9,
            review_upper=11,
            digest="unit-test",
            reference_similarity_floor=0.8,
        )
        (root / "evidence").mkdir()
        for name in ["original", "reference_1", "reference_2"] + [
            f"region_{i}" for i in range(1, regions + 1)
        ]:
            Image.new("RGB", (256, 256), (80, 90, 100)).save(
                root / "evidence" / f"{name}.png"
            )

    def invoke(self, name, args):
        if name == "inspect_image_quality":
            return dict(valid=True)
        if name == "detect_anomaly":
            return dict(
                score=self.score,
                threshold=10,
                evidence={"original": "evidence/original.png"},
                anomaly_map_path="map.npy",
            )
        if name == "analyze_suspicious_region":
            regions = [
                dict(
                    id=f"R{i}",
                    box_grid=[16, 24, 80, 96],
                    maximum_score=12,
                    crop_path=f"evidence/region_{i}.png",
                )
                for i in range(1, self.count + 1)
            ]
            return dict(regions=regions, crop_paths=[r["crop_path"] for r in regions])
        if name == "retrieve_normal_reference":
            return dict(
                reference_paths=[
                    "evidence/reference_1.png",
                    "evidence/reference_2.png",
                ],
                similarities=[0.99, 0.98],
                similarity_floor=0.8,
            )
        if name == "finalize_inspection":
            return ToolKit.finalize_inspection(self, FinalizeInput.model_validate(args))
        raise AssertionError(name)


def decision(
    action="finish",
    verdict="normal",
    region=None,
    reference=None,
    finding="normal",
    local=False,
):
    return ReviewDecision(
        observations=[
            dict(
                text="单元测试中的可见边缘观察",
                finding=finding,
                evidence_ids=["region:R1", "matched:R1:1"]
                if local
                else ["original", "reference:1"],
                region_id="R1" if local else None,
            )
        ],
        action=action,
        verdict=verdict,
        region_id=region,
        reference_id=reference,
        summary="明确标注的协议测试结论",
    )


class FixtureClient:
    backend = "unit-test-only"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def decide(self, context, images, directory):
        self.inputs.append((context, images))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value, dict(seconds=0.001, usage={})


def run(tmp_path, outputs, score=10, regions=1):
    tools = FixtureTools(tmp_path, score, regions)
    client = FixtureClient(outputs)
    report = ReviewAgent(tools, tools.policy, client).run("fixture.png", "bottle")
    return report, client


def test_actual_graph_selects_crop_then_passes_without_rewriting_original(tmp_path):
    report, client = run(
        tmp_path, [decision("compare_region", "unknown", "R1", 1), decision(local=True)]
    )
    assert report["initial_result"]["state"] == "normal_candidate"
    assert (
        report["result"]["state"] == "normal_candidate"
        and report["agent_review"]["accepted"]
    )
    assert report["vlm_calls"] == 2
    assert len(client.inputs[0][1]) == 2 and len(client.inputs[1][1]) == 3
    assert client.inputs[1][0]["view"] == ["region", "R1", 1]
    with Image.open(client.inputs[1][1][2]["path"]) as image:
        assert image.size == (64, 72)
    assert (
        read_json(tmp_path / "baseline-result.json")["result"]["state"]
        == "manual_review"
    )
    assert any(row["tool"] == "compare_region" for row in report["trace"])


@pytest.mark.parametrize(
    "score,state", [(2, "normal_candidate"), (20, "anomaly_candidate")]
)
def test_clear_cases_do_not_call_model(tmp_path, score, state):
    report, client = run(tmp_path, [], score=score)
    assert report["result"]["state"] == state and not client.inputs


def test_skipping_crop_cannot_pass(tmp_path):
    report, _ = run(tmp_path, [decision()])
    assert report["result"]["state"] == "manual_review"
    assert "局部核查" in "".join(report["result"]["reasons"])


def test_unknown_visible_evidence_cannot_pass_even_without_regions(tmp_path):
    report, _ = run(tmp_path, [decision(finding="unclear")], regions=0)
    assert report["result"]["state"] == "manual_review"


def test_fabricated_evidence_is_rejected(tmp_path):
    value = decision()
    value.observations[0].evidence_ids = ["invented.png"]
    report, _ = run(tmp_path, [value])
    assert (
        report["result"]["state"] == "manual_review"
        and report["agent_review"]["status"] == "failed"
    )


def test_repeated_view_and_call_budget_stop(tmp_path):
    report, _ = run(tmp_path, [decision("compare_reference", "unknown", reference=1)])
    assert report["vlm_calls"] == 1 and "重复" in "".join(report["result"]["reasons"])


def test_transport_retry_is_bounded_and_detection_evidence_survives(tmp_path):
    report, _ = run(
        tmp_path, [ToolFailure("vlm_unavailable", "test transport fault", True)] * 3
    )
    assert report["vlm_calls"] == 2
    assert report["result"]["state"] == "manual_review"
    assert report["outputs"]["detection"]["score"] == 10
    assert (tmp_path / "baseline-result.json").is_file()


def test_visible_anomaly_can_finish_as_candidate(tmp_path):
    report, _ = run(tmp_path, [decision(verdict="anomaly", finding="anomaly")])
    assert report["result"]["state"] == "anomaly_candidate"
    assert report["initial_result"]["state"] == "normal_candidate"


def test_local_normal_cannot_erase_previously_visible_anomaly(tmp_path):
    report, _ = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1, finding="anomaly"),
            decision(local=True),
        ],
    )
    assert report["result"]["state"] == "manual_review" and "冲突" in "".join(
        report["result"]["reasons"]
    )


@pytest.mark.parametrize(
    "verdict,finding,evidence",
    [
        ("normal", "normal", ["reference:1"]),
        ("anomaly", "anomaly", ["reference:1"]),
        ("normal", "normal", ["original"]),
    ],
)
def test_reference_only_or_uncompared_normal_cannot_decide_query(
    tmp_path, verdict, finding, evidence
):
    value = decision(verdict=verdict, finding=finding)
    value.observations[0].evidence_ids = evidence
    report, _ = run(tmp_path, [value], regions=0)
    assert report["result"]["state"] == "manual_review"
    assert not report["agent_review"]["accepted"]


@pytest.mark.parametrize("missing_region,conflicting", [(True, False), (False, True)])
def test_invalid_local_round_is_not_published_as_success(
    tmp_path, missing_region, conflicting
):
    value = decision(local=True)
    if missing_region:
        value.observations[0].region_id = None
        value.observations[0].evidence_ids = ["original"]
    if conflicting:
        other = value.observations[0].model_copy(update=dict(finding="anomaly"))
        value.observations.append(other)
    report, _ = run(tmp_path, [decision("compare_region", "unknown", "R1", 1), value])
    assert report["result"]["state"] == "manual_review"
    rounds = report["agent_review"]["rounds"]
    assert len(rounds) == 2 and rounds[-1].get("error")
    assert "observations" not in rounds[-1]
    assert report["agent_review"]["reviewed_regions"] == {}
