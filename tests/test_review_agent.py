"""Explicit synthetic protocol fixtures; these tests measure no model accuracy."""

import copy

import pytest
import yaml
from PIL import Image

from visioninspect.common import ROOT, read_json
from visioninspect.review_agent import (
    ReviewAgent,
    gate_codes,
    truncation_blocks_normal,
)
from visioninspect.review_contracts import REVIEW_PROMPT, ReviewDecision
from visioninspect.schemas import FinalizeInput
from visioninspect.tools import ToolFailure, ToolKit


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
    observed_region="R1",
):
    return ReviewDecision(
        observations=[
            dict(
                text="单元测试中的可见边缘观察",
                finding=finding,
                evidence_ids=[
                    f"region:{observed_region}",
                    f"matched:{observed_region}:1",
                ]
                if local
                else ["original", "reference:1"],
                region_id=observed_region if local else None,
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


def run(tmp_path, outputs, score=10, regions=1, policy=None):
    tools = FixtureTools(tmp_path, score, regions)
    if policy:
        merged = copy.deepcopy(tools.policy)
        for key, value in policy.items():
            if isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged[key] = value
        tools.policy = merged
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


def test_repeated_view_gets_one_corrective_call_then_stops(tmp_path):
    """A repeated view is a planning slip, so it earns exactly one retry."""
    repeated = decision("compare_reference", "unknown", reference=1)
    report, _ = run(tmp_path, [repeated, repeated])
    assert report["vlm_calls"] == 2
    assert any(row["tool"] == "duplicate_view" for row in report["trace"])
    assert "重复" in "".join(report["result"]["reasons"])


def test_corrective_call_lets_the_model_finish(tmp_path):
    """After a duplicate, the model still gets to submit its finish decision."""
    report, _ = run(
        tmp_path,
        [
            decision("compare_reference", "unknown", reference=1),
            decision(),
        ],
    )
    reasons = "".join(report["result"]["reasons"])
    assert report["vlm_calls"] == 2
    assert "重复" not in reasons
    # The region gate still applies: a normal verdict needs the region checked.
    assert "局部核查" in reasons


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


def test_truncation_vetoes_normal_only_when_the_policy_says_so():
    """The cap on extracted regions is a policy switch, not a hard rule."""
    conservative = {"regions": {"block_normal_when_truncated": True}}
    relaxed = {"regions": {"block_normal_when_truncated": False}}
    truncated = {"truncated": True}
    complete = {"truncated": False}

    assert truncation_blocks_normal(conservative, truncated) is True
    assert truncation_blocks_normal(relaxed, truncated) is False
    assert truncation_blocks_normal(conservative, complete) is False
    # A missing key means "not truncated", so old policies keep their behaviour.
    assert truncation_blocks_normal({"regions": {}}, complete) is False
    assert truncation_blocks_normal({"regions": {}}, truncated) is True


def test_context_spells_out_coverage_and_remaining_regions(tmp_path):
    """The model is told what is left instead of having to infer it."""
    _, client = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1),
            decision("compare_region", "unknown", "R2", 1, local=True),
            decision(local=True),
        ],
        regions=2,
    )
    first = client.inputs[0][0]
    assert first["remaining_regions"] == ["R1", "R2"]
    assert first["coverage"] == "0/2"

    third = client.inputs[2][0]
    assert third["remaining_regions"] == ["R2"]
    assert third["coverage"] == "1/2"


def test_four_region_case_that_used_to_fail_now_concludes(tmp_path):
    """Replay of the real failure: four normal regions, then a repeated view."""
    report, _ = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1),
            decision("compare_region", "unknown", "R2", 1, local=True),
            decision(
                "compare_region", "unknown", "R3", 1, local=True, observed_region="R2"
            ),
            decision(
                "compare_region", "unknown", "R4", 1, local=True, observed_region="R3"
            ),
            # The model re-requests R4 instead of finishing: one corrective call.
            decision(
                "compare_region", "unknown", "R4", 1, local=True, observed_region="R4"
            ),
            decision(local=True, observed_region="R4"),
        ],
        regions=4,
    )
    assert report["vlm_calls"] == 6
    assert report["result"]["state"] == "normal_candidate"
    assert report["agent_review"]["accepted"] is True


def test_silent_abstention_gets_one_request_for_a_conclusion(tmp_path):
    """A verdict of unknown on clean evidence is a default, not a judgement."""
    report, _ = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1),
            decision("finish", "unknown", local=True),
            decision(local=True),
        ],
    )
    assert report["vlm_calls"] == 3
    assert report["result"]["state"] == "normal_candidate"


def test_abstention_nudge_is_sent_exactly_once(tmp_path):
    """The follow-up question reaches the retry, not every later call."""
    report, client = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1),
            decision("finish", "unknown", local=True),
            decision(local=True),
        ],
    )
    assert report["vlm_calls"] == 3
    # call 1 selects the region, call 2 abstains, call 3 carries the follow-up.
    assert client.inputs[1][0]["abstention_nudge"] is None
    assert "unknown" in client.inputs[2][0]["abstention_nudge"]


def test_prompt_keeps_the_judgeable_parts():
    """Guard the wording that measurably changed behaviour; do not let it rot."""
    assert "判据" in REVIEW_PROMPT
    assert "反光" in REVIEW_PROMPT and "不算缺陷" in REVIEW_PROMPT
    assert "remaining_regions" in REVIEW_PROMPT
    assert "最后手段" in REVIEW_PROMPT
    assert "重复请求" in REVIEW_PROMPT


def test_gate_codes_are_machine_readable_and_prefix_failures():
    codes = gate_codes(["疑似区域数量超过提取上限", "未分类的原因"], "boom")
    assert codes[0] == "model_failure"
    assert "regions_truncated" in codes
    assert "other" in codes


def test_accepted_review_records_no_rejection_gates(tmp_path):
    """A concluded review has nothing to explain, so its gate list stays empty."""
    report, _ = run(
        tmp_path,
        [
            decision("compare_region", "unknown", "R1", 1),
            decision(local=True),
        ],
    )
    assert report["result"]["state"] == "normal_candidate"
    assert report["agent_review"]["gates"] == []


def test_corrections_share_one_bounded_pool(tmp_path):
    """Two corrections total: a duplicate and a protocol failure cannot both retry twice."""
    repeated = decision("compare_reference", "unknown", reference=1)
    report, _ = run(
        tmp_path,
        [
            repeated,
            ToolFailure("schema_error", "malformed reply", True),
            decision(local=True),
        ],
    )
    # call 1 repeats a view, call 2 fails to parse, call 3 still cites the wrong
    # region: the correction pool is spent and the loop stops instead of growing.
    assert report["vlm_calls"] == 3
    assert report["result"]["state"] == "manual_review"
    assert report["agent_review"]["gates"]


def test_unclear_region_reopens_for_one_targeted_recheck(tmp_path):
    """With the opt-in enabled, an unclear region is re-opened exactly once."""
    report, client = run(
        tmp_path,
        policy={"regions": {"reopen_unclear_regions": True}},
        outputs=[
            decision("compare_region", "unknown", "R1", 1),
            decision("compare_region", "unknown", "R2", 1, local=True),
            decision(
                "finish", "normal", local=True, observed_region="R2", finding="unclear"
            ),
            # The re-opened view is requested again, which used to be a duplicate.
            decision(
                "compare_region", "unknown", "R2", 1, local=True, observed_region="R2"
            ),
            decision(local=True, observed_region="R2"),
        ],
        regions=2,
    )
    assert report["result"]["state"] == "normal_candidate"
    assert report["agent_review"]["accepted"] is True
    assert "R2" in client.inputs[3][0]["abstention_nudge"]
