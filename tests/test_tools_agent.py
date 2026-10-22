import numpy as np
import pytest
import yaml
from PIL import Image
from visioninspect.common import ROOT
from visioninspect.tools import ToolKit, ToolFailure
from visioninspect.agent import InspectionAgent


def policy():
    return yaml.safe_load((ROOT / "configs/agent.yaml").read_text(encoding="utf8"))


def toolkit(tmp_path):
    from visioninspect.service import registry

    return ToolKit(registry()["bottle"]["patchcore"], policy(), tmp_path)


def test_quality_and_invalid_tool_fail_closed(tmp_path):
    tools = toolkit(tmp_path)
    path = tmp_path / "flat.png"
    Image.new("RGB", (256, 256), "gray").save(path)
    assert not tools.invoke("inspect_image_quality", dict(image_path=str(path)))[
        "valid"
    ]
    with pytest.raises(ToolFailure):
        tools.invoke("anything", {})
    with pytest.raises(ToolFailure):
        tools.invoke(
            "inspect_image_quality", dict(image_path=str(tmp_path / "missing.png"))
        )


def test_regions_and_optional_candidate_are_honest(tmp_path):
    tools = toolkit(tmp_path)
    path = tmp_path / "image.png"
    Image.new("RGB", (900, 900), "gray").save(path)
    heat = np.zeros((256, 256), np.float32)
    heat[10:20, 30:50] = tools.profile["pixel_threshold"] + 1
    np.save(tmp_path / "map.npy", heat)
    result = tools.invoke(
        "analyze_suspicious_region",
        dict(image_path=str(path), anomaly_map_path="map.npy"),
    )
    assert len(result["regions"]) == 1 and result["regions"][0]["pixels"] == 200
    assert (tmp_path / result["crop_paths"][0]).is_file()
    with pytest.raises(ToolFailure):
        tools.invoke(
            "analyze_suspicious_region",
            dict(image_path=str(path), anomaly_map_path="../map.npy"),
        )


class FakeTools:
    """Unit-test double, never used in experiments or API."""

    threshold = 10
    profile = dict(review_lower=9, review_upper=11, digest="unit-test")

    def __init__(self, score=2, bad=False, fail=None):
        self.score = score
        self.bad = bad
        self.fail = fail
        self.policy = policy()

    def invoke(self, name, args):
        if name == self.fail:
            raise ToolFailure("tool_failure", "injected unit-test failure")
        if name == "inspect_image_quality":
            return dict(valid=not self.bad)
        if name == "detect_anomaly":
            return dict(
                score=self.score,
                evidence={"original": "o.png"},
                anomaly_map_path="map.npy",
            )
        if name == "analyze_suspicious_region":
            return dict(crop_paths=["crop.png"])
        if name == "retrieve_normal_reference":
            return dict(
                reference_paths=["ref.png"], similarities=[1], similarity_floor=0.8
            )
        if name == "finalize_inspection":
            from visioninspect.schemas import FinalizeInput

            return ToolKit.finalize_inspection(self, FinalizeInput.model_validate(args))
        raise AssertionError(name)


def test_conditional_calls_and_ambiguous_degradation():
    easy = InspectionAgent(FakeTools(), policy()).run("unit.png", "bottle")
    hard = InspectionAgent(FakeTools(score=10), policy()).run("unit.png", "bottle")
    assert easy["tool_calls"] == 3 and easy["result"]["state"] == "normal_candidate"
    assert hard["tool_calls"] == 5 and hard["result"]["state"] == "manual_review"
    assert all(row["reason"] and row["status"] == "completed" for row in hard["trace"])


@pytest.mark.parametrize("bad,fail", [(True, None), (False, "detect_anomaly")])
def test_quality_and_tool_failures_never_pass(bad, fail):
    result = InspectionAgent(FakeTools(bad=bad, fail=fail), policy()).run(
        "unit.png", "bottle"
    )
    assert result["result"]["state"] == "manual_review"


def test_call_budget_still_finalizes_safely():
    rules = policy()
    rules["max_tool_calls"] = 2
    result = InspectionAgent(FakeTools(), rules).run("unit.png", "bottle")
    assert result["tool_calls"] <= 2 and result["result"]["state"] == "manual_review"


def test_transient_failure_recovers_with_one_bounded_retry():
    class Transient(FakeTools):
        attempts = 0

        def invoke(self, name, args):
            if name == "detect_anomaly":
                self.attempts += 1
                if self.attempts == 1:
                    raise ToolFailure(
                        "tool_failure", "temporary unit-test failure", True
                    )
            return super().invoke(name, args)

    result = InspectionAgent(Transient(), policy()).run("unit.png", "bottle")
    assert result["result"]["state"] == "normal_candidate"
    assert result["tool_calls"] == 4 and result["trace"][1]["status"] == "failed"
    assert result["trace"][2]["attempt"] == 2 and not result["errors"]


def test_tool_timeout_invalidates_otherwise_successful_output():
    rules = policy()
    rules["tool_timeout_seconds"] = 0
    result = InspectionAgent(FakeTools(), rules).run("unit.png", "bottle")
    assert result["result"]["state"] == "manual_review"
    assert result["trace"][0]["error"]["code"] == "timeout"


def test_category_mismatch_fails_before_image_or_model_loading(tmp_path):
    tools = toolkit(tmp_path)
    with pytest.raises(ToolFailure, match="类别"):
        tools.invoke(
            "detect_anomaly", dict(image_path="does-not-exist.png", category="pill")
        )


@pytest.mark.parametrize(
    "case,score,bad,fail,mode,region_available,reference_ok,expected_state,expected_tools",
    [
        (
            "confident_normal",
            2,
            False,
            None,
            "agent_review",
            True,
            True,
            "normal_candidate",
            ["quality", "detect", "final"],
        ),
        (
            "confident_anomaly",
            20,
            False,
            None,
            "agent_review",
            True,
            True,
            "anomaly_candidate",
            ["quality", "detect", "regions", "final"],
        ),
        (
            "boundary",
            10,
            False,
            None,
            "agent_review",
            True,
            True,
            "manual_review",
            ["quality", "detect", "regions", "references", "final"],
        ),
        (
            "poor_quality",
            2,
            True,
            None,
            "agent_review",
            True,
            True,
            "manual_review",
            ["quality", "final"],
        ),
        (
            "detector_failure",
            2,
            False,
            "detect_anomaly",
            "agent_review",
            True,
            True,
            "manual_review",
            ["quality", "detect", "final"],
        ),
        (
            "empty_high_score_region",
            20,
            False,
            None,
            "agent_review",
            False,
            True,
            "manual_review",
            ["quality", "detect", "regions", "final"],
        ),
        (
            "weak_normal_reference",
            9.2,
            False,
            None,
            "agent_review",
            True,
            False,
            "manual_review",
            ["quality", "detect", "regions", "references", "final"],
        ),
    ],
)
def test_predefined_routing_oracle(
    case,
    score,
    bad,
    fail,
    mode,
    region_available,
    reference_ok,
    expected_state,
    expected_tools,
    record_property,
):
    class CaseTools(FakeTools):
        def invoke(self, name, args):
            result = super().invoke(name, args)
            if name == "analyze_suspicious_region" and not region_available:
                result["crop_paths"] = []
            if name == "retrieve_normal_reference" and not reference_ok:
                result["similarities"] = [0.1]
            return result

    names = {
        "inspect_image_quality": "quality",
        "detect_anomaly": "detect",
        "analyze_suspicious_region": "regions",
        "retrieve_normal_reference": "references",
        "finalize_inspection": "final",
    }
    result = InspectionAgent(CaseTools(score, bad, fail), policy()).run(
        "UNIT-TEST-ONLY.png", "bottle", mode
    )
    assert result["result"]["state"] == expected_state
    assert [names[row["tool"]] for row in result["trace"]] == expected_tools
    record_property("routing_oracle_case", case)
    record_property("vision_accuracy_evaluation", "false")
