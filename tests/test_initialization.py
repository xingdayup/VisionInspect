"""Deterministic timing fixtures test the cold-load boundary, not model performance."""

import pytest
import yaml

from visioninspect.agent import InspectionAgent
from visioninspect.common import ROOT
from visioninspect.tools import ToolFailure


class PreparedTools:
    threshold = 10
    profile = dict(review_lower=9, review_upper=11, digest="UNIT TEST ONLY")

    def __init__(self, clock, delay=47, quality=True, error=None, detect_delay=0):
        self.clock = clock
        self.delay = delay
        self.quality = quality
        self.error = error
        self.detect_delay = detect_delay
        self.prepared = False
        self.names = []

    def prepare_detector(self):
        self.prepared = True
        self.clock["now"] += self.delay
        if self.error:
            raise self.error
        return dict(checkpoint_load_seconds=self.delay)

    def invoke(self, name, args):
        self.names.append(name)
        if name == "inspect_image_quality":
            return dict(valid=self.quality)
        if name == "detect_anomaly":
            assert self.prepared
            self.clock["now"] += self.detect_delay
            return dict(
                score=2,
                evidence={"original": "UNIT-TEST.png"},
                anomaly_map_path="unit.npy",
            )
        if name == "finalize_inspection":
            manual = (
                not args["quality_ok"] or args["score"] is None or bool(args["errors"])
            )
            return dict(
                state="manual_review" if manual else "normal_candidate",
                reasons=args["errors"],
                evidence_paths=[],
                rule_version="unit",
            )
        raise AssertionError(name)


@pytest.fixture
def timing(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("visioninspect.agent.time.perf_counter", lambda: clock["now"])
    policy = yaml.safe_load((ROOT / "configs/agent.yaml").read_text("utf8"))
    return clock, policy


def test_cold_initialization_47_seconds_does_not_consume_45_second_tool_budget(timing):
    clock, policy = timing
    tools = PreparedTools(clock)
    updates = []
    report = InspectionAgent(tools, policy, updates.append).run(
        "UNIT TEST ONLY", "bottle"
    )
    assert report["result"]["state"] == "normal_candidate" and not report["errors"]
    assert (
        report["initialization"]["seconds"] == 47
        and report["initialization"]["status"] == "completed"
    )
    assert (
        next(row for row in report["trace"] if row["tool"] == "detect_anomaly")[
            "seconds"
        ]
        == 0
    )
    assert report["seconds"] == 47 and report["tool_calls"] == 3
    assert any(row["stage"] == "initialize_model" for row in updates)


@pytest.mark.parametrize(
    "settings,delay",
    [
        (dict(model_initialization_timeout_seconds=90), 91),
        (dict(timeout_seconds=40), 47),
    ],
)
def test_initialization_and_task_limits_fail_closed_without_detection(
    timing, settings, delay
):
    clock, policy = timing
    policy.update(settings)
    tools = PreparedTools(clock, delay=delay)
    report = InspectionAgent(tools, policy).run("UNIT TEST ONLY", "bottle")
    assert (
        report["result"]["state"] == "manual_review"
        and report["initialization"]["status"] == "failed"
    )
    assert (
        "detect_anomaly" not in tools.names
        and report["initialization"]["error"]["code"] == "timeout"
    )


def test_real_tool_timeout_is_still_enforced_after_initialization(timing):
    clock, policy = timing
    tools = PreparedTools(clock, delay=2, detect_delay=46)
    report = InspectionAgent(tools, policy).run("UNIT TEST ONLY", "bottle")
    assert (
        report["initialization"]["status"] == "completed"
        and report["result"]["state"] == "manual_review"
    )
    assert (
        next(row for row in report["trace"] if row["tool"] == "detect_anomaly")[
            "error"
        ]["code"]
        == "timeout"
    )


def test_invalid_quality_skips_expensive_initialization(timing):
    clock, policy = timing
    tools = PreparedTools(clock, quality=False)
    report = InspectionAgent(tools, policy).run("UNIT TEST ONLY", "bottle")
    assert (
        not tools.prepared
        and report["initialization"] is None
        and report["result"]["state"] == "manual_review"
    )


def test_initialization_error_never_runs_detection(timing):
    clock, policy = timing
    tools = PreparedTools(
        clock, error=ToolFailure("model_missing", "UNIT TEST MISSING WEIGHT")
    )
    report = InspectionAgent(tools, policy).run("UNIT TEST ONLY", "bottle")
    assert (
        report["initialization"]["error"]["code"] == "model_missing"
        and report["result"]["state"] == "manual_review"
    )
    assert "detect_anomaly" not in tools.names
