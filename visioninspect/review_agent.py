"""LangGraph review loop with evidence binding and deterministic candidate gates.

The graph has four nodes:

    baseline -> decide -> tool -> decide ... -> finish -> END

Only an uncertain baseline score pays for a model call. Each decide step must
quote evidence this round actually supplied, and finish applies deterministic
gates, so a confident-sounding model cannot promote a candidate on its own.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from PIL import Image

from .agent import InspectionAgent
from .common import write_json
from .review_contracts import REVIEW_RULE_VERSION, ReviewDecision
from .schemas import MAX_REVIEW_MODEL_CALLS, FinalResult
from .tools import ToolFailure

if TYPE_CHECKING:
    from .tools import ToolKit
    from .vllm_client import VLLMClient


def truncation_blocks_normal(policy: dict, regions_output: dict) -> bool:
    """Decide whether an unextracted region must veto a normal verdict.

    Conservative by default: a capped extraction may hide a defect. Operators can
    relax it when the cap is routinely hit by texture noise on good parts, at the
    cost of trusting only the regions that were actually inspected.
    """
    if not regions_output.get("truncated"):
        return False
    return bool(policy["regions"].get("block_normal_when_truncated", True))


class ReviewState(TypedDict, total=False):
    """Single-key graph state: nodes return the next route only."""

    route: str


class ReviewAgent:
    """Run the bounded evidence-review loop around one baseline inspection.

    Every model call is budgeted (three by default, retries included). Regions
    and references are fixed by the baseline stage; the model may only choose
    which supplied view to look at next, or to finish.
    """

    def __init__(
        self,
        tools: ToolKit,
        policy: dict,
        client: VLLMClient,
        progress: Callable[[dict], None] | None = None,
        max_model_calls: int = MAX_REVIEW_MODEL_CALLS,
    ) -> None:
        self.tools, self.policy, self.client = tools, policy, client
        self.progress = progress or (lambda value: None)
        self.maximum = max(1, min(MAX_REVIEW_MODEL_CALLS, max_model_calls))
        self.calls = 0
        self.rounds: list[dict] = []
        self.reviewed: dict[str, str] = {}
        self.visited: set[tuple] = set()
        self.current: tuple = ("global", None, 1)
        self.retry_count = 0
        self.duplicate_retries = 0
        self.failure: str | None = None
        self.last: ReviewDecision | None = None

    def publish(self, stage: str) -> None:
        """Push a progress snapshot so the UI can show live review state."""
        self.progress(
            dict(
                stage=stage,
                trace=self.report["trace"],
                agent_review=dict(
                    calls=self.calls,
                    maximum_calls=self.maximum,
                    reviewed_regions=sorted(self.reviewed),
                    rounds=self.rounds,
                ),
            )
        )

    def baseline(self, state: ReviewState) -> dict[str, str]:
        """Run detection, record the raw candidate, and decide whether to review.

        Detection evidence is written to disk before any model call so a later
        failure cannot erase it.
        """
        self.report = InspectionAgent(self.tools, self.policy, self.progress).run(
            self.image_path, self.category, "agent_review"
        )
        self.report.update(schema_version=2, mode="agent_review")
        detection = self.report["outputs"].get("detection")
        self.regions = {
            r["id"]: r
            for r in (self.report["outputs"].get("regions") or {}).get("regions", [])
        }
        self.references = (self.report["outputs"].get("references") or {}).get(
            "reference_paths", []
        )
        eligible = bool(
            detection
            and self.tools.profile["review_lower"]
            <= detection["score"]
            <= self.tools.profile["review_upper"]
        )
        initial = dict(self.report["result"])
        if detection and not self.report["errors"]:
            initial = FinalResult(
                state="anomaly_candidate"
                if detection["score"] > detection["threshold"]
                else "normal_candidate",
                evidence_paths=list(self.report["result"]["evidence_paths"]),
                rule_version=self.policy["rule_version"],
                reasons=["依据冻结图像阈值生成的原始检测候选；Agent 复核结果另存"],
            ).model_dump()
        self.report["initial_result"] = initial
        self.report["review_eligible"] = eligible
        self.report["agent_review"] = dict(
            status="skipped", reason="未进入不确定区间或检测未完成", rounds=[], calls=0
        )
        write_json(self.tools.output / "baseline-result.json", self.report)
        if not eligible or self.report["errors"]:
            return dict(route="finish")
        if not self.references:
            self.failure = "正常训练参考图缺失"
            return dict(route="finish")
        return dict(route="decide")

    def images(self) -> list[dict]:
        """Materialize the 1-3 images that constitute the current view.

        A region view always pairs the query crop with the same coordinates in
        the reference image, so the model compares like with like.
        """
        kind, region_id, reference_id = self.current
        root = self.tools.output
        if kind == "global":
            return [
                dict(id="original", path=root / "evidence/original.png"),
                dict(
                    id=f"reference:{reference_id}",
                    path=root / self.references[reference_id - 1],
                ),
            ]
        region = self.regions[region_id]
        reference = root / self.references[reference_id - 1]
        destination = root / "evidence" / f"review_{region_id}_ref_{reference_id}.png"
        with Image.open(reference) as image:
            box = [
                round(value * image.size[index % 2] / 256)
                for index, value in enumerate(region["box_grid"])
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ToolFailure("artifact_missing", "正常参考区域坐标无效")
            image.convert("RGB").crop(box).save(destination)
        return [
            dict(id="original", path=root / "evidence/original.png"),
            dict(id=f"region:{region_id}", path=root / region["crop_path"]),
            dict(id=f"matched:{region_id}:{reference_id}", path=destination),
        ]

    def validate(self, decision: ReviewDecision, images: list[dict]) -> None:
        """Enforce evidence binding before a decision is allowed to count.

        Every observation must cite evidence ids that were actually supplied,
        and a normal verdict must come with a matching local comparison for
        every extracted region.
        """
        allowed = {item["id"] for item in images}
        kind, region_id, reference_id = self.current
        for observation in decision.observations:
            if not set(observation.evidence_ids).issubset(allowed):
                raise ToolFailure("schema_error", "模型引用了本轮未提供的图像证据")
            if kind == "global":
                if "original" not in observation.evidence_ids:
                    raise ToolFailure(
                        "schema_error",
                        "整体观察必须引用待检原图，不能以参考图代替待检证据",
                    )
                if (
                    observation.finding == "normal"
                    and f"reference:{reference_id}" not in observation.evidence_ids
                ):
                    raise ToolFailure(
                        "schema_error",
                        "正常整体观察必须同时引用待检原图和本轮正常参考图",
                    )
            if observation.region_id is not None:
                required = {
                    f"region:{region_id}",
                    f"matched:{region_id}:{reference_id}",
                }
                if (
                    region_id is None
                    or observation.region_id != region_id
                    or not required.issubset(observation.evidence_ids)
                ):
                    raise ToolFailure(
                        "schema_error", "局部观察未绑定正确的待检和参考区域"
                    )
        if kind == "region":
            findings = [
                o.finding for o in decision.observations if o.region_id == region_id
            ]
            if not findings or len(set(findings)) != 1:
                raise ToolFailure("schema_error", "本轮局部检查缺少一致的区域观察")
        if decision.action != "finish" and decision.verdict != "unknown":
            raise ToolFailure("schema_error", "补充检查动作不能同时提交最终结论")
        if decision.action == "compare_region":
            # reference_id is optional in the contract: an omitted value means
            # "the reference this view already uses", which is what the model
            # sees and what a reader would assume.
            if decision.reference_id is None:
                decision.reference_id = reference_id
            if (
                decision.region_id not in self.regions
                or decision.reference_id not in range(1, len(self.references) + 1)
            ):
                raise ToolFailure("schema_error", "模型请求了不存在的区域或参考图")
        if decision.action == "compare_reference":
            if decision.region_id is not None or decision.reference_id not in range(
                1, len(self.references) + 1
            ):
                raise ToolFailure("schema_error", "正常参考切换参数无效")
        if decision.action == "finish" and (
            decision.region_id is not None or decision.reference_id is not None
        ):
            raise ToolFailure("schema_error", "结束动作不得携带新的工具参数")

    def silent_abstention(self, decision: ReviewDecision) -> bool:
        """True when the model abstains despite having complete, clean evidence.

        Abstention is meant for genuine conflicts or unreadable evidence. Using
        it after every region came back normal is a default, not a judgement, so
        the loop asks for the conclusion once before falling back to a human.
        """
        if decision.verdict != "unknown":
            return False
        if any(value != "normal" for value in self.reviewed.values()):
            return False
        return all(region in self.reviewed for region in self.regions)

    def decide(self, state: ReviewState) -> dict[str, str]:
        """Ask the model for one bounded next step, then validate the reply."""
        started = time.perf_counter()
        self.calls += 1
        trace = dict(
            index=len(self.report["trace"]) + 1,
            tool="review_evidence",
            reason="根据当前视觉证据选择补充检查或结束复核",
            input=dict(view=list(self.current), round=self.calls),
            attempt=1,
            status="running",
        )
        self.report["trace"].append(trace)
        self.publish("agent_decision")
        images = []
        try:
            images = self.images()
            context = dict(
                category=self.category,
                detection=self.report["outputs"]["detection"],
                view=list(self.current),
                available_evidence=[i["id"] for i in images],
                regions=[
                    dict(
                        id=r["id"],
                        box_grid=r["box_grid"],
                        maximum_score=r["maximum_score"],
                    )
                    for r in self.regions.values()
                ],
                reference_ids=list(range(1, len(self.references) + 1)),
                reviewed_regions=self.reviewed,
                # Spell the coverage out: the model should not have to infer the
                # remaining work from a list, which is where it kept going wrong.
                remaining_regions=[i for i in self.regions if i not in self.reviewed],
                coverage=f"{len(self.reviewed)}/{len(self.regions)}"
                if self.regions
                else "0/0",
                previous_observations=[
                    o for item in self.rounds for o in item.get("observations", [])
                ],
                calls_remaining_after_this=self.maximum - self.calls,
            )
            decision, metrics = self.client.decide(
                context, images, self.tools.output / "vlm_input" / f"round_{self.calls}"
            )
            self.validate(decision, images)
            data = decision.model_dump()
            trace.update(status="completed", output=dict(**data, **metrics))
            self.rounds.append(
                dict(round=self.calls, view=list(self.current), **data, **metrics)
            )
            self.visited.add(self.current)
            self.last = decision
            if self.current[0] == "region":
                findings = [
                    o.finding
                    for o in decision.observations
                    if o.region_id == self.current[1]
                ]
                self.reviewed[self.current[1]] = findings[0]
            if decision.action == "finish":
                if self.silent_abstention(decision):
                    # Every region was checked and nothing abnormal was seen, yet
                    # the model still abstains. Ask once for the conclusion it is
                    # already holding instead of defaulting the case to a human.
                    raise ToolFailure(
                        "schema_error",
                        "已查看全部区域且无异常观察，请给出 normal 或 anomaly 结论",
                        retryable=True,
                    )
                return dict(route="finish")
            if self.calls >= self.maximum:
                self.failure = "模型调用预算耗尽，补充检查未完成"
                return dict(route="finish")
            return dict(route="tool")
        except ToolFailure as error:
            trace.update(
                status="failed",
                error=dict(
                    code=error.code, message=error.message, retryable=error.retryable
                ),
            )
            self.rounds.append(
                dict(round=self.calls, view=list(self.current), error=trace["error"])
            )
            # Schema errors are model formatting slips, not evidence
            # contradictions, so one bounded retry is worth spending.
            recoverable = error.retryable or error.code == "schema_error"
            if recoverable and self.retry_count < 1 and self.calls < self.maximum:
                self.retry_count += 1
                return dict(route="decide")
            self.failure = f"{error.code}: {error.message}"
            return dict(route="finish")
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            trace.update(
                status="failed",
                error=dict(code="tool_failure", message=self.failure, retryable=False),
            )
            return dict(route="finish")
        finally:
            trace["seconds"] = time.perf_counter() - started
            self.publish("agent_decision")

    def tool(self, state: ReviewState) -> dict[str, str]:
        """Switch to the requested view, refusing to revisit an already seen one."""
        decision = self.last
        view = (
            ("region", decision.region_id, decision.reference_id)
            if decision.action == "compare_region"
            else ("global", None, decision.reference_id)
        )
        if view in self.visited:
            # The model often re-requests a view it has already inspected instead
            # of finishing. That is a planning slip, not an evidence conflict, so
            # give it exactly one corrective call before giving up.
            if self.duplicate_retries < 1 and self.calls < self.maximum:
                self.duplicate_retries += 1
                self.report["trace"].append(
                    dict(
                        index=len(self.report["trace"]) + 1,
                        tool="duplicate_view",
                        reason="模型重复请求已查看的视图，要求重新选择或结束",
                        input=dict(
                            region_id=decision.region_id,
                            reference_id=decision.reference_id,
                        ),
                        output=dict(retried=True),
                        attempt=1,
                        status="failed",
                        seconds=0,
                    )
                )
                self.publish("duplicate_view")
                return dict(route="decide")
            self.failure = "Agent 请求重复视图，已停止循环"
            return dict(route="finish")
        self.current = view
        self.report["trace"].append(
            dict(
                index=len(self.report["trace"]) + 1,
                tool=decision.action,
                reason=decision.summary,
                input=dict(
                    region_id=decision.region_id, reference_id=decision.reference_id
                ),
                output=dict(next_view=list(view)),
                attempt=1,
                status="completed",
                seconds=0,
            )
        )
        self.publish(decision.action)
        return dict(route="decide")

    def finish(self, state: ReviewState) -> dict[str, str]:
        """Apply the deterministic gates and record the final candidate.

        A normal verdict is only accepted when every extracted region was
        compared, no anomaly observation was recorded, the references were
        similar enough, and the region extraction was not truncated.
        """
        if self.report["review_eligible"]:
            reasons = []
            final_state = "manual_review"
            if self.failure:
                reasons.append(self.failure)
                self.report["errors"].append(self.failure)
            elif (
                self.last is None
                or self.last.action != "finish"
                or self.last.verdict == "unknown"
            ):
                reasons.append("复核证据不足，Agent 未给出可接受的结论")
            else:
                all_findings = [
                    o["finding"]
                    for item in self.rounds
                    for o in item.get("observations", [])
                ]
                if self.last.verdict == "normal":
                    references = self.report["outputs"].get("references") or {}
                    regions_output = self.report["outputs"].get("regions") or {}
                    if any(
                        self.reviewed.get(region) != "normal" for region in self.regions
                    ):
                        reasons.append("尚有未完成正常局部核查的疑似区域")
                    if truncation_blocks_normal(self.policy, regions_output):
                        reasons.append("疑似区域数量超过提取上限")
                    if "anomaly" in all_findings:
                        reasons.append("正常结论与已记录异常观察冲突")
                    if not any(
                        o.finding == "normal" for o in self.last.observations
                    ) or any(o.finding == "unclear" for o in self.last.observations):
                        reasons.append("最终观察未形成明确的正常视觉依据")
                    if (
                        max(references.get("similarities", []), default=-1)
                        < self.tools.profile["reference_similarity_floor"]
                    ):
                        reasons.append("正常参考相似度低于冻结策略下限")
                    if not reasons:
                        final_state = "normal_candidate"
                elif "anomaly" not in all_findings:
                    reasons.append("异常结论缺少可见异常观察")
                elif (
                    "normal" in self.reviewed.values()
                    and "anomaly" not in self.reviewed.values()
                ):
                    reasons.append("整体异常结论与已检查局部证据冲突")
                else:
                    final_state = "anomaly_candidate"
                if not reasons:
                    reasons.append(self.last.summary)
            self.report["result"] = FinalResult(
                state=final_state,
                evidence_paths=self.report["result"]["evidence_paths"],
                rule_version=REVIEW_RULE_VERSION,
                reasons=reasons,
            ).model_dump()
            self.report["agent_review"] = dict(
                status="failed" if self.failure else "completed",
                calls=self.calls,
                maximum_calls=self.maximum,
                rounds=self.rounds,
                reviewed_regions=self.reviewed,
                verdict=self.last.verdict if self.last else "unknown",
                accepted=final_state != "manual_review",
                policy_version=REVIEW_RULE_VERSION,
                backend=self.client.backend,
                reasons=reasons,
            )
        self.report.update(
            notice="研究质检候选结果，Agent 复核与人工记录独立保存；不构成工业放行授权。",
            vlm_calls=self.calls,
            tool_calls=len(self.report["trace"]),
            seconds=time.perf_counter() - self.started,
            stop_reason="manual_review_required"
            if self.report["result"]["state"] == "manual_review"
            else "rules_complete",
            human_review=None,
            effective_result=dict(
                self.report["result"], source="agent" if self.calls else "baseline"
            ),
        )
        self.publish("completed")
        return dict(route="end")

    def run(self, image_path: str | Path, category: str) -> dict:
        """Compile and execute the graph; the compiled report is the return value."""
        from langgraph.graph import END, START, StateGraph

        self.image_path, self.category = str(image_path), category
        self.started = time.perf_counter()
        graph = StateGraph(ReviewState)
        for name in ("baseline", "decide", "tool", "finish"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "baseline")
        for name in ("baseline", "decide", "tool"):
            graph.add_conditional_edges(
                name,
                lambda state: state["route"],
                dict(decide="decide", tool="tool", finish="finish"),
            )
        graph.add_edge("finish", END)
        graph.compile().invoke({}, config=dict(recursion_limit=16))
        return self.report
