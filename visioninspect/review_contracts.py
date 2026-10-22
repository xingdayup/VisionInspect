"""Contracts for bounded evidence-driven review; no arbitrary tool arguments."""

from typing import Literal

from pydantic import Field

from .schemas import Contract

REVIEW_RULE_VERSION = "agent-review-v1.1"


class Observation(Contract):
    """One visible finding, bound to the evidence ids it was derived from."""

    text: str = Field(min_length=1, max_length=300)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    region_id: str | None = Field(default=None, pattern=r"^R[1-4]$")
    finding: Literal["normal", "anomaly", "unclear"]


class ReviewDecision(Contract):
    """One bounded review step: what was seen, and which view to look at next."""

    observations: list[Observation] = Field(min_length=1, max_length=8)
    action: Literal["compare_region", "compare_reference", "finish"]
    region_id: str | None = Field(default=None, pattern=r"^R[1-4]$")
    reference_id: int | None = Field(default=None, ge=1, le=2)
    verdict: Literal["normal", "anomaly", "unknown"]
    summary: str = Field(min_length=1, max_length=500)


REVIEW_PROMPT = """你是图像质检研究中的证据复核 Agent。只报告实际可见的形状、边缘、纹理和异物差异。
图片身份由 evidence_ids 决定：original 是待检原图；reference:N 是第 N 张正常训练参考；region:Rk 是待检区域 Rk 的裁剪；matched:Rk:N 是同一参考图在 Rk 对应坐标处的裁剪。
它们不是两个待检产品。参考图不是待检图片的另一侧，也不是缺陷标签。
整体视图只提供 original 与 reference:N，此时 observation 的 region_id 必须为 null。
只有提供了 region:Rk 与 matched:Rk:N 的局部视图，observation 才填写 region_id，并且必须同时引用这两个 id。
不能凭反光、颜色差异推断生锈、污染成因、材质、合格公差或产品安全。无法看清时 finding=unclear，最终 verdict=unknown。
每条观察必须引用本轮提供的 evidence_ids。整体观察必须引用 original，正常观察还必须引用本轮 reference，不能把正常参考图的状态当作待检结论。
局部观察还必须写正确 region_id，同时引用 region:Rk 和 matched:Rk:N；每轮局部检查必须有一致的区域观察。
可选动作：compare_region 查看一个未检查区域；compare_reference 更换整体参考图；finish 结束。
需要正常结论时，所有已提取区域必须先完成局部比较，并有一致的 normal 观察。不能省略局部检查。
每次选择一个动作，避免重复查看同一个视图。剩余调用不足以查完时结束为 unknown。
非 finish 动作的 verdict 必须为 unknown。只输出给定 schema 的 JSON，summary 和观察用中文写简短的可见依据。"""
