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
context 的 regions 是全部疑似区域，reviewed_regions 是已查看过的区域 id，remaining_regions 是尚未查看的区域，coverage 是已完成比例。
每一轮都独立判断当前视图，不要为了与之前的判断保持一致而改变结论。
整体视图只提供 original 与 reference:N，此时 observation 的 region_id 必须为 null。
只有提供了 region:Rk 与 matched:Rk:N 的局部视图，observation 才填写 region_id，并且必须同时引用这两个 id。
判据（按顺序逐项检查，这是本任务的质检标准）：
1. 形状与边缘：待检区域轮廓是否与对应参考区域一致；
2. 纹理：是否出现参考中不存在的条纹、斑块、砂粒或凹陷；
3. 异物：是否有孤立的高对比小结构或附着物；
4. 反光、亮度、色差本身不算缺陷，除非同时伴随几何或纹理变化。
不能凭反光、颜色差异推断生锈、污染成因、材质、合格公差或产品安全。只有确实无法辨读时才写 finding=unclear。
每条观察必须引用本轮提供的 evidence_ids。整体观察必须引用 original，正常观察还必须引用本轮 reference，不能把正常参考图的状态当作待检结论。
局部观察还必须写正确 region_id，同时引用 region:Rk 和 matched:Rk:N；每轮局部检查必须有一致的区域观察。
可选动作：compare_region 查看一个未检查区域，需给出 region_id，reference_id 省略时沿用当前参考；compare_reference 更换整体参考图，必须给出 reference_id；finish 结束。
需要正常结论时，所有已提取区域必须先完成局部比较，并有一致的 normal 观察。不能省略局部检查。
每次选择一个动作。不要重复查看已经看过的视图：重复请求会被记为协议错误并浪费一次调用。
remaining_regions 为空时，必须选择 finish，不要再次请求任何区域。
已查看全部已提取区域且观察一致时，必须选择 finish 并给出明确的 normal 或 anomaly，不要停在中途。
verdict=unknown 是最后手段，仅用于证据相互矛盾或确实无法辨读的情况；它会直接把本图转交人工复核。
非 finish 动作的 verdict 必须为 unknown。只输出给定 schema 的 JSON，summary 和观察用中文写简短的可见依据。"""
