"""Contracts for bounded evidence-driven review; no arbitrary tool arguments."""
from typing import Literal
from pydantic import Field
from .schemas import Contract


class Observation(Contract):
    text: str = Field(min_length=1, max_length=300)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    region_id: str | None = Field(default=None, pattern=r'^R[1-4]$')
    finding: Literal['normal', 'anomaly', 'unclear']


class ReviewDecision(Contract):
    observations: list[Observation] = Field(min_length=1, max_length=8)
    action: Literal['compare_region', 'compare_reference', 'finish']
    region_id: str | None = Field(default=None, pattern=r'^R[1-4]$')
    reference_id: int | None = Field(default=None, ge=1, le=2)
    verdict: Literal['normal', 'anomaly', 'unknown']
    summary: str = Field(min_length=1, max_length=500)


REVIEW_PROMPT = '''你是图像质检研究中的证据复核 Agent。只报告实际可见的形状、边缘、纹理和异物差异。
图片有明确身份：original 是待检原图，reference 是正常训练参考；局部视图中的 query_crop 是待检区域，reference_crop 是参考图对应坐标区域。
它们不是两个待检产品。参考图不是待检图片的另一侧，也不是缺陷标签。
不能凭反光、颜色差异推断生锈、污染成因、材质、合格公差或产品安全。无法看清时 finding=unclear，最终 verdict=unknown。
每条观察必须引用本轮提供的 evidence_ids；局部观察还必须写正确 region_id，同时引用 query_crop 和 reference_crop。
可选动作：compare_region 查看一个未检查区域；compare_reference 更换整体参考图；finish 结束。
需要正常结论时，所有已提取区域必须先完成局部比较，并有一致的 normal 观察。不能省略局部检查。
每次选择一个动作，避免重复查看同一个视图。剩余调用不足以查完时结束为 unknown。
非 finish 动作的 verdict 必须为 unknown。只输出给定 schema 的 JSON，summary 和观察用中文写简短的可见依据。'''
