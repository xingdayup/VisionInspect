"""Versioned local tool contracts, validated before and after each invocation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ImageInput(Contract):
    image_path: str = Field(min_length=1)
    category: str = Field(default='bottle',pattern=r'^[a-z_]+$')


class QualityResult(Contract):
    valid: bool
    issues: list[str]
    width: int = 0
    height: int = 0
    brightness: float = 0
    contrast: float = 0
    sharpness: float = 0


class DetectionResult(Contract):
    score: FiniteFloat
    anomaly_map_path: str
    model_version: str
    elapsed_seconds: float = Field(ge=0)
    threshold: FiniteFloat
    evidence: dict[str,str]
    device: Literal['cuda:0']
    peak_gpu_allocated_bytes: int = Field(ge=0)
    checkpoint_origin_device: Literal['cpu','cuda']


class ReferenceResult(Contract):
    reference_paths: list[str]
    similarities: list[FiniteFloat]
    source_split: Literal['train'] = 'train'
    similarity_floor: FiniteFloat


class RegionInput(ImageInput):
    anomaly_map_path: str


class RegionResult(Contract):
    crop_paths: list[str]
    evidence_summary: list[str]
    regions: list[dict]
    pixel_threshold: FiniteFloat
    anomaly_area_ratio: float = Field(ge=0,le=1)
    total_regions: int = 0
    truncated: bool = False


class FinalizeInput(Contract):
    score: FiniteFloat | None
    threshold: FiniteFloat
    evidence_paths: list[str]
    errors: list[str] = Field(default_factory=list)
    quality_ok: bool = True
    ambiguous: bool = False
    conflict: bool = False


class FinalResult(Contract):
    state: Literal['normal_candidate','anomaly_candidate','manual_review']
    evidence_paths: list[str]
    rule_version: str
    reasons: list[str]


class ToolError(Contract):
    code: Literal['invalid_image','model_missing','schema_error','timeout','tool_failure','budget_exceeded','artifact_missing','vlm_unavailable']
    message: str
    retryable: bool = False


TOOL_CONTRACTS = {
    'inspect_image_quality': (ImageInput, QualityResult),
    'detect_anomaly': (ImageInput, DetectionResult),
    'retrieve_normal_reference': (ImageInput, ReferenceResult),
    'analyze_suspicious_region': (RegionInput, RegionResult),
    'finalize_inspection': (FinalizeInput, FinalResult),
}
