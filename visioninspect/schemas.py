"""Versioned local tool contracts, validated before and after each invocation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Contract(BaseModel):
    """Base for every tool contract: unknown fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


class ImageInput(Contract):
    """A request to inspect one local image of a registered category."""

    image_path: str = Field(min_length=1)
    category: str = Field(default="bottle", pattern=r"^[a-z_]+$")


class QualityResult(Contract):
    """Whether the input can be judged at all, plus the measured properties."""

    valid: bool
    issues: list[str]
    width: int = 0
    height: int = 0
    brightness: float = 0
    contrast: float = 0
    sharpness: float = 0


class DetectionResult(Contract):
    """Output of one CUDA PatchCore pass, including where its evidence lives."""

    score: FiniteFloat
    anomaly_map_path: str
    model_version: str
    elapsed_seconds: float = Field(ge=0)
    threshold: FiniteFloat
    evidence: dict[str, str]
    device: Literal["cuda:0"]
    peak_gpu_allocated_bytes: int = Field(ge=0)
    checkpoint_origin_device: Literal["cpu", "cuda"]


class ReferenceResult(Contract):
    """Normal training references chosen for comparison, with their similarities."""

    reference_paths: list[str]
    similarities: list[FiniteFloat]
    source_split: Literal["train"] = "train"
    similarity_floor: FiniteFloat


class RegionInput(ImageInput):
    """A request to extract suspicious regions from an existing anomaly map."""

    anomaly_map_path: str


class RegionResult(Contract):
    """Extracted regions; truncated records whether the cap hid further regions."""

    crop_paths: list[str]
    evidence_summary: list[str]
    regions: list[dict]
    pixel_threshold: FiniteFloat
    anomaly_area_ratio: float = Field(ge=0, le=1)
    total_regions: int = 0
    truncated: bool = False


class FinalizeInput(Contract):
    """Completeness report used to derive the candidate state by rule."""

    score: FiniteFloat | None
    threshold: FiniteFloat
    evidence_paths: list[str]
    errors: list[str] = Field(default_factory=list)
    quality_ok: bool = True
    ambiguous: bool = False
    conflict: bool = False


class FinalResult(Contract):
    """The rule-derived candidate plus the reasons a human may need to review it."""

    state: Literal["normal_candidate", "anomaly_candidate", "manual_review"]
    evidence_paths: list[str]
    rule_version: str
    reasons: list[str]


class ToolError(Contract):
    """A typed tool failure; retryable marks errors that may succeed on a retry."""

    code: Literal[
        "invalid_image",
        "model_missing",
        "schema_error",
        "timeout",
        "tool_failure",
        "budget_exceeded",
        "artifact_missing",
        "vlm_unavailable",
    ]
    message: str
    retryable: bool = False


TOOL_CONTRACTS = {
    "inspect_image_quality": (ImageInput, QualityResult),
    "detect_anomaly": (ImageInput, DetectionResult),
    "retrieve_normal_reference": (ImageInput, ReferenceResult),
    "analyze_suspicious_region": (RegionInput, RegionResult),
    "finalize_inspection": (FinalizeInput, FinalResult),
}

# Upper bound for the review loop's model budget. A normal verdict needs one
# call per extracted region plus a global look and a finish, so the ceiling has
# to stay above the region cap for the loop to be able to conclude at all.
MAX_REVIEW_MODEL_CALLS = 5
