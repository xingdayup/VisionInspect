"""Local visual tools. Assets are validated; tools never download or execute arbitrary names.

Every public method here is callable through invoke(), which validates the
request and the response against a versioned contract in schemas.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from .common import identity, load_config, project_path, read_json, sha256
from .schemas import TOOL_CONTRACTS

if TYPE_CHECKING:
    from .model import Detector
    from .model_cache import ModelCache
    from .schemas import FinalizeInput, ImageInput, RegionInput


def quality_metrics(image: Image.Image) -> dict[str, float | int]:
    """Measure the cheap image properties used to reject unusable inputs."""
    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return dict(
        width=image.width,
        height=image.height,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


class ToolFailure(Exception):
    """A typed tool error; retryable distinguishes transient from permanent."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(message)


class ToolKit:
    """The complete local tool surface for one inspection task.

    The toolkit owns no scheduling and no policy decisions: it validates assets,
    runs the detector, and reports evidence. Deciding what the evidence means
    happens in agent.py and review_agent.py.
    """

    def __init__(
        self,
        registration: dict,
        policy: dict,
        output: Path,
        cache: ModelCache | None = None,
    ) -> None:
        self.registration = registration
        self.policy = policy
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.profile = read_json(project_path(registration["profile"]))
        if (
            identity(
                {key: value for key, value in self.profile.items() if key != "digest"}
            )
            != self.profile["digest"]
        ):
            raise ToolFailure("model_missing", "校准策略校验失败")
        self._detector: Detector | None = None
        self._references: tuple[np.ndarray, list[str], list[str]] | None = None
        self.initialization_timings: dict[str, object] = {}
        self.cache = cache
        if cache is not None:
            cache.bind_detector(registration, self.profile)
        self.threshold = self.profile["image_threshold"]

    def detector(self) -> Detector:
        """Validate and load the detector once per process, then reuse it.

        The checkpoint is hash-checked against the calibration profile before
        loading, and the memory bank must land on CUDA; no CPU fallback exists.
        """
        if self._detector is None:
            if self.cache is not None and self.cache.detector is not None:
                self._detector = self.cache.detector
                self.initialization_timings = dict(
                    cache_hit=True,
                    import_seconds=0.0,
                    checkpoint_verification_seconds=0.0,
                    checkpoint_load_seconds=0.0,
                    runtime_setup_seconds=0.0,
                )
                return self._detector
            started = time.perf_counter()
            from .model import Detector, setup

            imported = time.perf_counter()
            checkpoint = project_path(self.registration["fit_run"]) / "model.ckpt"
            if not checkpoint.is_file():
                raise ToolFailure("model_missing", "模型权重缺失")
            if sha256(checkpoint) != self.profile["checkpoint_sha256"]:
                raise ToolFailure("model_missing", "模型校验值与校准策略不一致")
            verified = time.perf_counter()
            detector, metadata = Detector.load(
                checkpoint, load_config(project_path(self.registration["config"]))
            )
            loaded = time.perf_counter()
            if metadata["manifest_digest"] != self.profile["manifest_digest"]:
                raise ToolFailure("model_missing", "模型与参考图索引不一致")
            setup(detector.config)
            self.initialization_timings = dict(
                cache_hit=False,
                import_seconds=imported - started,
                checkpoint_verification_seconds=verified - imported,
                checkpoint_load_seconds=loaded - verified,
                runtime_setup_seconds=time.perf_counter() - loaded,
            )
            self._detector = detector
            if self.cache is not None:
                self.cache.detector = detector
        return self._detector

    def prepare_detector(self) -> dict[str, object]:
        """Cold imports/checkpoint loading are a separately budgeted initialization stage."""
        self.detector()
        return dict(self.initialization_timings)

    def image(self, path: str | Path) -> Image.Image:
        """Open an image under the pixel cap and normalize orientation to RGB."""
        try:
            with Image.open(path) as image:
                if (
                    image.width * image.height
                    > self.policy["quality"]["maximum_pixels"]
                ):
                    raise ToolFailure("invalid_image", "图像像素数量超过限制")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ToolFailure("invalid_image", "图像不存在、损坏或无法解码") from error

    def invoke(self, name: str, arguments: dict) -> dict:
        """Validate, dispatch and re-validate one tool call by contract name."""
        if name not in TOOL_CONTRACTS:
            raise ToolFailure("schema_error", "未知工具")
        input_type, output_type = TOOL_CONTRACTS[name]
        request = input_type.model_validate(arguments)
        if (
            hasattr(request, "category")
            and request.category != self.profile["category"]
        ):
            raise ToolFailure("schema_error", "工具请求类别与模型校准策略不一致")
        return output_type.model_validate(getattr(self, name)(request)).model_dump(
            mode="json"
        )

    def inspect_image_quality(self, request: ImageInput) -> dict:
        """Reject images the model cannot interpret, before any GPU work."""
        image = self.image(request.image_path)
        metrics = quality_metrics(image)
        limits = self.policy["quality"]
        issues = []
        if min(image.size) < limits["minimum_side"]:
            issues.append("图像分辨率过低")
        if metrics["contrast"] < limits["minimum_contrast"]:
            issues.append("图像缺少可辨识纹理")
        if metrics["sharpness"] < self.profile["quality"]["sharpness_floor"]:
            issues.append("清晰度低于正常验证样本范围")
        if (
            not self.profile["quality"]["brightness_low"]
            <= metrics["brightness"]
            <= self.profile["quality"]["brightness_high"]
        ):
            issues.append("曝光超出正常验证样本范围")
        return dict(valid=not issues, issues=issues, **metrics)

    def detect_anomaly(self, request: ImageInput) -> dict:
        """Run PatchCore on CUDA and persist the score, map and renders."""
        import torch

        from .artifacts import save_evidence

        array = np.asarray(self.image(request.image_path))
        score, anomaly_map, seconds = self.detector().predict(array)
        evidence = save_evidence(
            self.output / "evidence", "inspection", array, anomaly_map
        )
        Image.fromarray(array).save(self.output / "evidence/original.png")
        return dict(
            score=score,
            threshold=self.threshold,
            anomaly_map_path="evidence/inspection.npy",
            device="cuda:0",
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(
                self.detector().device
            ),
            checkpoint_origin_device=self.detector().source_device,
            model_version=self.profile["checkpoint_sha256"][:12],
            elapsed_seconds=seconds,
            evidence=dict(
                original="evidence/original.png",
                heatmap="evidence/" + evidence["heatmap"],
                overlay="evidence/" + evidence["overlay"],
            ),
        )

    def embedding(self, array: np.ndarray) -> np.ndarray:
        """Return an L2-normalized mean feature vector for reference matching."""
        import torch

        detector = self.detector()
        with torch.inference_mode():
            features = detector.model.feature_extractor(detector.tensor(array))
            vector = torch.cat(
                [
                    features[layer].mean(dim=(2, 3))
                    for layer in detector.config["model"]["layers"]
                ],
                dim=1,
            )
        result = vector[0].detach().cpu().numpy().astype(np.float32)
        return result / max(float(np.linalg.norm(result)), 1e-12)

    def retrieve_normal_reference(self, request: ImageInput) -> dict:
        """Pick the two closest normal training images and copy them as evidence.

        Each candidate must resolve inside the dataset root, come from the
        train/good split, and match its recorded hash; anything else is an error.
        """
        index_path = project_path(self.registration["references"])
        if (
            not index_path.is_file()
            or sha256(index_path) != self.profile["reference_index_sha256"]
        ):
            raise ToolFailure("artifact_missing", "正常参考图索引缺失或校验失败")
        if self._references is None:
            if self.cache is not None and self.cache.references is not None:
                self._references = self.cache.references
            else:
                with np.load(index_path, allow_pickle=False) as data:
                    self._references = (
                        data["embeddings"].copy(),
                        data["paths"].tolist(),
                        data["hashes"].tolist(),
                    )
                if self.cache is not None:
                    self.cache.references = self._references
        embeddings, paths, hashes = self._references
        scores = embeddings @ self.embedding(np.asarray(self.image(request.image_path)))
        chosen = np.argsort(-scores, kind="stable")[:2]
        source = project_path(
            self.registration.get("data_root", self.profile["data_root"])
        )
        saved = []
        directory = self.output / "evidence"
        directory.mkdir(exist_ok=True)
        for rank, index in enumerate(chosen):
            path = (source / paths[index]).resolve()
            if (
                not path.is_relative_to(source.resolve())
                or not paths[index].startswith("train/good/")
                or sha256(path) != hashes[index]
            ):
                raise ToolFailure("artifact_missing", "参考图不是已校验的正常训练样本")
            destination = directory / f"reference_{rank + 1}.png"
            self.image(str(path)).save(destination)
            saved.append(destination.relative_to(self.output).as_posix())
        return dict(
            reference_paths=saved,
            similarities=[float(scores[index]) for index in chosen],
            source_split="train",
            similarity_floor=self.profile["reference_similarity_floor"],
        )

    def analyze_suspicious_region(self, request: RegionInput) -> dict:
        """Extract connected high-score regions as crops for the review loop.

        Crops are capped by policy, and truncated reports the true region count
        so a caller cannot mistake the cap for the full extent of the anomaly.
        """
        import cv2

        path = (self.output / request.anomaly_map_path).resolve()
        if not path.is_relative_to(self.output.resolve()) or not path.is_file():
            raise ToolFailure("artifact_missing", "异常热图缺失或路径无效")
        anomaly_map = np.load(path, allow_pickle=False)
        if anomaly_map.shape != (256, 256) or not np.isfinite(anomaly_map).all():
            raise ToolFailure("artifact_missing", "异常热图尺寸或数值无效")
        binary = (anomaly_map > self.profile["pixel_threshold"]).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        original = self.image(request.image_path)
        regions = []
        directory = self.output / "evidence"
        directory.mkdir(exist_ok=True)
        indices = sorted(
            range(1, count), key=lambda index: int(stats[index, 4]), reverse=True
        )
        for index in indices:
            x, y, width, height, area = (int(value) for value in stats[index])
            if area < self.policy["regions"]["minimum_pixels"]:
                continue
            box = [
                max(0, x - 3),
                max(0, y - 3),
                min(256, x + width + 3),
                min(256, y + height + 3),
            ]
            original_box = [
                int(round(box[0] * original.width / 256)),
                int(round(box[1] * original.height / 256)),
                int(round(box[2] * original.width / 256)),
                int(round(box[3] * original.height / 256)),
            ]
            filename = f"region_{len(regions) + 1}.png"
            original.crop(original_box).save(directory / filename)
            regions.append(
                dict(
                    id=f"R{len(regions) + 1}",
                    box=original_box,
                    box_grid=box,
                    pixels=area,
                    maximum_score=float(anomaly_map[labels == index].max()),
                    crop_path="evidence/" + filename,
                )
            )
            if len(regions) >= self.policy["regions"]["maximum_regions"]:
                break
        total_regions = sum(
            int(stats[index, 4]) >= self.policy["regions"]["minimum_pixels"]
            for index in range(1, count)
        )
        return dict(
            crop_paths=[region["crop_path"] for region in regions],
            regions=regions,
            evidence_summary=[
                f"{region['id']}: {region['pixels']} 个高分像素，框 {region['box']}"
                for region in regions
            ],
            pixel_threshold=self.profile["pixel_threshold"],
            anomaly_area_ratio=float(binary.mean()),
            total_regions=total_regions,
            truncated=total_regions > len(regions),
        )

    def finalize_inspection(self, request: FinalizeInput) -> dict:
        """Turn detection plus completeness flags into a candidate state.

        Any missing, ambiguous, conflicting or failed evidence becomes
        manual_review; the model score and the frozen threshold are never
        adjusted here.
        """
        reasons = list(request.errors)
        if not request.quality_ok:
            reasons.append("输入质量未通过，无法可靠判定")
        if request.ambiguous:
            reasons.append("分数位于正常验证集校准的不确定区间")
        if request.conflict:
            reasons.append("模型与复核证据不一致")
        if request.score is None:
            reasons.append("缺少有效模型分数")
        state = (
            "manual_review"
            if reasons
            else (
                "anomaly_candidate"
                if request.score > request.threshold
                else "normal_candidate"
            )
        )
        if not reasons:
            reasons.append("依据冻结图像阈值与完整性规则生成候选状态")
        return dict(
            state=state,
            evidence_paths=request.evidence_paths,
            rule_version=self.policy["rule_version"],
            reasons=reasons,
        )
