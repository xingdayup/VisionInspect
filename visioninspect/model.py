from __future__ import annotations

import time
import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from .common import project_path, read_json, sha256


def setup(config: dict):
    torch.set_num_threads(config["runtime"]["threads"])
    torch.manual_seed(config["runtime"]["seed"])
    np.random.seed(config["runtime"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PatchCore; CPU fallback is disabled")
    return torch.device("cuda:0")


class Detector:
    """Drive Anomalib's PatchcoreModel directly: no Trainer, auto-download, or threshold fitting."""

    def __init__(self, config: dict, size: int | None = None):
        self.device = cuda_device()
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.config = config
        self.source_device = config["runtime"]["device"]
        settings = config["model"]
        self.size = size or settings["image_size"]
        self.model = (
            PatchcoreModel(
                layers=settings["layers"],
                backbone=settings["backbone"],
                pre_trained=False,
                num_neighbors=settings["num_neighbors"],
            )
            .to(self.device)
            .float()
        )

    def load_backbone(self):
        from safetensors.torch import load_file

        path = project_path(self.config["model"]["weights"])
        metadata = read_json(project_path(self.config["model"]["weights_metadata"]))
        actual_hash = sha256(path)
        if (
            metadata["sha256"] != actual_hash
            or metadata["model"] != "timm/resnet18.a1_in1k"
        ):
            raise ValueError("Backbone source metadata / checksum mismatch")
        state = load_file(str(path), device="cpu")
        extractor = self.model.feature_extractor.feature_extractor
        expected = extractor.state_dict()
        if not set(expected).issubset(state):
            raise ValueError(
                f"Backbone keys missing: {sorted(set(expected) - set(state))[:5]}"
            )
        extractor.load_state_dict({key: state[key] for key in expected}, strict=True)
        return actual_hash

    def tensor(self, array: np.ndarray):
        tensor = (
            torch.from_numpy(np.array(array, copy=True))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        tensor = resize(
            tensor,
            [self.size, self.size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return (
            normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            .unsqueeze(0)
            .to(self.device)
        )

    def fit(self, root: Path, records: list[dict], progress=None):
        if any(
            record["split"] != "train" or record["label"] != 0 for record in records
        ):
            raise ValueError("Only normal training records can build the memory bank")
        self.model.train()
        self.model.feature_extractor.eval()  # Never update ImageNet batch norm statistics.
        with torch.no_grad():
            for index, record in enumerate(records):
                with Image.open(root / record["path"]) as image:
                    tensor = self.tensor(np.asarray(image.convert("RGB")))
                self.model(tensor)
                if progress:
                    progress(f"features {index + 1}/{len(records)}")
            self.model.subsample_embedding(
                self.config["model"]["coreset_sampling_ratio"]
            )
        self.model.eval()

    def predict(self, array: np.ndarray):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            output = self.model(self.tensor(array))
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        score = float(output.pred_score.item())
        anomaly_map = output.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        if (
            not np.isfinite(score)
            or not np.isfinite(anomaly_map).all()
            or anomaly_map.shape != (self.size, self.size)
        ):
            raise ValueError("Non-finite score / invalid anomaly map")
        return score, anomaly_map, seconds

    def save(self, path: Path, metadata: dict):
        torch.save(
            dict(
                schema_version=1,
                size=self.size,
                config=self.config,
                state_dict=self.model.state_dict(),
                metadata=metadata,
            ),
            path,
        )

    @classmethod
    def load(cls, path: Path, config: dict | None = None):
        cuda_device()  # Reject unavailable CUDA before reading a potentially large checkpoint.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != 1:
            raise ValueError("Unsupported checkpoint schema")
        runtime_config = copy.deepcopy(config or checkpoint["config"])
        if runtime_config["model"] != checkpoint["config"]["model"]:
            raise ValueError(
                "GPU deployment cannot change frozen model or preprocessing settings"
            )
        runtime_config["runtime"]["device"] = "cuda"
        detector = cls(runtime_config, checkpoint["size"])
        detector.source_device = checkpoint["config"]["runtime"]["device"]
        detector.model.load_state_dict(checkpoint["state_dict"], strict=True)
        detector.model.eval()
        if detector.model.memory_bank.device.type != "cuda" or any(
            p.device.type != "cuda" for p in detector.model.parameters()
        ):
            raise RuntimeError(
                "PatchCore parameters and memory bank must all be on CUDA"
            )
        return detector, checkpoint["metadata"]
