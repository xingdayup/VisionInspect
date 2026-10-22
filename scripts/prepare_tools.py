"""Prepare reference index and policy using normal training/validation only."""

from pathlib import Path
import json
import sys
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visioninspect.common import (
    configure_environment,
    load_config,
    offline,
    project_path,
    read_json,
    sha256,
    write_json,
    identity,
)
from visioninspect.data import load_manifest
from visioninspect.model import Detector, setup
from visioninspect.tools import quality_metrics


def prepare(category="bottle"):
    registry = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )
    registration = registry["models"][category]["patchcore"]
    if (
        project_path(registration["profile"]).exists()
        or project_path(registration["references"]).exists()
    ):
        raise FileExistsError(
            "Frozen profile/index already exist; use new registered paths for a GPU experiment"
        )
    config = load_config(project_path(registration["config"]))
    setup(config)
    fit = project_path(registration["fit_run"])
    calibration = read_json(fit / "calibration.json")
    detector, metadata = Detector.load(fit / "model.ckpt")
    manifest, training = load_manifest(config, ("train",))
    _, validation = load_manifest(config, ("validation",))
    root = ROOT / config["dataset"]["root"]
    vectors = []
    quality = []
    maps = []
    scores = []
    import torch

    def embedding(array):
        with torch.inference_mode():
            features = detector.model.feature_extractor(detector.tensor(array))
            vector = (
                torch.cat(
                    [
                        features[layer].mean(dim=(2, 3))
                        for layer in config["model"]["layers"]
                    ],
                    dim=1,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    for record in training:
        with Image.open(root / record["path"]) as image:
            vectors.append(embedding(np.asarray(image.convert("RGB"))))
    similarities = []
    for record in validation:
        with Image.open(root / record["path"]) as image:
            image = image.convert("RGB")
            array = np.asarray(image)
            quality.append(quality_metrics(image))
        score, anomaly_map, _ = detector.predict(array)
        scores.append(score)
        maps.append(anomaly_map)
        similarities.append(float(np.max(np.asarray(vectors) @ embedding(array))))
    directory = project_path(registration["references"]).parent
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        project_path(registration["references"]),
        embeddings=np.asarray(vectors, dtype=np.float32),
        paths=np.asarray([r["path"] for r in training]),
        hashes=np.asarray([r["sha256"] for r in training]),
    )
    threshold = calibration["threshold"]
    median = float(np.median(scores))
    profile = dict(
        schema_version=1,
        category=category,
        data_root=str(root),
        checkpoint_sha256=sha256(fit / "model.ckpt"),
        manifest_digest=manifest["digest"],
        reference_index_sha256=sha256(project_path(registration["references"])),
        image_threshold=threshold,
        review_lower=float(np.quantile(scores, 0.9, method="higher")),
        review_upper=threshold + max(threshold - median, 1e-6),
        pixel_threshold=float(np.quantile(np.asarray(maps), 0.995, method="higher")),
        reference_similarity_floor=float(
            np.quantile(similarities, 0.05, method="lower")
        ),
        quality=dict(
            sharpness_floor=max(
                0.0, float(np.quantile([q["sharpness"] for q in quality], 0.01)) * 0.5
            ),
            brightness_low=max(0.0, float(min(q["brightness"] for q in quality)) - 15),
            brightness_high=min(
                255.0, float(max(q["brightness"] for q in quality)) + 15
            ),
        ),
        calibration_source="normal validation only; no test images or labels loaded",
        training_count=len(training),
        validation_count=len(validation),
    )
    profile["digest"] = identity(profile)
    write_json(project_path(registration["profile"]), profile)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    configure_environment()
    with offline():
        prepare(sys.argv[1] if len(sys.argv) > 1 else "bottle")
