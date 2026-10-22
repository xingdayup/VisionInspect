from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .common import identity, project_path, read_json, sha256, write_json


def _image(path: Path, size: int, mask: bool = False):
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (size, size):
                raise ValueError(f"Expected {size}x{size}: {path}, got {image.size}")
            if mask:
                array = np.asarray(image)
                if array.ndim != 2 or not set(np.unique(array)).issubset({0, 1, 255}):
                    raise ValueError(f"Non-binary / non-grayscale mask: {path}")
                if not array.any():
                    raise ValueError(f"Empty anomaly mask: {path}")
            elif image.mode != "RGB":
                raise ValueError(f"Expected RGB: {path}, got {image.mode}")
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Cannot decode image: {path}") from error


def check_data(config: dict) -> dict:
    """Validate an already present dataset. Never imports an auto-downloading datamodule."""
    import torch

    settings = config["dataset"]
    root = project_path(settings["root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset missing: {root}. Automatic download is disabled."
        )
    defects = set(settings["defects"])
    expected_test_dirs = defects | {"good"}
    actual_dirs = (
        {p.name for p in (root / "test").iterdir() if p.is_dir()}
        if (root / "test").is_dir()
        else set()
    )
    if actual_dirs != expected_test_dirs:
        raise ValueError(
            f"Unexpected test categories: {actual_dirs}, expected {expected_test_dirs}"
        )
    records = []
    for partition, category in [("train", "good")] + [
        ("test", d) for d in sorted(expected_test_dirs)
    ]:
        directory = root / partition / category
        files = sorted(directory.glob("*.png"))
        if not files or any(
            p.is_file() and p.suffix.lower() != ".png" for p in directory.iterdir()
        ):
            raise ValueError(f"Empty directory or unexpected file: {directory}")
        for path in files:
            _image(path, settings["image_size"])
            record = dict(
                path=path.relative_to(root).as_posix(),
                sha256=sha256(path),
                label=int(category != "good"),
                defect=category,
                source_split=partition,
            )
            if record["label"]:
                mask = root / "ground_truth" / category / (path.stem + "_mask.png")
                if not mask.is_file():
                    raise FileNotFoundError(f"Mask missing: {mask}")
                _image(mask, settings["image_size"], mask=True)
                record.update(
                    mask=mask.relative_to(root).as_posix(), mask_sha256=sha256(mask)
                )
            records.append(record)
    train = [r for r in records if r["source_split"] == "train"]
    test = [r for r in records if r["source_split"] == "test"]
    counts = (
        len(train),
        sum(r["label"] == 0 for r in test),
        sum(r["label"] == 1 for r in test),
    )
    expected = (
        settings["expected_train"],
        settings["expected_test_good"],
        settings["expected_test_anomaly"],
    )
    if counts != expected:
        raise ValueError(f"Dataset counts {counts} do not match expected {expected}")
    mask_paths = {r["mask"] for r in test if r["label"]}
    actual_masks = {
        p.relative_to(root).as_posix() for p in (root / "ground_truth").rglob("*.png")
    }
    if actual_masks != mask_paths:
        raise ValueError("Orphan or missing ground-truth masks")
    if len({r["sha256"] for r in records}) != len(records):
        raise ValueError("Duplicate image content; cannot guarantee independent splits")
    # Identical to anomalib.data.utils.random_split: floor, then distribute remainder.
    n_val = int(len(train) * settings["validation_ratio"])
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(config["runtime"]["seed"])
    ).tolist()
    n_train = len(train) - n_val
    for index in order[:n_train]:
        train[index]["split"] = "train"
    for index in order[n_train:]:
        train[index]["split"] = "validation"
    for record in test:
        record["split"] = "test"
    manifest = dict(
        schema_version=1,
        dataset="mvtec_ad",
        category=settings["category"],
        root=str(root),
        seed=config["runtime"]["seed"],
        validation_ratio=settings["validation_ratio"],
        source="Official MVTec AD; locally supplied files validated, archive provenance recorded separately",
        records=records,
    )
    manifest["digest"] = identity(manifest)
    existing = project_path(settings["manifest"])
    if os.environ.get("VISIONINSPECT_RELOCATED_ASSETS") == "1" and existing.is_file():
        previous = read_json(existing)
        if previous.get("digest") != identity(
            {k: v for k, v in previous.items() if k != "digest"}
        ):
            raise ValueError("Original manifest digest mismatch")
        if {k: v for k, v in previous.items() if k not in {"root", "digest"}} == {
            k: v for k, v in manifest.items() if k not in {"root", "digest"}
        }:
            # Preserve the source manifest identity used by the checkpoint and calibration.
            return previous
    write_json(project_path(settings["manifest"]), manifest)
    return manifest


def load_manifest(config: dict, splits: tuple[str, ...]) -> tuple[dict, list[dict]]:
    manifest = read_json(project_path(config["dataset"]["manifest"]))
    if manifest["digest"] != identity(
        {k: v for k, v in manifest.items() if k != "digest"}
    ):
        raise ValueError("Manifest digest mismatch")
    if (
        manifest["root"] != str(project_path(config["dataset"]["root"]))
        and os.environ.get("VISIONINSPECT_RELOCATED_ASSETS") != "1"
    ):
        raise ValueError("Manifest root does not match config")
    if (
        manifest["seed"] != config["runtime"]["seed"]
        or manifest["validation_ratio"] != config["dataset"]["validation_ratio"]
    ):
        raise ValueError("Manifest split parameters do not match config")
    records = [r for r in manifest["records"] if r["split"] in splits]
    if not records:
        raise ValueError(f"Empty split: {splits}")
    root = project_path(config["dataset"]["root"])
    # Only inspect selected split contents: fit/calibrate never open test images or masks.
    for record in records:
        for field, hash_field in [("path", "sha256"), ("mask", "mask_sha256")]:
            if field in record:
                path = (root / record[field]).resolve()
                if (
                    not path.is_relative_to(root.resolve())
                    or sha256(path) != record[hash_field]
                ):
                    raise ValueError(f"File changed or unsafe path: {record[field]}")
    return manifest, records
