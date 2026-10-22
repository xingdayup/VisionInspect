import copy
import socket

import numpy as np
import pytest
from PIL import Image

from visioninspect.common import offline
from visioninspect.data import check_data, load_manifest
from visioninspect.metrics import calibrate, evaluate


def test_split_is_reproducible_disjoint_and_normal(tiny_dataset):
    config, _ = tiny_dataset
    with offline():
        first = check_data(config)
        second = check_data(config)
    assert first == second
    records = first["records"]
    assert {
        s: sum(r["split"] == s for r in records)
        for s in ["train", "validation", "test"]
    } == {"train": 16, "validation": 4, "test": 8}
    assert all(
        r["label"] == 0 for r in records if r["split"] in {"train", "validation"}
    )
    paths = [r["path"] for r in records]
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize(
    "damage",
    [
        "missing_root",
        "missing_dir",
        "corrupt_image",
        "missing_mask",
        "wrong_size",
        "nonbinary_mask",
        "orphan_mask",
        "wrong_count",
        "duplicate",
    ],
)
def test_invalid_data_fails_without_network(tiny_dataset, damage):
    config, root = tiny_dataset
    image = root / "train/good/000.png"
    mask = root / "ground_truth/broken_large/000_mask.png"
    if damage == "missing_root":
        config["dataset"]["root"] = str(root / "missing")
    elif damage == "missing_dir":
        config["dataset"]["defects"].append("not_present")
    elif damage == "corrupt_image":
        image.write_bytes(b"not an image")
    elif damage == "missing_mask":
        mask.unlink()
    elif damage == "wrong_size":
        Image.new("L", (4, 4), 255).save(mask)
    elif damage == "nonbinary_mask":
        Image.new("L", (32, 32), 42).save(mask)
    elif damage == "orphan_mask":
        Image.new("L", (32, 32), 255).save(mask.with_name("orphan_mask.png"))
    elif damage == "wrong_count":
        image.unlink()
    elif damage == "duplicate":
        (root / "test/good/000.png").write_bytes(image.read_bytes())
    with offline(), pytest.raises((ValueError, FileNotFoundError)):
        check_data(config)


def test_changed_train_detected_but_test_not_opened_during_fit(tiny_dataset):
    config, root = tiny_dataset
    manifest = check_data(config)
    test_record = next(r for r in manifest["records"] if r["split"] == "test")
    (root / test_record["path"]).write_bytes(b"damaged test image")
    with offline():
        _, train = load_manifest(config, ("train",))
        assert len(train) == 16
        with pytest.raises(ValueError):
            load_manifest(config, ("test",))
    (root / train[0]["path"]).write_bytes(b"damaged train image")
    with pytest.raises(ValueError):
        load_manifest(config, ("train",))


def test_calibration_rejects_test_and_anomaly_records():
    good = [dict(split="validation", label=0, score=s) for s in [1.0, 2.0, 3.0, 4.0]]
    assert calibrate(good) == 4.0
    assert calibrate(good) == calibrate(copy.deepcopy(good))
    for split, label in [("test", 0), ("validation", 1), ("train", 0)]:
        with pytest.raises(ValueError):
            calibrate([dict(split=split, label=label, score=3.0)])
    with pytest.raises(ValueError):
        calibrate([dict(split="validation", label=0, score=float("nan"))])


def test_test_scores_cannot_affect_calibration():
    validation = [
        dict(split="validation", label=0, score=s) for s in [0.1, 0.2, 0.3, 0.4]
    ]
    expected = calibrate(validation)
    for scores in [[1, 2, 3], [1e6, -1e6, 0]]:
        unrelated = [dict(split="test", label=1, score=s) for s in scores]
        with pytest.raises(ValueError):
            calibrate(validation + unrelated)
        assert calibrate(validation) == expected


def test_network_guard():
    with offline(), pytest.raises(RuntimeError, match="Network disabled"):
        socket.create_connection(("example.com", 443))


def test_metrics_perfect_detector():
    maps = np.zeros((2, 16, 16), dtype=np.float32)
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    maps[1, 4:8, 4:8] = 1.0
    masks[1, 4:8, 4:8] = 1
    result = evaluate([0, 1], [0.0, 1.0], maps, masks, 0.5)
    for name in [
        "image_auroc",
        "image_average_precision",
        "image_f1",
        "anomaly_recall",
        "pixel_auroc",
        "aupro",
    ]:
        assert result[name] == pytest.approx(1.0, abs=1e-6)
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]


def test_metrics_reject_nonfinite():
    maps = np.full((2, 4, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.0, 1.0], maps, np.zeros_like(maps), 0.5)
