from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def calibrate(records: list[dict], quantile: float = 0.95) -> float:
    if not records or any(
        r.get("split") != "validation" or r.get("label") != 0 for r in records
    ):
        raise ValueError("Calibration accepts normal validation predictions only")
    scores = np.array([r["score"] for r in records], dtype=np.float64)
    if not np.isfinite(scores).all() or not 0 < quantile < 1:
        raise ValueError("Invalid calibration scores or quantile")
    return float(np.quantile(scores, quantile, method="higher"))


def evaluate(labels, scores, maps, masks, threshold, fpr_limit=0.3):
    import torch
    from anomalib.metrics.aupro import _AUPRO

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    maps = np.asarray(maps, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    if (
        maps.shape != masks.shape
        or not np.isfinite(maps).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid evaluation arrays")
    if set(np.unique(labels)) != {0, 1} or set(np.unique(masks)) != {0, 1}:
        raise ValueError("Evaluation requires both normal and anomaly labels/pixels")
    predicted = scores > threshold
    metric = _AUPRO(fpr_limit=fpr_limit)
    metric.update(torch.from_numpy(maps), torch.from_numpy(masks))
    aupro = float(metric.compute().item())
    if not np.isfinite(aupro):
        raise ValueError("AUPRO is non-finite")
    return dict(
        image_auroc=float(roc_auc_score(labels, scores)),
        image_average_precision=float(average_precision_score(labels, scores)),
        image_f1=float(f1_score(labels, predicted, zero_division=0)),
        anomaly_recall=float(recall_score(labels, predicted, zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        confusion_matrix_order=["normal", "anomaly"],
        pixel_auroc=float(roc_auc_score(masks.ravel(), maps.ravel())),
        aupro=aupro,
        aupro_fpr_limit=fpr_limit,
        metric_grid=list(maps.shape[-2:]),
        pixel_masks="nearest-neighbor binary",
        pixels_include_normal_images=True,
    )
