"""Persist anomaly evidence as raw float maps plus display-only renders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_evidence(
    output: Path, key: str, array: np.ndarray, anomaly_map: np.ndarray
) -> dict[str, object]:
    """Write the raw map and two renders, returning filenames and display scale.

    The min/max stretch below is a visualization detail only; every metric and
    threshold decision uses the untouched float map saved as key + '.npy'.
    """
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / (key + ".npy"), anomaly_map, allow_pickle=False)
    low, high = float(anomaly_map.min()), float(anomaly_map.max())
    display = (anomaly_map - low) / max(high - low, 1e-12)
    import matplotlib

    colors = (matplotlib.colormaps["inferno"](display)[..., :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(colors)
    heatmap.save(output / (key + "_heatmap.png"))
    original = Image.fromarray(array).resize(heatmap.size, Image.Resampling.BILINEAR)
    Image.blend(original, heatmap, 0.45).save(output / (key + "_overlay.png"))
    return dict(
        raw_map=key + ".npy",
        heatmap=key + "_heatmap.png",
        overlay=key + "_overlay.png",
        display_min=low,
        display_max=high,
        display_normalization="per-image, visualization only",
    )
