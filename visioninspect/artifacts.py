from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def save_evidence(output: Path, key: str, array: np.ndarray, anomaly_map: np.ndarray):
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / (key + ".npy"), anomaly_map, allow_pickle=False)
    # Min/max normalization is only for display; metrics always use raw float maps.
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
