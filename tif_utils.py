from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch


def save_tif(mask: np.ndarray, meta: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    meta.update(count=1, dtype="uint8", compress="lzw", nodata=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env():
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mask, 1)


def load_tif(
    path: str | Path, bands: list[int] | None = None, as_tensor: bool = True
) -> tuple[torch.Tensor | np.ndarray, dict[str, Any]]:
    """Loads a TIF file and returns it as a C x H x W tensor or array."""
    with rasterio.Env():
        with rasterio.open(path) as src:
            if bands is not None:
                for b in bands:
                    if b < 1 or b > src.count:
                        raise ValueError(f"Invalid band {b} for {path}")

                img = src.read(bands).astype(np.float32)
            else:
                img = src.read().astype(np.float32)

                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
            
            meta = src.meta

    if as_tensor:
        img = torch.from_numpy(img)

    return img, meta


def normalize_channels(
    img: torch.Tensor | np.ndarray, channels: list[int] | None = None
) -> torch.Tensor | np.ndarray:
    """
    Normalizes specified channels of an image to the [0, 1] range independently.
    If channels is None, normalizes all channels.
    """
    is_tensor = isinstance(img, torch.Tensor)
    out = img.clone() if is_tensor else img.copy()

    c_dim = out.shape[0]
    channels_to_norm = channels if channels is not None else list(range(c_dim))

    for c in channels_to_norm:
        if c < 0 or c >= c_dim:
            raise ValueError(f"Channel index {c} out of bounds for {c_dim} channels.")

        channel_data = out[c]
        c_min = channel_data.min()
        c_max = channel_data.max()

        if c_max > c_min:
            out[c] = (channel_data - c_min) / (c_max - c_min)
        else:
            out[c] = channel_data - c_min

    return out


def pair_data_to_mask_tif(
    data_root: str | Path, mask_root: str | Path, split: str
) -> list[tuple[str, str]]:
    img_dir = Path(data_root) / split
    mask_dir = Path(mask_root) / split

    mask_map = {m.name.replace("-label.tif", ""): m for m in mask_dir.rglob("*-label.tif")}
    pairs = []

    for img_path in img_dir.rglob("*.tif"):
        key = img_path.stem
        if key in mask_map:
            pairs.append((str(img_path), str(mask_map[key])))

    return pairs
