import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
import torch


def save_tif(mask: np.ndarray, meta: dict[str, Any], out_path: str | Path) -> None:
    """Saves a single-band mask to a GeoTIFF file."""
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
    """Normalizes specified channels to the [0, 1] range independently."""
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


def reproject_to_match(
    src_data: np.ndarray,
    src_meta: dict[str, Any],
    dst_meta: dict[str, Any],
    is_mask: bool = False,
) -> np.ndarray:
    """Reprojects source data to match the destination metadata's CRS, transform, and shape."""
    dst_shape = (dst_meta["height"], dst_meta["width"])
    c_dim = src_data.shape[0] if src_data.ndim == 3 else 1
    destination = np.zeros((c_dim, *dst_shape), dtype=src_data.dtype)
    
    reproject(
        source=src_data,
        destination=destination,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=Resampling.nearest if is_mask else Resampling.bilinear,
    )
    return destination


def pair_temporal_samples(
    data_root: str | Path, mask_root: str | Path, split: str
) -> list[dict[str, str]]:
    """Finds pairs of images exactly one year apart in the same location."""
    img_dir = Path(data_root) / split
    mask_dir = Path(mask_root) / split
    mask_map = {m.name.replace("-label.tif", ""): m for m in mask_dir.rglob("*-label.tif")}
    
    pairs = []
    pattern = re.compile(r"(.+)__s2_l2a_(\d{4})_(\d+)")

    if not img_dir.exists():
        return []

    locations = [d for d in img_dir.iterdir() if d.is_dir()]
    for loc in locations:
        files = list(loc.glob("*.tif"))
        metadata = []
        for f in files:
            match = pattern.search(f.name)
            if match:
                metadata.append({
                    "path": f,
                    "stem": f.stem,
                    "year": int(match.group(2)),
                    "month": int(match.group(3))
                })
        
        for early in metadata:
            for late in metadata:
                if late["year"] == early["year"] + 1 and late["month"] == early["month"]:
                    if early["stem"] in mask_map and late["stem"] in mask_map:
                        pairs.append({
                            "img_early": str(early["path"]),
                            "img_late": str(late["path"]),
                            "mask_early": str(mask_map[early["stem"]]),
                            "mask_late": str(mask_map[late["stem"]])
                        })
                        
    return pairs
