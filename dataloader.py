import random
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tif_utils import load_tif

BLACKLIST: list[str] = [
    "48PXC_7_7__s2_l2a_2024_9.tif",
    "48PYB_3_6__s2_l2a_2020_12.tif",
    "47QMB_0_8__s2_l2a_2024_11.tif",
    "48PXC_7_7__s2_l2a_2025_12.tif",
    "48PWV_7_8__s2_l2a_2024_9.tif",
    "47QMB_0_8__s2_l2a_2024_6.tif",
]


def _is_blacklisted(path: Path) -> bool:
    if not BLACKLIST:
        return False
    path_str = str(path)
    return any(b in path_str for b in BLACKLIST)


def image_quality_check(
    img: torch.Tensor | None,
    min_size: int = 256,
    max_size: int = 2000,
    saturation_threshold: float = 0.95,
) -> bool:
    """Checks for sensor artifacts, clipping, and sufficient variance."""
    if img is None or img.ndim != 3:
        return False

    _, h, w = img.shape
    if not (min_size <= h <= max_size and min_size <= w <= max_size):
        return False

    img_float = img.float()
    if img_float.abs().mean() < 1e-4 or img_float.std() < 1e-3:
        return False

    max_val = img_float.max()
    if max_val > 0:
        saturation_ratio = (img_float == max_val).float().mean()
        if saturation_ratio > saturation_threshold:
            return False

    return True


def mask_quality_check(
    mask: torch.Tensor, min_foreground_ratio: float = 0.0001
) -> bool:
    """Filters empty masks or those with negligible foreground signal."""
    total_pixels = mask.numel()
    if total_pixels == 0:
        return False

    foreground_pixels = torch.count_nonzero(mask)
    ratio = foreground_pixels.float() / total_pixels
    return ratio > min_foreground_ratio


def _extract_int(val: Any) -> int:
    """Recursively extracts an integer from nested sequences."""
    if isinstance(val, (list, tuple)):
        return _extract_int(val[0])
    return int(val)


def _parse_target_size(target_size: Any) -> tuple[int, int]:
    """Ensures target_size is a flat tuple of integers."""
    return _extract_int(target_size[0]), _extract_int(target_size[1])


def resize_img(img: torch.Tensor, target_size: Any) -> torch.Tensor:
    """Resizes C,H,W tensor to target_size using bilinear interpolation."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    int_size = _parse_target_size(target_size)
    img = F.interpolate(
        img.float(), size=int_size, mode="bilinear", align_corners=False
    )
    return img.squeeze(0)


def resize_mask(mask: torch.Tensor, target_size: Any) -> torch.Tensor:
    """Resizes H,W or 1,H,W mask using nearest neighbor."""
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    int_size = _parse_target_size(target_size)
    mask = F.interpolate(mask, size=int_size, mode="nearest")
    return mask.squeeze(0).squeeze(0).long()


def normalize_batch(
    batch: torch.Tensor, channels: list[int] | None = None
) -> torch.Tensor:
    """Min-Max normalization per channel across the batch."""
    out = batch.clone()
    _, c_dim, _, _ = out.shape
    channels_to_norm = channels if channels is not None else list(range(c_dim))

    for c in channels_to_norm:
        channel_data = out[:, c, ...]
        c_min, c_max = channel_data.min(), channel_data.max()

        if c_max > c_min:
            out[:, c, ...] = (channel_data - c_min) / (c_max - c_min)
        else:
            out[:, c, ...] = 0.0

    return out


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Internal dataset class.
    Users should prefer SegDataLoader.create_split_loaders.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        target_size: Any,
        bands: list[int] | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.bands = bands

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, _ = load_tif(str(self.image_paths[idx]), bands=self.bands)
        mask, _ = load_tif(str(self.mask_paths[idx]))

        if data is None or mask is None:
            raise RuntimeError(
                f"Failed to load TIFF at index {idx}: {self.image_paths[idx]}"
            )

        return resize_img(data, self.target_size), resize_mask(mask, self.target_size)


def batch_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacks samples and applies normalization."""
    data_stack = normalize_batch(torch.stack([item[0] for item in batch]))
    mask_stack = torch.stack([item[1] for item in batch])
    return data_stack, mask_stack


class SegDataLoader:
    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        target_size: Any,
        bands: list[int] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        worker_init_fn: Any = None,
    ) -> None:
        self.dataset = SegDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            target_size=target_size,
            bands=bands,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=batch_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    @classmethod
    def create_split_loaders(
        cls,
        img_root: str | Path,
        mask_root: str | Path,
        target_size: Any,
        split: str | None = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        apply_filter: bool = True,
        bands: list[int] | None = None,
        **loader_kwargs,
    ) -> tuple["SegDataLoader", "SegDataLoader"]:
        """
        Handles discovery, quality filtering, and splitting in one step.
        """
        img_dir = Path(img_root) / split if split else Path(img_root)
        mask_dir = Path(mask_root) / split if split else Path(mask_root)

        # 1. Discover potential pairs
        mask_map = {
            m.name.replace("-label.tif", ""): m
            for m in mask_dir.rglob("*-label.tif")
            if not _is_blacklisted(m)
        }

        potential_pairs = []
        for img_path in img_dir.rglob("*.tif"):
            if img_path.name.endswith("-label.tif"):
                continue
            if _is_blacklisted(img_path):
                continue
            if img_path.stem in mask_map:
                potential_pairs.append((img_path, mask_map[img_path.stem]))

        # 2. Filter during "Preprocessing"
        valid_pairs = []
        if apply_filter:
            for img_p, mask_p in potential_pairs:
                img, _ = load_tif(str(img_p), bands=bands)
                if not image_quality_check(img):
                    continue
                mask, _ = load_tif(str(mask_p))
                if mask_quality_check(mask):
                    valid_pairs.append((img_p, mask_p))
        else:
            valid_pairs = potential_pairs

        # 3. Split
        random.seed(seed)
        random.shuffle(valid_pairs)

        split_idx = int(len(valid_pairs) * train_ratio)
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]

        # 4. Instantiate loaders
        train_loader = cls(
            image_paths=[p[0] for p in train_pairs],
            mask_paths=[p[1] for p in train_pairs],
            target_size=target_size,
            bands=bands,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = cls(
            image_paths=[p[0] for p in val_pairs],
            mask_paths=[p[1] for p in val_pairs],
            target_size=target_size,
            bands=bands,
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)
