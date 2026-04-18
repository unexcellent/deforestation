import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tif_utils import load_tif, reproject_to_match


def resize_img(img: torch.Tensor, target_size: Any) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    img = F.interpolate(img.float(), size=target_size, mode="bilinear", align_corners=False)
    return img.squeeze(0)


def resize_mask(mask: torch.Tensor, target_size: Any) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, size=target_size, mode="nearest")
    return mask.squeeze(0).squeeze(0).long()


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    out = batch.clone()
    for c in range(out.shape[1]):
        channel_data = out[:, c, ...]
        c_min, c_max = channel_data.min(), channel_data.max()
        if c_max > c_min:
            out[:, c, ...] = (channel_data - c_min) / (c_max - c_min)
        else:
            out[:, c, ...] = 0.0
    return out


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, 
        pairs: list[dict[str, str]], 
        target_size: tuple[int, int],
        bands: list[int] | None = None
    ) -> None:
        self.pairs = pairs
        self.target_size = target_size
        self.bands = bands

    def __len__(self) -> int:
        return len(self.pairs)

    def _ensure_alignment(self, data, meta, ref_meta, is_mask=False):
        shape_match = (data.shape[1] == ref_meta["height"] and 
                       data.shape[2] == ref_meta["width"])
        if (meta["crs"] != ref_meta["crs"] or 
            meta["transform"] != ref_meta["transform"] or 
            not shape_match):
            return reproject_to_match(data, meta, ref_meta, is_mask=is_mask)
        return data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.pairs[idx]
        img1, meta1 = load_tif(p["img_early"], bands=self.bands, as_tensor=False)
        img2, meta2 = load_tif(p["img_late"], bands=self.bands, as_tensor=False)
        m1, mm1 = load_tif(p["mask_early"], as_tensor=False)
        m2, mm2 = load_tif(p["mask_late"], as_tensor=False)

        img2 = self._ensure_alignment(img2, meta2, meta1)
        m1 = self._ensure_alignment(m1, mm1, meta1, is_mask=True)
        m2 = self._ensure_alignment(m2, mm2, meta1, is_mask=True)

        mask_diff = (m2 == 1) & (m1 == 0)
        
        combined_img = torch.cat([torch.from_numpy(img1), torch.from_numpy(img2)], dim=0)
        target_mask = torch.from_numpy(mask_diff.astype(np.uint8))

        return (
            resize_img(combined_img, self.target_size), 
            resize_mask(target_mask, self.target_size)
        )


class SegDataLoader:
    def __init__(
        self,
        pairs: list[dict[str, str]],
        target_size: Any,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        bands: list[int] | None = None,
        worker_init_fn: Any = None,
    ) -> None:
        self.dataset = SegDataset(pairs=pairs, target_size=target_size, bands=bands)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.batch_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    @staticmethod
    def batch_collate_fn(batch):
        data_stack = normalize_batch(torch.stack([item[0] for item in batch]))
        mask_stack = torch.stack([item[1] for item in batch])
        return data_stack, mask_stack

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
