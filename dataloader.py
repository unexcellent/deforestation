import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Iterator

from tif_utils import load_tif


def image_quality_check(
    img: torch.Tensor | None, min_size: int = 256, max_size: int = 2000
) -> bool:
    if img is None or img.ndim != 3:
        return False

    c, h, w = img.shape

    if h < min_size or w < min_size or h > max_size or w > max_size:
        return False

    if img.abs().mean() < 1e-4:
        return False

    if img.float().std() < 1e-3:
        return False

    if c >= 3:
        diff1 = (img[0] - img[1]).abs().mean()
        diff2 = (img[0] - img[2]).abs().mean()
        if diff1 < 1e-3 and diff2 < 1e-3:
            return False

    return True


def mask_quality_check(mask: torch.Tensor) -> bool:
    u = torch.unique(mask)
    return not (len(u) == 1 and u[0] == 0)


def filter_input_data(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    filtered_pairs = []

    for img_path, mask_path in pairs:
        img, _ = load_tif(img_path)
        mask, _ = load_tif(mask_path)

        if image_quality_check(img) and mask_quality_check(mask):
            filtered_pairs.append((img_path, mask_path))

    return filtered_pairs


def pad_to_square(x: torch.Tensor) -> torch.Tensor:
    _, h, w = x.shape
    size = max(h, w)
    return F.pad(x, (0, size - w, 0, size - h), value=0)


def resize_img(img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    img = pad_to_square(img).unsqueeze(0)
    img = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
    return img.squeeze(0)


def resize_mask(mask: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, size=target_size, mode="nearest")
    return mask.squeeze(0).squeeze(0).long()


def normalize_batch(
    batch: torch.Tensor, channels: list[int] | None = None
) -> torch.Tensor:
    """Normalizes specified channels across an entire [B, C, H, W] batch."""
    out = batch.clone()
    c_dim = out.shape[1]
    channels_to_norm = channels if channels is not None else list(range(c_dim))

    for c in channels_to_norm:
        if c < 0 or c >= c_dim:
            raise ValueError(f"Channel index {c} out of bounds for {c_dim} channels.")

        channel_data = out[:, c, ...]
        c_min = channel_data.min()
        c_max = channel_data.max()

        if c_max > c_min:
            out[:, c, ...] = (channel_data - c_min) / (c_max - c_min)
        else:
            out[:, c, ...] = channel_data - c_min

    return out


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, 
        pairs: list[tuple[str, str]], 
        target_size: tuple[int, int],
        bands: list[int] | None = None
    ) -> None:
        self.pairs = pairs
        self.target_size = target_size
        self.bands = bands

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_path, mask_path = self.pairs[idx]

        data, _ = load_tif(data_path, bands=self.bands)
        mask, _ = load_tif(mask_path)

        if data is None:
            raise RuntimeError(f"Failed to load valid data for index {idx}: {data_path}")

        return resize_img(data, self.target_size), resize_mask(mask, self.target_size)


def batch_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacks samples into batches and applies batch-level normalization."""
    data_stack = normalize_batch(torch.stack([item[0] for item in batch]))
    mask_stack = torch.stack([item[1] for item in batch])
    return data_stack, mask_stack


class SegDataLoader:
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        target_size: tuple[int, int],
        bands: list[int] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        apply_filter: bool = True,
    ) -> None:
        valid_pairs = filter_input_data(pairs) if apply_filter else pairs
        self.dataset = SegDataset(
            pairs=valid_pairs, 
            target_size=target_size, 
            bands=bands
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=batch_collate_fn,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)