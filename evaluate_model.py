import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader import resize_img, resize_mask
from model import SegmentationModel
from model_utils import find_latest_best_model
from tif_utils import load_tif, normalize_channels, pair_data_to_mask_tif


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)

IMG_ROOT = Path("data/makeathon-challenge/sentinel-2")
MASK_ROOT = Path("data/preprocessed/labels")

IMG_SIZE = (256, 256)
NUM_SAMPLES_TO_SHOW = 10

RGB_BANDS = [4, 3, 2]


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    pred_fg = pred > 0
    tgt_fg = target > 0
    inter = (pred_fg & tgt_fg).sum().item()
    union = (pred_fg | tgt_fg).sum().item()
    return None if union == 0 else inter / union


def compute_acc(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    fg = target > 0
    if fg.sum().item() == 0:
        return None
    return ((pred == target) & fg).sum().item() / fg.sum().item()


def show(rgb: torch.Tensor, gt: np.ndarray, pred: np.ndarray) -> None:
    rgb_np = rgb.detach().cpu().numpy().transpose(1, 2, 0)
    rgb_np = np.clip(rgb_np, 0, 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb_np)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Pred")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(rgb_np)
    plt.imshow(pred, cmap="Reds", alpha=0.4)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def run(
    checkpoint_path: Path,
    img_root: Path,
    mask_root: Path,
    split: str,
    bands: list[int],
    num_samples: int,
) -> None:
    pairs = pair_data_to_mask_tif(img_root, mask_root, split)

    print(f"Found {len(pairs)} samples in {split}")
    if not pairs:
        return

    model = SegmentationModel(in_channels=len(bands)).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    samples = random.sample(pairs, min(num_samples, len(pairs)))

    for img_path_str, mask_path_str in samples:
        img_path = Path(img_path_str)

        img, _ = load_tif(img_path, bands=[4, 3, 2])
        data, _ = load_tif(img_path, bands=bands)
        gt, _ = load_tif(mask_path_str)

        data = resize_img(data, IMG_SIZE)
        gt = resize_mask(gt, IMG_SIZE)

        data = normalize_channels(data)

        norm_img = data.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(norm_img)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu()

        iou = compute_iou(pred, gt)
        acc = compute_acc(pred, gt)

        name = img_path.name

        if iou is None or acc is None:
            print(f"{name} | No foreground")
        else:
            print(f"{name} | IoU: {iou:.3f} | Acc: {acc:.3f}")

        img = resize_img(img, IMG_SIZE)
        img = normalize_channels(img)
        show(img, gt.numpy(), pred.numpy())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--img-root", type=str, default=str(IMG_ROOT))
    parser.add_argument("--mask-root", type=str, default=str(MASK_ROOT))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--bands", nargs="+", type=int, default=RGB_BANDS)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES_TO_SHOW)

    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_best_model()
    )

    run(
        checkpoint_path=checkpoint,
        img_root=Path(args.img_root),
        mask_root=Path(args.mask_root),
        split=args.split,
        bands=args.bands,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()