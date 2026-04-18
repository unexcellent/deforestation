import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from dataloader import SegDataLoader
from model import SegmentationModel
from model_utils import find_latest_best_model
from tif_utils import pair_temporal_samples


def compute_iou(preds: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> float:
    """Computes the Intersection over Union for the foreground class."""
    if preds.shape[1] > 1:
        preds = torch.argmax(preds, dim=1)
    else:
        preds = (torch.sigmoid(preds) > 0.5).squeeze(1)
        
    intersection = ((preds == 1) & (masks == 1)).float().sum(dim=(1, 2))
    union = ((preds == 1) | (masks == 1)).float().sum(dim=(1, 2))
    
    return ((intersection + eps) / (union + eps)).mean().item()


def validate(
    checkpoint_path: Path,
    img_root: Path,
    mask_root: Path,
    img_size: tuple[int, int],
    bands: list[int],
    batch_size: int,
    base_c: int,
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.mps.is_available() 
        else "cpu"
    )

    # Maintain split consistency with train.py
    pairs = pair_temporal_samples(img_root, mask_root, "train")
    random.seed(42) 
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    val_pairs = pairs[split_idx:]

    print(f"Validating on {len(val_pairs)} temporal pairs...")

    loader = SegDataLoader(
        val_pairs, 
        img_size, 
        batch_size, 
        bands=bands, 
        shuffle=False
    )

    # Pass base_c to match the trained checkpoint architecture
    model = SegmentationModel(
        in_channels=len(bands) * 2, 
        num_classes=2, 
        base_c=base_c
    ).to(device)
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    total_iou = 0.0
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_iou += compute_iou(preds, masks)

    avg_iou = total_iou / len(loader)
    print(f"\nResults for {checkpoint_path.name}:")
    print(f"Validation IoU: {avg_iou:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model IoU on the validation split.")
    parser.add_argument("--checkpoint", type=str, help="Path to best.pth")
    parser.add_argument("--img-root", type=str, default="data/makeathon-challenge/sentinel-2")
    parser.add_argument("--mask-root", type=str, default="data/preprocessed/labels")
    parser.add_argument("--img-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bands", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--base-c", type=int, default=32)

    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_best_model()
    )

    validate(
        checkpoint_path=checkpoint,
        img_root=Path(args.img_root),
        mask_root=Path(args.mask_root),
        img_size=tuple(args.img_size),
        bands=args.bands,
        batch_size=args.batch_size,
        base_c=args.base_c,
    )


if __name__ == "__main__":
    main()
