import argparse
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from dataloader import SegDataLoader
from model import FocalLoss, SegmentationModel


def worker_init(worker_id: int) -> None:
    """
    Forces a clean, single-threaded GDAL/Rasterio environment per worker
    to prevent C-level segmentation faults during concurrent TIFF reads.
    """
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def compute_iou(preds: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> float:
    """Computes IoU for binary segmentation handling 1 or 2 output channels."""
    # Handle both 1-channel (BCE style) and 2-channel (CE style) outputs
    if preds.shape[1] > 1:
        # Multiclass: [B, 2, H, W] -> [B, H, W]
        preds = torch.argmax(preds, dim=1)
    else:
        # Binary: [B, 1, H, W] -> [B, H, W]
        preds = (torch.sigmoid(preds) > 0.5).squeeze(1)

    # Intersection and Union for the positive class (1)
    # We use logical operators to ensure we only count deforestation pixels
    intersection = ((preds == 1) & (masks == 1)).float().sum(dim=(1, 2))
    union = ((preds == 1) | (masks == 1)).float().sum(dim=(1, 2))
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def train_one_epoch(
    model: torch.nn.Module,
    loader: SegDataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric = compute_iou(preds, masks)
        total_loss += loss.item()
        total_iou += metric
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metric:.4f}")

    return total_loss / len(loader), total_iou / len(loader)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: SegDataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    for imgs, masks in tqdm(loader, desc="Evaluating", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        
        loss = loss_fn(preds, masks)
        metric = compute_iou(preds, masks)
        
        total_loss += loss.item()
        total_iou += metric

    return total_loss / len(loader), total_iou / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentinel-2 Segmentation Model")
    parser.add_argument("--img-root", type=str, default="data/makeathon-challenge/sentinel-2")
    parser.add_argument("--mask-root", type=str, default="data/preprocessed/labels")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--img-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bands", nargs="+", type=int, default=[4, 3, 2])
    parser.add_argument("--split-ratio", type=float, default=0.8)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"--- Training on Device: {device} ---")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = SegDataLoader.create_split_loaders(
        img_root=args.img_root,
        mask_root=args.mask_root,
        target_size=args.img_size,
        bands=args.bands,
        split="train", 
        train_ratio=args.split_ratio,
        batch_size=args.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init
    )

    print(f"Dataset Split: {len(train_loader.dataset)} train | {len(val_loader.dataset)} val")

    model = SegmentationModel(
        in_channels=len(args.bands), 
        num_classes=2, 
        base_c=16
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    run_dir = checkpoint_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        val_loss, val_iou = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}"
        )

        torch.save(model.state_dict(), run_dir / "last.pth")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), run_dir / "best.pth")
            print(f"New best IoU: {best_iou:.4f} (Saved best.pth)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()