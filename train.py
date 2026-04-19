import argparse
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from dataloader import SegDataLoader
from model import FocalLoss, ResNet18UNet


def worker_init(worker_id: int) -> None:
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def compute_iou(preds: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> float:
    if preds.shape[1] > 1:
        preds = torch.argmax(preds, dim=1)
    else:
        preds = (torch.sigmoid(preds) > 0.5).squeeze(1)

    intersection = ((preds == 1) & (masks == 1)).float().sum(dim=(1, 2))
    union = ((preds == 1) | (masks == 1)).float().sum(dim=(1, 2))
    return ((intersection + eps) / (union + eps)).mean().item()


def train_one_epoch(
    model: torch.nn.Module,
    loader: SegDataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: FocalLoss,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss, total_iou = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for imgs, masks, valid_masks in pbar:
        imgs, masks, valid_masks = (
            imgs.to(device),
            masks.to(device),
            valid_masks.to(device),
        )

        preds = model(imgs)
        loss = loss_fn(preds, masks, mask=valid_masks)

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
    loss_fn: FocalLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for imgs, masks, valid_masks in tqdm(loader, desc="Evaluating", leave=False):
        imgs, masks, valid_masks = (
            imgs.to(device),
            masks.to(device),
            valid_masks.to(device),
        )
        preds = model(imgs)
        loss = loss_fn(preds, masks, mask=valid_masks)
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)

    return total_loss / len(loader), total_iou / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-root", type=str, default="data/makeathon-challenge/sentinel-2"
    )
    parser.add_argument("--mask-root", type=str, default="data/preprocessed/labels")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume training")
    parser.add_argument("--img-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bands", nargs="+", type=int, default=[4, 3, 2])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = SegDataLoader.create_split_loaders(
        img_root=args.img_root,
        mask_root=args.mask_root,
        target_size=args.img_size,
        bands=args.bands,
        batch_size=args.batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    model = ResNet18UNet(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    start_epoch = 1
    best_iou = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Check if checkpoint is a full state dict or just model weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_iou = checkpoint.get("best_iou", 0.0)
            else:
                # Fallback for simple weight loading
                model.load_state_dict(checkpoint)
            
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at: {args.resume}")

    run_dir = checkpoint_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        val_loss, val_iou = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_iou": best_iou,
        }

        torch.save(state, run_dir / "last.pth")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(state, run_dir / "best.pth")
            print(f"New best IoU: {best_iou:.4f} (Saved best.pth)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
