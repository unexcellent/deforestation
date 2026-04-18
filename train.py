import random
import argparse
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from dataloader import SegDataLoader
from model import FocalLoss, SegmentationModel
from tif_utils import pair_temporal_samples


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


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iou = compute_iou(preds, masks)
        total_loss += loss.item()
        total_iou += iou
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")
    return total_loss / len(loader), total_iou / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, default="data/makeathon-challenge/sentinel-2")
    parser.add_argument("--mask-root", type=str, default="data/preprocessed/labels")
    parser.add_argument("--img-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bands", nargs="+", type=int, default=[4, 3, 2])
    parser.add_argument("--base-c", type=int, default=32)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = pair_temporal_samples(args.img_root, args.mask_root, "train")
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    
    train_loader = SegDataLoader(pairs[:split_idx], args.img_size, args.batch_size, bands=args.bands, worker_init_fn=worker_init)
    val_loader = SegDataLoader(pairs[split_idx:], args.img_size, args.batch_size, bands=args.bands, shuffle=False)

    model = SegmentationModel(in_channels=len(args.bands) * 2, num_classes=2, base_c=args.base_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss()

    run_dir = Path("checkpoints") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        t_loss, t_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print(f"Epoch {epoch} | Loss: {t_loss:.4f} | IoU: {t_iou:.4f}")
        if t_iou > best_iou:
            best_iou = t_iou
            torch.save(model.state_dict(), run_dir / "best.pth")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
