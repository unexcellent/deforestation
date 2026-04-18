import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import SegDataset, batch_collate_fn
from model import FocalLoss, SegmentationModel
from tif_utils import pair_data_to_mask_tif


def train_one_epoch(
    model: torch.nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: torch.nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentinel-2 Segmentation Model")
    parser.add_argument("--img-root", type=str, default="data/makeathon-challenge/sentinel-2")
    parser.add_argument("--mask-root", type=str, default="data/preprocessed/labels")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bands", nargs="+", type=int, default=[4, 3, 2])
    
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.mps.is_available() 
        else "cpu"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    img_label_pairs = pair_data_to_mask_tif(
        Path(args.img_root), Path(args.mask_root), "train"
    )
    
    dataset = SegDataset(
        pairs=img_label_pairs, 
        target_size=(args.img_size, args.img_size),
        bands=args.bands
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        collate_fn=batch_collate_fn,
        shuffle=True
    )

    model = SegmentationModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    run_dir = checkpoint_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()

    best = float("inf")

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss:.4f}")

        torch.save(model.state_dict(), run_dir / "last.pth")

        if loss < best:
            best = loss
            torch.save(model.state_dict(), run_dir / "best.pth")
            print("Saved best model")


if __name__ == "__main__":
    main()