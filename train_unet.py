import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rasterio
import torchvision.models as models


# -------------------------
# CONFIG
# -------------------------

IMG_ROOT = Path("data/preprocessed/sentinel-2/images")
MASK_ROOT = Path("data/preprocessed/labels")

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("checkpoints")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# -------------------------
# IMAGE FILTERING (NEW)
# -------------------------

def image_quality_check(img, min_size=256, max_size=2000):
    if img is None:
        return False

    if img.ndim != 3:
        return False

    c, h, w = img.shape

    # --- size filtering (remove extreme outliers)
    if h < min_size or w < min_size:
        return False
    if h > max_size or w > max_size:
        return False

    # --- black image detection
    if img.abs().mean() < 1e-4:
        return False

    # --- flat / gray image detection
    if img.float().std() < 1e-3:
        return False

    # --- channel collapse check (RGB identical)
    if c >= 3:
        diff1 = (img[0] - img[1]).abs().mean()
        diff2 = (img[0] - img[2]).abs().mean()
        if diff1 < 1e-3 and diff2 < 1e-3:
            return False

    return True


def mask_quality_check(mask):
    u = torch.unique(mask)
    return not (len(u) == 1 and u[0] == 0)


# -------------------------
# PAIR BUILDING (FIXED)
# -------------------------

def build_pairs(split="train"):
    img_dir = IMG_ROOT / split
    mask_dir = MASK_ROOT / split

    mask_index = {
        m.stem.replace("-label", ""): m
        for m in mask_dir.rglob("*-label.tif")
    }

    pairs = []
    removed = 0

    for img_path in img_dir.rglob("*.npy"):
        key = img_path.stem
        if key not in mask_index:
            continue

        img = load_image(img_path)
        mask = load_mask(mask_index[key])

        if not image_quality_check(img):
            removed += 1
            continue

        if not mask_quality_check(mask):
            removed += 1
            continue

        pairs.append((img_path, mask_index[key]))

    print(f"Filtered out {removed} invalid samples")
    print(f"Kept {len(pairs)} valid samples")

    return pairs


# -------------------------
# LOADERS
# -------------------------

def load_image(path):
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr) if arr.ndim == 3 else None


def load_mask(path):
    with rasterio.open(path) as src:
        return torch.from_numpy(src.read(1).astype(np.int64))


# -------------------------
# PREPROCESSING
# -------------------------

def pad_to_square(x):
    c, h, w = x.shape
    size = max(h, w)

    pad_h = size - h
    pad_w = size - w

    return F.pad(x, (0, pad_w, 0, pad_h), value=0)


def resize_img(img):
    img = pad_to_square(img)
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return img.squeeze(0)


def resize_mask(mask):
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, size=IMG_SIZE, mode="nearest")
    return mask.squeeze(0).squeeze(0).long()


def normalize(img):
    return (img - IMAGENET_MEAN) / IMAGENET_STD


# -------------------------
# DATASET
# -------------------------

class SegDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = load_image(img_path)
        mask = load_mask(mask_path)

        if img is None:
            idx = (idx + 1) % len(self.pairs)
            img_path, mask_path = self.pairs[idx]
            img = load_image(img_path)
            mask = load_mask(mask_path)

        img = img[:3]

        img = resize_img(img)
        mask = resize_mask(mask)

        img = normalize(img)

        return img, mask


# -------------------------
# FOCAL LOSS
# -------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.85):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device)
        )

        return (loss * alpha_t).mean()


# -------------------------
# MODEL
# -------------------------

class Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_classes, 4, stride=4)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# -------------------------
# TRAINING
# -------------------------

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# MAIN
# -------------------------

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    pairs = build_pairs("train")
    dataset = SegDataset(pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = Model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    run_dir = CHECKPOINT_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()

    best = float("inf")

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, loss_fn)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

        torch.save(model.state_dict(), run_dir / "last.pth")

        if loss < best:
            best = loss
            torch.save(model.state_dict(), run_dir / "best.pth")
            print("Saved best model")


if __name__ == "__main__":
    main()