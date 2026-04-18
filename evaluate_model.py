import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import rasterio
import torchvision.models as models


# -------------------------
# CONFIG
# -------------------------

IMG_ROOT = Path("data/preprocessed/sentinel-2/images")
MASK_ROOT = Path("data/preprocessed/labels")
CHECKPOINT_ROOT = Path("checkpoints")

IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
NUM_SAMPLES_TO_SHOW = 10

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# -------------------------
# MODEL (MATCH TRAINING EXACTLY)
# -------------------------

class PretrainedSegModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(256, 128, 2, stride=2),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 2, stride=2),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, num_classes, 4, stride=4)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# -------------------------
# CHECKPOINT
# -------------------------

def find_latest_best_model():
    runs = sorted(CHECKPOINT_ROOT.iterdir(), reverse=True)

    for run in runs:
        model_path = run / "best.pth"
        if model_path.exists():
            print(f"Using model: {model_path}")
            return model_path

    raise FileNotFoundError("No best.pth found")


# -------------------------
# PAIRS
# -------------------------

def build_pairs(split="test"):
    img_dir = IMG_ROOT / split
    mask_dir = MASK_ROOT / split

    masks = list(mask_dir.rglob("*-label.tif"))

    pairs = []

    for img_path in img_dir.rglob("*.npy"):
        key = img_path.stem

        mask_path = next((m for m in masks if m.name.startswith(key)), None)

        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


# -------------------------
# LOADERS
# -------------------------

def load_image(path):
    img = np.load(path).astype(np.float32)
    return torch.from_numpy(img)


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


def resize_image(img):
    img = pad_to_square(img)
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return img.squeeze(0)


def resize_mask(mask):
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, size=IMG_SIZE, mode="nearest")
    mask = mask.squeeze(0).squeeze(0).long()

    return (mask > 0).long()


def normalize(img):
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def denormalize(img):
    return img * IMAGENET_STD + IMAGENET_MEAN


# -------------------------
# METRICS (FIXED)
# -------------------------

def compute_foreground_iou(pred, target):
    pred_fg = pred > 0
    target_fg = target > 0

    intersection = (pred_fg & target_fg).sum().item()
    union = (pred_fg | target_fg).sum().item()

    if union == 0:
        return None

    return intersection / union


def compute_foreground_accuracy(pred, target):
    fg = target > 0

    if fg.sum().item() == 0:
        return None

    correct = (pred == target) & fg
    return correct.sum().item() / fg.sum().item()


# -------------------------
# VISUALIZATION
# -------------------------

def show(rgb, gt, pred):
    rgb = rgb.numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
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
    plt.imshow(rgb)
    plt.imshow(pred, cmap="Reds", alpha=0.4)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# MAIN
# -------------------------

def main():
    model_path = find_latest_best_model()

    pairs = build_pairs("train")
    print(f"Found {len(pairs)} test samples")

    if len(pairs) == 0:
        return

    model = PretrainedSegModel(NUM_CLASSES).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)

    model.eval()

    samples = random.sample(pairs, min(NUM_SAMPLES_TO_SHOW, len(pairs)))

    for img_path, mask_path in samples:

        img = load_image(img_path)[:3]
        mask = load_mask(mask_path)

        img = resize_image(img)
        mask = resize_mask(mask)

        img = normalize(img)

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(pred, dim=1).squeeze().cpu()

        iou = compute_foreground_iou(pred, mask)
        acc = compute_foreground_accuracy(pred, mask)

        if iou is None or acc is None:
            print(f"{img_path.name} | No foreground present (metrics skipped)")
        else:
            print(f"{img_path.name} | FG IoU: {iou:.3f} | FG Acc: {acc:.3f}")

        show(denormalize(img.cpu()), mask.numpy(), pred.numpy())


if __name__ == "__main__":
    main()