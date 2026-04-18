from pathlib import Path
import numpy as np
import torch
from model import SegmentationModel


def load_model(checkpoint_path, device="cuda"):
    model = SegmentationModel(num_classes=2)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def find_latest_best_model(checkpoint_root=Path("checkpoints")):
    runs = sorted(checkpoint_root.iterdir(), reverse=True)

    for run in runs:
        model_path = run / "best.pth"
        if model_path.exists():
            print(f"Using model: {model_path}")
            return model_path

    raise FileNotFoundError("No best.pth found")


@torch.no_grad()
def predict(model, x, device="cuda"):
    x = x.to(device)
    logits = model(x)
    pred = torch.argmax(logits, dim=1)
    return pred.squeeze(0).cpu().numpy().astype(np.uint8)