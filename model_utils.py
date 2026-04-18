from pathlib import Path
import numpy as np
import torch
from model import SegmentationModel


def load_model(checkpoint_path, in_channels, device="cuda"):
    model = SegmentationModel(num_classes=2, in_channels=in_channels)
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

import torch
import torch.nn.functional as F

@torch.inference_mode()
def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Predicts a mask while handling:
    1. Device mismatch (moves input to model device)
    2. Dimension mismatch (pads to multiples of 32 for U-Net)
    3. Output format (Argmax for multiclass, Sigmoid for binary)
    """
    model.eval().cpu()
    x = x.cpu()
    
    # Handle U-Net dimension constraints (must be multiple of 32)
    _, _, h, w = x.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        # Using reflect padding to avoid edge artifacts in satellite imagery
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    
    logits = model(x)
    
    # Crop back to original dimensions if we padded
    if pad_h > 0 or pad_w > 0:
        logits = logits[:, :, :h, :w]
        
    # Determine output based on channel count
    # Shape [B, C, H, W]
    if logits.shape[1] > 1:
        # Multiclass: return class indices
        return torch.argmax(logits, dim=1)
    else:
        # Binary: return 0 or 1
        return (torch.sigmoid(logits) > 0.5).long()