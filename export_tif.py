import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model_utils import load_model, predict
from tif_utils import load_tif, normalize_channels, save_tif


def get_device() -> str:
    """Detects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(
    model: torch.nn.Module, 
    input_path: Path, 
    output_path: Path, 
    bands: list[int] | None = None
) -> None:
    """
    Loads a single TIFF, runs inference, and saves the georeferenced mask.
    """
    # 1. Load and prepare input
    x, meta = load_tif(str(input_path), bands=bands)
    
    # Model expects 4D: [Batch, Channel, H, W]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    x = normalize_channels(x)
    
    # 2. Inference (predict handles padding internally)
    mask_tensor = predict(model, x)
    
    # 3. Format for Rasterio (Squeeze to 2D and move to CPU NumPy)
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    mask_np = np.squeeze(mask_np) 

    # 4. Update metadata for a single-band classification map
    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 0
    })

    save_tif(mask_np, meta, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Export Predicted Deforestation Masks")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained .pth checkpoint"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .tif file or directory containing .tif files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/model_output_tif",
        help="Directory where predicted GeoTIFFs will be saved"
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=[4, 3, 2],
        help="Band indices used during training (e.g., 4 3 2 for RGB)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="Force device (cuda, mps, or cpu)"
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once for all files
    model = load_model(
        args.checkpoint, 
        device=args.device, 
        in_channels=len(args.bands)
    )
    print(f"Loaded model onto {args.device}. Starting batch processing...")

    # Discover files
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        # Recursively find all TIFs, but ignore files we already predicted
        files_to_process = [
            p for p in input_path.rglob("*.tif") 
            if not p.stem.endswith("_pred")
        ]

    if not files_to_process:
        print(f"No valid .tif files found at {input_path}")
        return

    # Process loop with progress bar
    start_time = time.time()
    for p in tqdm(files_to_process, desc="Exporting TIFs"):
        try:
            # We mirror the stem but add a suffix to avoid overwriting inputs
            out_path = output_dir / f"{p.stem}_pred.tif"
            run(model, p, out_path, bands=args.bands)
        except Exception as e:
            print(f"\nError processing {p.name}: {e}")
            continue

    duration = time.time() - start_time
    print(f"\nSuccessfully processed {len(files_to_process)} files in {duration:.2f}s.")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()