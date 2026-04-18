import argparse
from pathlib import Path

import torch
from evaluate_model import find_latest_best_model
from model_utils import load_model, predict
from tif_utils import load_tif, normalize_channels, save_tif


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def run(model, input_path, output_path, bands=None, device=DEVICE):
    x, meta = load_tif(input_path, bands=bands)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x  = normalize_channels(x)
    mask = predict(model, x, device=device)
    save_tif(mask, meta, output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (defaults to latest best.pth)"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .tif file or folder"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/model_output_tif",
        help="Output folder for predicted GeoTIFFs"
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=None,
        help="Band indices (e.g. 4 3 2)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE
    )

    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_best_model()
    )

    model = load_model(checkpoint, device=args.device)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        out_path = output_dir / f"{input_path.stem}_pred.tif"
        run(model, input_path, out_path, args.bands, args.device)

    else:
        tifs = list(input_path.rglob("*.tif"))
        for p in tifs:
            out_path = output_dir / f"{p.stem}_pred.tif"
            run(model, p, out_path, args.bands, args.device)

    print("Done.")


if __name__ == "__main__":
    main()