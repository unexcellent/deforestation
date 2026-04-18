import re
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn.functional as F

from dataloader import normalize_batch, resize_img
from model import SegmentationModel
from model_utils import find_latest_best_model
from tif_utils import load_tif, reproject_to_match, save_tif


def get_late_image_path(early_path: Path) -> Path:
    match = re.search(r"([A-Z0-9_]+)__s2_l2a_(\d{4})_(\d+)", early_path.name)
    if not match:
        raise ValueError(
            f"Could not parse tile metadata from filename: {early_path.name}"
        )

    year = int(match.group(2))
    month = int(match.group(3))

    late_name = early_path.name.replace(f"_{year}_{month}", f"_{year + 1}_{month}")
    late_path = early_path.parent / late_name

    if not late_path.exists():
        raise FileNotFoundError(f"Expected late image not found: {late_path}")

    return late_path


def align_data(
    data: np.ndarray, meta: dict[str, Any], ref_meta: dict[str, Any]
) -> np.ndarray:
    shape_match = (
        data.shape[1] == ref_meta["height"] and data.shape[2] == ref_meta["width"]
    )
    if (
        meta["crs"] != ref_meta["crs"]
        or meta["transform"] != ref_meta["transform"]
        or not shape_match
    ):
        return reproject_to_match(data, meta, ref_meta, is_mask=False)

    return data


def predict_temporal_pair(
    early_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    img_size: tuple[int, int],
    bands: list[int],
    base_c: int,
) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    late_path = get_late_image_path(early_path)

    img1, meta1 = load_tif(early_path, bands=bands, as_tensor=False)
    img2, meta2 = load_tif(late_path, bands=bands, as_tensor=False)

    img2 = align_data(img2, meta2, meta1)

    combined_img = torch.cat([torch.from_numpy(img1), torch.from_numpy(img2)], dim=0)

    resized_img = resize_img(combined_img, img_size)
    input_tensor = normalize_batch(resized_img.unsqueeze(0)).to(device)

    model = SegmentationModel(
        in_channels=len(bands) * 2,
        num_classes=2,
        base_c=base_c,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        if logits.shape[1] > 1:
            pred = torch.argmax(logits, dim=1)
        else:
            pred = (torch.sigmoid(logits) > 0.5).squeeze(1)

    pred_float = pred.float().unsqueeze(0)
    original_size = (meta1["height"], meta1["width"])
    pred_resized = F.interpolate(pred_float, size=original_size, mode="nearest")

    out_mask = pred_resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

    save_tif(out_mask, meta1, output_path)
    click.echo(f"Prediction saved to {output_path}")


@click.group()
def cli() -> None:
    pass


@cli.command("model")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
@click.option(
    "--output", type=click.Path(path_type=Path), default=Path("data/prediction.tif")
)
@click.option("--img-size", type=int, nargs=2, default=(512, 512))
@click.option("--bands", type=str, default="2 3 4 5 6 7 8 9")
@click.option("--base-c", type=int, default=32)
def model_cmd(
    input_path: Path,
    checkpoint: Path | None,
    output: Path,
    img_size: tuple[int, int],
    bands: str,
    base_c: int,
) -> None:
    # Click does not natively support argparse's nargs='+' on options cleanly.
    # To pass multiple space-separated bands as a single string, wrap them in quotes in the CLI:
    # python predict.py model <path> --img-size 512 512 --bands "2 3 4 5 6 7 8 9"
    checkpoint_path = checkpoint if checkpoint else find_latest_best_model()
    parsed_bands = [int(b) for b in bands.replace(",", " ").split()]

    predict_temporal_pair(
        early_path=input_path,
        checkpoint_path=checkpoint_path,
        output_path=output,
        img_size=img_size,
        bands=parsed_bands,
        base_c=base_c,
    )


if __name__ == "__main__":
    cli()
