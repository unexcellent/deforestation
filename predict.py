from pathlib import Path

import click
import numpy as np
import rasterio
import rasterio.features

from process import _create_merged_mask_for_path, calculate_ndvi


def calculate_iou(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) between two binary arrays.
    """
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


@click.group()
def _cli() -> None:
    pass


@_cli.command()
@click.argument("earlier_path", type=click.Path(exists=True, path_type=Path))
@click.argument("later_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--labels-dir", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--red-band", type=int, default=4, help="1-based index for the Red band")
@click.option("--nir-band", type=int, default=8, help="1-based index for the NIR band")
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="NDVI threshold to define a pixel as forest",
)
@click.option(
    "--sieve-size",
    type=int,
    default=0,
    help="Minimum pixel cluster size to keep (removes noise).",
)
def nvdi(
    earlier_path: Path,
    later_path: Path,
    labels_dir: Path | None = None,
    red_band: int = 4,
    nir_band: int = 8,
    threshold: float = 0.5,
    sieve_size: int = 0,
) -> None:
    """
    Calculates IoU between NDVI-based deforestation predictions and ground truth labels.
    """
    split = "test" if "test" in later_path.parts else "train"
    if labels_dir is None:
        labels_dir = Path(__file__).parent / f"data/makeathon-challenge/labels/{split}"

    later_label, _ = _create_merged_mask_for_path(later_path, labels_dir)
    earlier_label, _ = _create_merged_mask_for_path(earlier_path, labels_dir)

    true_diff = ((later_label > 0) & (earlier_label == 0)).astype(np.uint8)

    with rasterio.open(earlier_path) as src:
        earlier_red = src.read(red_band)
        earlier_nir = src.read(nir_band)

    with rasterio.open(later_path) as src:
        later_red = src.read(red_band)
        later_nir = src.read(nir_band)

    earlier_ndvi = calculate_ndvi(earlier_red, earlier_nir)
    later_ndvi = calculate_ndvi(later_red, later_nir)

    earlier_forest = (earlier_ndvi > threshold).astype(np.uint8)
    later_forest = (later_ndvi > threshold).astype(np.uint8)

    pred_diff = ((earlier_forest == 1) & (later_forest == 0)).astype(np.uint8)

    if sieve_size > 0:
        pred_diff = rasterio.features.sieve(pred_diff, size=sieve_size, connectivity=8)

    iou = calculate_iou(pred_diff, true_diff)

    click.echo(f"Earlier Image: {earlier_path.name}")
    click.echo(f"Later Image:   {later_path.name}")
    click.echo(f"Threshold:     {threshold}")
    click.echo(f"Sieve Size:    {sieve_size}")
    click.echo(f"IoU:           {iou:.4f}")


if __name__ == "__main__":
    _cli()
