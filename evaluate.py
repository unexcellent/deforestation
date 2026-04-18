from pathlib import Path
import click
import numpy as np
import rasterio
import rasterio.features
from process import generate_merged_labels


def calculate_iou(pred: np.ndarray, true: np.ndarray) -> float:
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


@click.group()
def _cli() -> None:
    pass


@_cli.command()
@click.option(
    "--diff-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./data/preprocessed/nvdi-diffs"),
)
@click.option(
    "--labels-base",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./data/makeathon-challenge/labels"),
)
@click.option(
    "--sieve-size", type=int, default=0, help="Minimum pixel cluster size to keep."
)
def nvdis(diff_dir: Path, labels_base: Path, sieve_size: int) -> None:
    """
    Calculates IoU for precomputed NDVI diffs with a live running average and optional sieve.
    """
    diff_files = sorted(list(diff_dir.rglob("*.tif")))

    total_iou = 0.0
    count = 0

    for diff_path in diff_files:
        name = diff_path.stem
        parts = name.split("_diff_")
        early_stem = parts[0]
        late_stem = parts[1]

        split = "test" if "test" in diff_path.parts else "train"
        tile_id = early_stem.split("__")[0]

        if split == "test":
            continue

        s2_base = (
            Path("./data/makeathon-challenge/sentinel-2") / split / f"{tile_id}__s2_l2a"
        )
        early_s2 = s2_base / f"{early_stem}.tif"
        late_s2 = s2_base / f"{late_stem}.tif"

        # Ground Truth Label generation
        labels_dir = labels_base / split
        tmp_label_path = Path(f"./data/preprocessed/tmp_diff_{name}.tif")
        generate_merged_labels(late_s2, labels_dir, tmp_label_path, early_s2)

        with rasterio.open(diff_path) as d_src, rasterio.open(tmp_label_path) as l_src:
            pred = d_src.read(1)
            true = l_src.read(1)

            if sieve_size > 0:
                pred = rasterio.features.sieve(pred, size=sieve_size, connectivity=8)

            iou = calculate_iou(pred, true)

        total_iou += iou
        count += 1
        avg_iou = total_iou / count

        e_p = early_stem.split("_")
        l_p = late_stem.split("_")
        click.echo(
            f"{tile_id} {e_p[-2]}/{e_p[-1]}-{l_p[-2]}/{l_p[-1]}: {iou:.4f} (Avg: {avg_iou:.4f})"
        )

        tmp_label_path.unlink()


if __name__ == "__main__":
    _cli()
