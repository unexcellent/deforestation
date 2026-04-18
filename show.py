import click
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
import json


def normalize_rgb(
    data: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    """
    Normalizes a multi-band array using percentile stretching.
    """
    # Rescale each band independently for better color balance
    stretched = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        band = data[i].astype(np.float32)
        # Filter out 0 (usually NoData) for percentile calculation
        valid_pixels = band[band > 0]

        if valid_pixels.size > 0:
            low = np.percentile(valid_pixels, p_low)
            high = np.percentile(valid_pixels, p_high)
            stretched[i] = np.clip((band - low) / (high - low), 0, 1)
        else:
            stretched[i] = band

    # Move from (bands, rows, cols) to (rows, cols, bands) for matplotlib
    return np.transpose(stretched, (1, 2, 0))


def plot_image(image: np.ndarray, title: str) -> None:
    """
    Renders the image using matplotlib.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


@click.group()
def _cli() -> None:
    pass


@_cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--indices",
    "-i",
    nargs=3,
    type=int,
    default=[4, 3, 2],
    help="The band indices for Red, Green, and Blue (1-based).",
)
def image(path: str, indices: Sequence[int]) -> None:
    """
    Reads a multi-band TIFF and displays an RGB composite.
    """
    with rasterio.open(path) as src:
        # Rasterio uses 1-based indexing for bands
        if max(indices) > src.count:
            raise click.BadParameter(
                f"Requested band index {max(indices)} exceeds file band count ({src.count})."
            )

        rgb_data = src.read(list(indices))

    processed_image = normalize_rgb(rgb_data)
    plot_image(processed_image, f"Sentinel-2 RGB: {path}")


@_cli.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str) -> None:
    """
    Displays all metadata associated with a given TIFF.
    """
    with rasterio.open(path) as src:
        profile = {k: str(v) for k, v in src.profile.items()}
        tags = src.tags()

        # Collect band-specific descriptions and tags
        band_info = {}
        for i in range(1, src.count + 1):
            band_info[f"Band {i}"] = {
                "description": src.descriptions[i - 1],
                "tags": src.tags(i),
                "color_interp": src.colorinterp[i - 1].name,
            }

        metadata = {
            "profile": profile,
            "global_tags": tags,
            "bands": band_info,
        }

    click.echo(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    _cli()
