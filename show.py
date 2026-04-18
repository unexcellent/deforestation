import json
from typing import Sequence

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def normalize_bands(
    data: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    """
    Normalizes a multi-band or single-band array using percentile stretching.
    """
    stretched = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        band = data[i].astype(np.float32)
        valid_pixels = band[band > 0]

        if valid_pixels.size > 0:
            low = np.percentile(valid_pixels, p_low)
            high = np.percentile(valid_pixels, p_high)
            stretched[i] = np.clip((band - low) / (high - low), 0, 1)
        else:
            stretched[i] = band

    if stretched.shape[0] == 1:
        return stretched[0]

    return np.transpose(stretched, (1, 2, 0))


def reproject_raster(
    source_array: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    dst_shape: tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: rasterio.crs.CRS,
) -> np.ndarray:
    """
    Reprojects a 2D raster array to match a destination coordinate system.
    """
    destination = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=source_array,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return destination


def plot_image(image: np.ndarray, title: str) -> None:
    """
    Renders the image using matplotlib.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_overlay(base: np.ndarray, overlay: np.ndarray, title: str) -> None:
    """
    Renders a grayscale base image with a red overlay for positive label pixels.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(base, cmap="gray")

    masked_overlay = np.ma.masked_where(overlay == 0, overlay)
    cmap = mcolors.ListedColormap(["red"])

    plt.imshow(masked_overlay, cmap=cmap, interpolation="nearest", alpha=0.6)
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
    Reads a TIFF and displays it. Uses RGB for multi-band and grayscale for single-band.
    """
    with rasterio.open(path) as src:
        if src.count == 1:
            click.echo("Single band detected. Displaying grayscale.")
            data = src.read([1])
        else:
            if max(indices) > src.count:
                raise click.BadParameter(
                    f"Requested band index {max(indices)} exceeds file band count ({src.count})."
                )
            data = src.read(list(indices))

    processed_image = normalize_bands(data)
    plot_image(processed_image, f"Displaying: {path}")


@_cli.command()
@click.argument("base_path", type=click.Path(exists=True))
@click.argument("label_path", type=click.Path(exists=True))
@click.option(
    "--indices",
    "-i",
    nargs=3,
    type=int,
    default=[4, 3, 2],
    help="The band indices to average into grayscale (1-based).",
)
def overlay(base_path: str, label_path: str, indices: Sequence[int]) -> None:
    """
    Averages base image bands into grayscale and overlays reprojected labels in red.
    """
    with rasterio.open(base_path) as base_src:
        if max(indices) > base_src.count:
            raise click.BadParameter(
                f"Requested band index {max(indices)} exceeds file band count ({base_src.count})."
            )
        base_data = base_src.read(list(indices))
        base_transform = base_src.transform
        base_crs = base_src.crs
        base_shape = (base_src.height, base_src.width)

    grayscale_base = np.mean(base_data, axis=0, keepdims=True)
    processed_base = normalize_bands(grayscale_base)

    with rasterio.open(label_path) as label_src:
        label_data = label_src.read(1)

        reprojected_label = reproject_raster(
            source_array=label_data,
            src_transform=label_src.transform,
            src_crs=label_src.crs,
            dst_shape=base_shape,
            dst_transform=base_transform,
            dst_crs=base_crs,
        )

    plot_overlay(
        processed_base, reprojected_label, f"Overlay: {base_path} + {label_path}"
    )


@_cli.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str) -> None:
    """
    Displays all metadata associated with a given TIFF.
    """
    with rasterio.open(path) as src:
        profile = {k: str(v) for k, v in src.profile.items()}
        tags = src.tags()

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
