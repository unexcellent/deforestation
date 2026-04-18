import calendar
import datetime
import re
from pathlib import Path
from typing import Any

import click
import numpy as np
import rasterio
import rasterio.features
import tqdm
from rasterio.warp import Resampling, reproject


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


def _parse_sentinel_metadata(path: Path) -> tuple[str, int, int]:
    """
    Parses the tile ID, year, and month from the Sentinel-2 filename.
    """
    match = re.search(r"([A-Z0-9_]+)__s2_l2a_(\d{4})_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse tile metadata from filename: {path.name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _get_glads2_mask(
    glads2_dir: Path, tile_id: str, year: int, month: int
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    """
    Retrieves and filters the GLAD-S2 mask for a given tile and date.
    """
    alert_path = glads2_dir / f"glads2_{tile_id}_alert.tif"
    date_path = glads2_dir / f"glads2_{tile_id}_alertDate.tif"

    if not alert_path.exists() or not date_path.exists():
        return None

    last_day = calendar.monthrange(year, month)[1]
    target_date = datetime.date(year, month, last_day)
    epoch_date = datetime.date(2019, 1, 1)
    max_days = (target_date - epoch_date).days

    with rasterio.open(alert_path) as alert_src, rasterio.open(date_path) as date_src:
        alert_data = alert_src.read(1)
        date_data = date_src.read(1)
        crs = alert_src.crs
        transform = alert_src.transform

    mask = ((alert_data > 0) & (date_data > 0) & (date_data <= max_days)).astype(
        np.uint8
    )

    return mask, transform, crs


def _get_radd_mask(
    radd_dir: Path, tile_id: str, year: int, month: int
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    """
    Retrieves and filters the RADD mask for a given tile and date.
    """
    radd_path = radd_dir / f"radd_{tile_id}_labels.tif"

    if not radd_path.exists():
        return None

    last_day = calendar.monthrange(year, month)[1]
    target_date = datetime.date(year, month, last_day)
    epoch_date = datetime.date(2014, 12, 31)
    max_days = (target_date - epoch_date).days

    with rasterio.open(radd_path) as radd_src:
        radd_data = radd_src.read(1)
        crs = radd_src.crs
        transform = radd_src.transform

    days_data = radd_data % 10000
    mask = ((radd_data > 0) & (days_data <= max_days)).astype(np.uint8)

    return mask, transform, crs


def _create_merged_mask_for_path(
    sentinel_path: Path, labels_base_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Creates a combined binary deforestation mask up to the date of the Sentinel-2 image.
    """
    tile_id, year, month = _parse_sentinel_metadata(sentinel_path)

    with rasterio.open(sentinel_path) as s2_src:
        s2_meta = s2_src.meta.copy()
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs
        s2_shape = (s2_src.height, s2_src.width)

    merged_mask = np.zeros(s2_shape, dtype=np.uint8)

    glads2_result = _get_glads2_mask(labels_base_dir / "glads2", tile_id, year, month)
    if glads2_result:
        mask, transform, crs = glads2_result
        reprojected = reproject_raster(
            source_array=mask,
            src_transform=transform,
            src_crs=crs,
            dst_shape=s2_shape,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
        )
        merged_mask |= reprojected

    radd_result = _get_radd_mask(labels_base_dir / "radd", tile_id, year, month)
    if radd_result:
        mask, transform, crs = radd_result
        reprojected = reproject_raster(
            source_array=mask,
            src_transform=transform,
            src_crs=crs,
            dst_shape=s2_shape,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
        )
        merged_mask |= reprojected

    return merged_mask, s2_meta


def generate_merged_labels(
    sentinel_path: Path,
    labels_base_dir: Path,
    output_path: Path,
    earlier_path: Path | None = None,
) -> None:
    """
    Generates a merged binary deforestation mask matching the spatial extent and CRS of the given Sentinel-2 TIFF.
    If an earlier_path is provided, it returns only the new deforestation alerts between the two images.
    """
    merged_mask, s2_meta = _create_merged_mask_for_path(sentinel_path, labels_base_dir)

    if earlier_path is not None:
        earlier_mask, _ = _create_merged_mask_for_path(earlier_path, labels_base_dir)
        merged_mask = merged_mask & (earlier_mask == 0)

    s2_meta.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **s2_meta) as dst:
        dst.write(merged_mask, 1)


def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Vegetation Index from Red and NIR arrays.
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    denominator = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator == 0, 0.0, (nir - red) / denominator)

    return ndvi.astype(np.float32)


def generate_ndvi_raster(
    sentinel_path: Path,
    output_path: Path,
    red_idx: int,
    nir_idx: int,
    earlier_path: Path | None = None,
    threshold: float | None = None,
    sieve_size: int = 0,
) -> None:
    """
    Reads Red and NIR bands from a multispectral TIFF, calculates NDVI, and writes a single-band TIFF.
    If threshold is provided alongside earlier_path, it binarizes both images first to isolate
    areas that transitioned from "above threshold" to "below threshold".
    If sieve_size > 0 and threshold is active, removes salt-and-pepper noise from the final mask.
    """
    with rasterio.open(sentinel_path) as src:
        red_data = src.read(red_idx)
        nir_data = src.read(nir_idx)
        meta = src.meta.copy()

    ndvi_data = calculate_ndvi(red_data, nir_data)

    if earlier_path is not None:
        with rasterio.open(earlier_path) as src_earlier:
            red_earlier = src_earlier.read(red_idx)
            nir_earlier = src_earlier.read(nir_idx)

        earlier_ndvi = calculate_ndvi(red_earlier, nir_earlier)

        if threshold is not None:
            earlier_forest = (earlier_ndvi > threshold).astype(np.uint8)
            current_forest = (ndvi_data > threshold).astype(np.uint8)

            # Isolated loss: Was forest, is now NOT forest
            ndvi_data = ((earlier_forest == 1) & (current_forest == 0)).astype(np.uint8)

            if sieve_size > 0:
                ndvi_data = rasterio.features.sieve(
                    ndvi_data, size=sieve_size, connectivity=8
                )

            meta.update(count=1, dtype="uint8", compress="lzw", nodata=0)
        else:
            # Raw difference
            ndvi_data = np.where(
                earlier_ndvi > ndvi_data, earlier_ndvi - ndvi_data, 0.0
            ).astype(np.float32)
            meta.update(count=1, dtype="float32", compress="lzw", nodata=None)

    else:
        if threshold is not None:
            ndvi_data = (ndvi_data > threshold).astype(np.uint8)

            if sieve_size > 0:
                ndvi_data = rasterio.features.sieve(
                    ndvi_data, size=sieve_size, connectivity=8
                )

            meta.update(count=1, dtype="uint8", compress="lzw", nodata=0)
        else:
            meta.update(count=1, dtype="float32", compress="lzw", nodata=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(ndvi_data, 1)


@click.group()
def _cli() -> None:
    pass


@_cli.command()
@click.argument("sentinel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--earlier-path", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option(
    "--labels-dir", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--output-path", type=click.Path(path_type=Path), required=False)
def label(
    sentinel_path: Path,
    earlier_path: Path | None = None,
    labels_dir: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Generates a combined binary deforestation mask up to the date of the given Sentinel-2 image.
    If an earlier image path is provided, returns only the deforestation alerts between the two images.
    """
    split = "test" if "test" in sentinel_path.parts else "train"

    if labels_dir is None:
        labels_dir = Path(__file__).parent / f"data/makeathon-challenge/labels/{split}"
    if output_path is None:
        output_path = (
            Path(__file__).parent
            / f"data/preprocessed/labels/{split}/{sentinel_path.stem}-label.tif"
        )

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    generate_merged_labels(sentinel_path, labels_dir, output_path, earlier_path)


@_cli.command()
@click.option("--s2-dir", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--labels-dir", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=False)
def labels(
    s2_dir: Path | None = None,
    labels_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Iterates over all Sentinel-2 TIFFs and generates merged labels for each, matching train/test splits.
    """
    base_path = Path(__file__).parent

    if s2_dir is None:
        s2_dir = base_path / "data/makeathon-challenge/sentinel-2"
    if labels_dir is None:
        labels_dir = base_path / "data/makeathon-challenge/labels"
    if output_dir is None:
        output_dir = base_path / "data/preprocessed/labels"

    s2_files = list(s2_dir.rglob("*.tif"))
    if not s2_files:
        click.echo(f"No Sentinel-2 TIFFs found in {s2_dir}")
        return

    for sentinel_path in tqdm.tqdm(s2_files, desc="Processing labels"):
        split = "test" if "test" in sentinel_path.parts else "train"

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        split_labels_dir = labels_dir / split
        output_path = split_output_dir / f"{sentinel_path.stem}-label.tif"

        generate_merged_labels(sentinel_path, split_labels_dir, output_path)


@_cli.command()
@click.argument("sentinel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--earlier-path", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--output-path", type=click.Path(path_type=Path), required=False)
@click.option("--red-band", type=int, default=4, help="1-based index for the Red band")
@click.option("--nir-band", type=int, default=8, help="1-based index for the NIR band")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Binarize output: 1 if > threshold else 0",
)
@click.option(
    "--sieve-size",
    type=int,
    default=0,
    help="Minimum pixel cluster size to keep (removes noise). Requires threshold.",
)
def nvdi(
    sentinel_path: Path,
    earlier_path: Path | None = None,
    output_path: Path | None = None,
    red_band: int = 4,
    nir_band: int = 8,
    threshold: float | None = None,
    sieve_size: int = 0,
) -> None:
    """
    Generates a single-band NDVI raster from a multi-band Sentinel-2 image.
    If an earlier image path is provided, isolated areas of deforestation are returned.
    """
    if output_path is None:
        split = "test" if "test" in sentinel_path.parts else "train"
        output_path = (
            Path(__file__).parent
            / f"data/preprocessed/ndvi/{split}/{sentinel_path.stem}-ndvi.tif"
        )

    generate_ndvi_raster(
        sentinel_path,
        output_path,
        red_band,
        nir_band,
        earlier_path,
        threshold,
        sieve_size,
    )


@_cli.command()
@click.option("--s2-dir", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--output-dir", type=click.Path(path_type=Path), required=False)
@click.option("--red-band", type=int, default=4, help="1-based index for the Red band")
@click.option("--nir-band", type=int, default=8, help="1-based index for the NIR band")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Binarize output: 1 if > threshold else 0",
)
@click.option(
    "--sieve-size",
    type=int,
    default=0,
    help="Minimum pixel cluster size to keep (removes noise). Requires threshold.",
)
def nvdis(
    s2_dir: Path | None = None,
    output_dir: Path | None = None,
    red_band: int = 4,
    nir_band: int = 8,
    threshold: float | None = None,
    sieve_size: int = 0,
) -> None:
    """
    Iterates over all Sentinel-2 TIFFs and generates a single-band NDVI raster for each.
    """
    base_path = Path(__file__).parent

    if s2_dir is None:
        s2_dir = base_path / "data/makeathon-challenge/sentinel-2"
    if output_dir is None:
        output_dir = base_path / "data/preprocessed/ndvi"

    s2_files = list(s2_dir.rglob("*.tif"))
    if not s2_files:
        click.echo(f"No Sentinel-2 TIFFs found in {s2_dir}")
        return

    for sentinel_path in tqdm.tqdm(s2_files, desc="Processing tree indices"):
        split = "test" if "test" in sentinel_path.parts else "train"

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = split_output_dir / f"{sentinel_path.stem}-ndvi.tif"

        generate_ndvi_raster(
            sentinel_path, output_path, red_band, nir_band, None, threshold, sieve_size
        )


if __name__ == "__main__":
    _cli()
