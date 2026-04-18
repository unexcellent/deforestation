import calendar
import datetime
import re
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def reproject_raster(
    source_array: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    dst_shape: tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: rasterio.crs.CRS,
) -> np.ndarray:
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
    match = re.search(r"([A-Z0-9_]+)__s2_l2a_(\d{4})_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse tile metadata from filename: {path.name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _get_glads2_mask(
    glads2_dir: Path, tile_id: str, year: int, month: int
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    alert_path = glads2_dir / f"glads2_{tile_id}_alert.tif"
    date_path = glads2_dir / f"glads2_{tile_id}_alertDate.tif"

    if not alert_path.exists() or not date_path.exists():
        return None

    last_day = calendar.monthrange(year, month)[1]
    target_date = datetime.date(year, month, last_day)
    epoch_date = datetime.date(2019, 1, 1)
    max_days = (target_date - epoch_date).days

    click.echo(
        f"[GLAD-S2] Target Date: {target_date} (Max days since epoch: {max_days})"
    )

    with rasterio.open(alert_path) as alert_src, rasterio.open(date_path) as date_src:
        alert_data = alert_src.read(1)
        date_data = date_src.read(1)
        crs = alert_src.crs
        transform = alert_src.transform

    total_alerts = np.count_nonzero(alert_data > 0)
    click.echo(f"[GLAD-S2] Total raw alerts found: {total_alerts}")

    mask = ((alert_data > 0) & (date_data > 0) & (date_data <= max_days)).astype(
        np.uint8
    )

    applied_alerts = np.count_nonzero(mask)
    click.echo(f"[GLAD-S2] Alerts applied (within date range): {applied_alerts}")

    return mask, transform, crs


def _get_radd_mask(
    radd_dir: Path, tile_id: str, year: int, month: int
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    radd_path = radd_dir / f"radd_{tile_id}_labels.tif"

    if not radd_path.exists():
        return None

    last_day = calendar.monthrange(year, month)[1]
    target_date = datetime.date(year, month, last_day)
    epoch_date = datetime.date(2014, 12, 31)
    max_days = (target_date - epoch_date).days

    click.echo(f"[RADD] Target Date: {target_date} (Max days since epoch: {max_days})")

    with rasterio.open(radd_path) as radd_src:
        radd_data = radd_src.read(1)
        crs = radd_src.crs
        transform = radd_src.transform

    total_alerts = np.count_nonzero(radd_data > 0)
    click.echo(f"[RADD] Total raw alerts found: {total_alerts}")

    days_data = radd_data % 10000
    mask = ((radd_data > 0) & (days_data <= max_days)).astype(np.uint8)

    applied_alerts = np.count_nonzero(mask)
    click.echo(f"[RADD] Alerts applied (within date range): {applied_alerts}")

    return mask, transform, crs


def generate_merged_labels(
    sentinel_path: Path,
    labels_base_dir: Path,
    output_path: Path,
) -> None:
    tile_id, year, month = _parse_sentinel_metadata(sentinel_path)
    click.echo(f"--- Processing Tile: {tile_id} | Date: {year}-{month:02d} ---")

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
        click.echo(
            f"[GLAD-S2] Reprojected mask non-zero pixels: {np.count_nonzero(reprojected)}"
        )
        merged_mask |= reprojected
    else:
        click.echo("[GLAD-S2] Files not found for this tile.")

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
        click.echo(
            f"[RADD] Reprojected mask non-zero pixels: {np.count_nonzero(reprojected)}"
        )
        merged_mask |= reprojected
    else:
        click.echo("[RADD] Files not found for this tile.")

    final_count = np.count_nonzero(merged_mask)
    click.echo(f"--- Total merged deforestation pixels: {final_count} ---")

    s2_meta.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **s2_meta) as dst:
        dst.write(merged_mask, 1)


@click.command()
@click.argument("sentinel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--labels-dir", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--output-path", type=click.Path(path_type=Path), required=False)
def label(
    sentinel_path: Path, labels_dir: Path | None = None, output_path: Path | None = None
) -> None:
    if labels_dir is None:
        labels_dir = Path(__file__).parent / "data/makeathon-challenge/labels/train"
    if output_path is None:
        output_path = (
            Path(__file__).parent / f"data/processed/{sentinel_path.stem}-label.tif"
        )

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    generate_merged_labels(sentinel_path, labels_dir, output_path)


if __name__ == "__main__":
    label()
