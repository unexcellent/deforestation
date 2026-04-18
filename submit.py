import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import rasterio
import tqdm
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
    time_step: str | None = None,
) -> dict:
    """Convert a binary deforestation prediction raster to a GeoJSON FeatureCollection.

    Reads a single-band GeoTIFF where 1 indicates deforestation and 0 indicates
    no deforestation, vectorises the foreground pixels into polygons, removes
    polygons smaller than ``min_area_ha``, reprojects the result to EPSG:4326,
    and returns (and optionally writes) a GeoJSON FeatureCollection.

    The caller is responsible for binarising their model output before passing
    it to this function. This function is designed to be the final step in the
    submission pipeline: take your binarised prediction raster, call this
    function, and upload the resulting ``.geojson`` file to the leaderboard.

    Args:
        raster_path: Path to the input GeoTIFF. Must be a single-band raster
            with binary values (0 = no deforestation, 1 = deforestation).
        output_path: Optional path at which to write the GeoJSON file. Parent
            directories are created automatically. If ``None``, the result is
            returned but not written to disk.
        min_area_ha: Minimum polygon area in hectares. Polygons smaller than
            this threshold are removed before the output is written. Area is
            computed in the appropriate UTM projection so the filter is
            metric-accurate regardless of the raster's native CRS. Defaults
            to ``0.5``.
        time_step: Optional string representing the YYMM date of deforestation.

    Returns:
        A GeoJSON-compatible ``dict`` representing a FeatureCollection. Each
        Feature corresponds to one contiguous deforestation polygon in
        EPSG:4326 (longitude/latitude, WGS-84).

    Raises:
        FileNotFoundError: If ``raster_path`` does not point to an existing file.
        ValueError: If the raster contains no deforestation pixels (all zeros),
            or if all polygons are smaller than ``min_area_ha``.

    Example:
        >>> geojson = raster_to_geojson(
        ...     raster_path="predictions/tile_18NVJ.tif",
        ...     output_path="submission/tile_18NVJ.geojson",
        ...     min_area_ha=0.5,
        ... )
        >>> print(len(geojson["features"]), "deforestation polygons")
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        raise ValueError(
            f"No deforestation pixels (value=1) found in {raster_path}. "
            "Ensure the raster has been binarised before calling this function."
        )

    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    gdf["time_step"] = time_step

    geojson = json.loads(gdf.to_json())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)

    return geojson


def _aggregate_diffs(diff_paths: list[Path], min_area_ha: float) -> dict | None:
    if not diff_paths:
        return None

    with rasterio.open(diff_paths[0]) as src:
        meta = src.meta.copy()
        base_shape = (src.height, src.width)
        base_transform = src.transform
        base_crs = src.crs

    combined_mask = np.zeros(base_shape, dtype=np.uint8)

    for path in diff_paths:
        with rasterio.open(path) as src:
            data = src.read(1)

            if data.shape != base_shape:
                reprojected_data = np.zeros(base_shape, dtype=data.dtype)
                reproject(
                    source=data,
                    destination=reprojected_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=base_transform,
                    dst_crs=base_crs,
                    resampling=Resampling.nearest,
                )
                data = reprojected_data

            combined_mask |= (data > 0).astype(np.uint8)

    meta.update(dtype="uint8", nodata=0, count=1)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(combined_mask, 1)

        return raster_to_geojson(tmp_path, output_path=None, min_area_ha=min_area_ha)
    except ValueError:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--diffs-dir", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=False)
@click.option("--min-area-ha", type=float, default=0.5)
def nvdis(
    diffs_dir: Path | None = None,
    output_dir: Path | None = None,
    min_area_ha: float = 0.5,
) -> None:
    """
    Aggregates binary difference rasters for each location and exports them to a single GeoJSON per split.
    """
    base_path = Path(__file__).parent
    if diffs_dir is None:
        diffs_dir = base_path / "data/preprocessed/nvdi-diffs"
    if output_dir is None:
        output_dir = base_path / "data/submission"

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for f in diffs_dir.rglob("*.tif"):
        split = f.parent.name
        loc_id = f.name.split("__s2_l2a")[0]
        groups[(split, loc_id)].append(f)

    split_features: dict[str, list[dict]] = defaultdict(list)

    for (split, loc_id), paths in tqdm.tqdm(
        groups.items(), desc="Aggregating locations"
    ):
        geojson_data = _aggregate_diffs(paths, min_area_ha)
        if geojson_data and "features" in geojson_data:
            split_features[split].extend(geojson_data["features"])

    for split, features in split_features.items():
        if not features:
            continue

        final_geojson = {"type": "FeatureCollection", "features": features}

        out_path = output_dir / f"{split}_submission.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(final_geojson, f)


@cli.command("model")
@click.option(
    "--preds-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/model_output_test"),
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("data/submission/test_submission.geojson"),
)
@click.option("--min-area-ha", type=float, default=0.5)
def model_cmd(preds_dir: Path, output_path: Path, min_area_ha: float) -> None:
    """
    Creates the final GeoJSON by accumulating deforestation predictions chronologically.
    Assigns the time_step (YYMM) based on the subsequent year's observation.
    """
    loc_groups: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)

    for f in preds_dir.glob("*_pred.tif"):
        match = re.search(r"([A-Z0-9_]+)__s2_l2a_(\d{4})_(\d+)", f.name)
        if match:
            loc_id = match.group(1)
            year = int(match.group(2))
            month = int(match.group(3))
            loc_groups[loc_id].append((year, month, f))

    all_features = []

    for loc_id, items in tqdm.tqdm(
        loc_groups.items(), desc="Processing temporal predictions"
    ):
        items.sort(key=lambda x: (x[0], x[1]))

        accumulated_mask = None
        base_meta = None

        for early_year, month, fpath in items:
            with rasterio.open(fpath) as src:
                data = src.read(1)

                if base_meta is None:
                    base_meta = src.meta.copy()
                    accumulated_mask = np.zeros_like(data, dtype=np.uint8)
                elif data.shape != accumulated_mask.shape:
                    reprojected_data = np.zeros_like(accumulated_mask)
                    reproject(
                        source=data,
                        destination=reprojected_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=base_meta["transform"],
                        dst_crs=base_meta["crs"],
                        resampling=Resampling.nearest,
                    )
                    data = reprojected_data

            new_deforestation = ((data == 1) & (accumulated_mask == 0)).astype(np.uint8)

            if new_deforestation.sum() > 0:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                base_meta.update(dtype="uint8", nodata=0, count=1)
                with rasterio.open(tmp_path, "w", **base_meta) as dst:
                    dst.write(new_deforestation, 1)

                late_year = early_year + 1
                time_step = f"{late_year % 100:02d}{month:02d}"

                try:
                    geojson = raster_to_geojson(
                        raster_path=tmp_path,
                        min_area_ha=min_area_ha,
                        time_step=time_step,
                    )
                    all_features.extend(geojson["features"])
                except ValueError:
                    pass
                finally:
                    tmp_path.unlink(missing_ok=True)

            accumulated_mask |= (data == 1).astype(np.uint8)

    if not all_features:
        click.secho("No features passed the threshold.", fg="yellow")
        return

    final_geojson = {"type": "FeatureCollection", "features": all_features}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(final_geojson, f)

    click.secho(f"Submission successfully saved to {output_path}", fg="green")


if __name__ == "__main__":
    cli()
