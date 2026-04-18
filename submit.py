import json
import tempfile
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

"""Utilities for converting deforestation prediction rasters into submittable GeoJSON."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
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

    # Vectorise connected foreground regions into polygons
    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    # Filter by area: project to UTM for metric-accurate ha calculation
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    gdf["time_step"] = None

    geojson = json.loads(gdf.to_json())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)

    return geojson



def _merge_location_tifs(paths: list[Path]) -> tuple[np.ndarray, dict]:
    """Merges multiple temporal rasters into a single binary mask."""
    with rasterio.open(paths[0]) as src:
        meta = src.meta.copy()
        base_shape = (src.height, src.width)
        base_transform = src.transform
        base_crs = src.crs

    # Combined mask: 1 if any time-step predicted deforestation (> 0)
    combined = np.zeros(base_shape, dtype=np.uint8)

    for p in paths:
        with rasterio.open(p) as src:
            data = src.read(1)
            
            if data.shape != base_shape:
                reprojected = np.zeros(base_shape, dtype=data.dtype)
                reproject(
                    source=data,
                    destination=reprojected,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=base_transform,
                    dst_crs=base_crs,
                    resampling=Resampling.nearest,
                )
                data = reprojected

            combined |= (data > 0).astype(np.uint8)

    meta.update(dtype="uint8", nodata=0, count=1)
    return combined, meta


@click.command()
@click.option(
    "--preds-dir", 
    type=click.Path(exists=True, path_type=Path), 
    default="data/model_output_tif",
    help="Directory containing your *_pred.tif model outputs."
)
@click.option(
    "--output-path", 
    type=click.Path(path_type=Path), 
    default="submission/final_submission.geojson",
    help="Path for the final combined GeoJSON."
)
@click.option("--min-area-ha", type=float, default=0.5)
def main(preds_dir: Path, output_path: Path, min_area_ha: float) -> None:
    """
    Groups predictions by location, merges time-steps, and generates a submission GeoJSON.
    """
    # 1. Group files by location ID (e.g., 18NWG_6_6)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in preds_dir.rglob("*_pred.tif"):
        # Extracts 18NWG_6_6 from 18NWG_6_6__s2_l2a_2020_1_pred.tif
        loc_id = f.name.split("__s2_l2a")[0]
        groups[loc_id].append(f)

    all_features = []
    print(f"Aggregating {len(groups)} locations...")

    # 2. Process each location
    for loc_id, paths in tqdm(groups.items()):
        merged_data, meta = _merge_location_tifs(paths)
        
        if merged_data.sum() == 0:
            continue

        # Use tempfile to bridge between numpy/rasterio and the submission utility
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(merged_data, 1)

            # Convert to GeoJSON using the provided utility
            # We don't save intermediate GeoJSONs to disk, just collect features
            geojson = raster_to_geojson(tmp_path, output_path=None, min_area_ha=min_area_ha)
            all_features.extend(geojson.get("features", []))
            
        finally:
            tmp_path.unlink(missing_ok=True)

    # 3. Finalize Submission
    submission = {
        "type": "FeatureCollection",
        "features": all_features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)

    print(f"\nCreated submission with {len(all_features)} polygons at {output_path}")


if __name__ == "__main__":
    main()