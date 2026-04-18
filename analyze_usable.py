import numpy as np
import rasterio
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray
from tqdm import tqdm


@dataclass(frozen=True)
class ImageMetrics:
    """Stores quality metrics for a single satellite image."""

    file_path: Path
    nodata_fraction: float
    cloud_fraction: float


def _calculate_nodata_fraction(image_array: NDArray[np.uint16]) -> float:
    """Calculates the fraction of pixels that are 0 across all bands."""
    nodata_mask = np.all(image_array == 0, axis=0)
    return float(np.mean(nodata_mask))


def _calculate_cloud_fraction(image_array: NDArray[np.uint16]) -> float:
    """Estimates cloud fraction using a threshold on the Blue band."""
    blue_band = image_array[1]
    cloud_mask = blue_band > 2000
    return float(np.mean(cloud_mask))


def process_image(file_path: Path) -> ImageMetrics:
    """Reads a GeoTIFF and computes quality metrics."""
    with rasterio.open(file_path) as src:
        image_array: NDArray[np.uint16] = src.read()

    return ImageMetrics(
        file_path=file_path,
        nodata_fraction=_calculate_nodata_fraction(image_array),
        cloud_fraction=_calculate_cloud_fraction(image_array),
    )


def analyze_dataset(directory: Path) -> tuple[ImageMetrics, ...]:
    """Processes all TIFF files in the specified directory."""
    return tuple(process_image(p) for p in tqdm(list(directory.rglob("*.tif"))))


def main() -> None:
    """Executes the analysis and prints sorted results."""
    target_dir = Path("data/makeathon-challenge/sentinel-2/train")
    metrics = analyze_dataset(target_dir)

    sorted_metrics = sorted(metrics, key=lambda m: m.nodata_fraction, reverse=True)

    for metric in sorted_metrics:
        print(f"{metric.file_path.name}: {metric.nodata_fraction:.1%} No-Data")


if __name__ == "__main__":
    main()
