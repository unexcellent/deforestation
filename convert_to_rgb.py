import numpy as np
from pathlib import Path
import rasterio

# -------------------------
# RGB EXTRACTION
# -------------------------

def extract_rgb(src):
    red = src.read(4).astype(np.float32)
    green = src.read(3).astype(np.float32)
    blue = src.read(2).astype(np.float32)

    rgb = np.stack([red, green, blue], axis=0)  # (C, H, W)

    # -------------------------
    # NORMALIZATION
    # -------------------------
    # Clip using percentiles to remove outliers
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip(rgb, p2, p98)

    # Scale to [0, 1]
    rgb = (rgb - p2) / (p98 - p2 + 1e-6)

    return rgb


# -------------------------
# MAIN
# -------------------------

def main(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root) / "images"

    output_root.mkdir(parents=True, exist_ok=True)

    # iterate train / test / etc.
    for split_dir in input_root.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name  # train / test

        for tile_dir in split_dir.iterdir():
            if not tile_dir.is_dir():
                continue

            tile_name = tile_dir.name

            for tif_path in tile_dir.glob("*.tif"):

                # skip incomplete downloads
                if ".tif." in tif_path.name:
                    continue

                rel_path = Path(split_name) / tile_name
                out_dir = output_root / rel_path
                out_dir.mkdir(parents=True, exist_ok=True)

                out_name = tif_path.stem + ".npy"
                out_path = out_dir / out_name

                try:
                    with rasterio.open(tif_path) as src:
                        rgb = extract_rgb(src)

                    np.save(out_path, rgb)

                    print(f"[OK] {rel_path / out_name}")

                except Exception as e:
                    print(f"[SKIP] {tif_path} -> {e}")


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    main(
        input_root="./data/makeathon-challenge/sentinel-2",
        output_root="./data/preprocessed/sentinel-2"
    )