import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def norm01(x, eps=1e-6):
    x = x.astype(np.float32)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


def read_dem(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def derive(dem):
    dem = norm01(dem)
    gy, gx = np.gradient(dem)
    slope = norm01(np.sqrt(gx * gx + gy * gy))
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    curvature = norm01(np.abs(gxx + gyy))
    mean = cv2.blur(dem, (7, 7))
    mean2 = cv2.blur(dem * dem, (7, 7))
    roughness = norm01(np.sqrt(np.maximum(mean2 - mean * mean, 0)))
    return np.stack([dem, slope, curvature, roughness], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dem_dir = Path(args.dem_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in dem_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    for p in files:
        topo = derive(read_dem(p))
        np.save(out_dir / f"{p.stem}.npy", topo)
    print(f"Converted {len(files)} DEM files to {out_dir}")


if __name__ == "__main__":
    main()
