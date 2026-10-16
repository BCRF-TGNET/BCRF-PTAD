import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if is_image(p)])


def list_topos(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower() == ".npy")])


def build_stem_map(files):
    d = {}
    for p in files:
        if p.stem not in d or p.suffix.lower() == ".npy":
            d[p.stem] = p
    return d


def mask_non_empty(mask_path: Path):
    try:
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.max() > 0
    except Exception:
        return True


def match_by_stem(stem, file_map):
    if stem in file_map:
        return file_map[stem]
    for k, v in file_map.items():
        if stem in k or k in stem:
            return v
    return None


def split_items(items, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]


def write_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for row in items:
            f.write(" ".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to Bijie-landslide-dataset")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topo-folder-name", default="topo_npy")
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    land_img_dir = root / "landslide" / "image"
    land_mask_dir = root / "landslide" / "mask"
    land_topo_dir = root / "landslide" / args.topo_folder_name

    neg_img_dir = root / "non-landslide" / "image"
    neg_topo_dir = root / "non-landslide" / args.topo_folder_name

    land_images = list_images(land_img_dir)
    land_masks = build_stem_map(list_images(land_mask_dir))
    land_topos = build_stem_map(list_topos(land_topo_dir))

    neg_images = list_images(neg_img_dir)
    neg_topos = build_stem_map(list_topos(neg_topo_dir))

    pos_items = []
    missing_pos_mask = 0
    missing_pos_topo = 0
    empty_masks = 0

    for img in land_images:
        mask = match_by_stem(img.stem, land_masks)
        topo = match_by_stem(img.stem, land_topos)
        if mask is None:
            missing_pos_mask += 1
            continue
        if not mask_non_empty(mask):
            empty_masks += 1
            continue
        if topo is None:
            missing_pos_topo += 1
            topo_s = "none"
        else:
            topo_s = str(topo)
        pos_items.append((str(img), str(mask), topo_s))

    neg_items = []
    missing_neg_topo = 0
    for img in neg_images:
        topo = match_by_stem(img.stem, neg_topos)
        if topo is None:
            missing_neg_topo += 1
            topo_s = "none"
        else:
            topo_s = str(topo)
        neg_items.append((str(img), "none", topo_s))

    p_train, p_val, p_test = split_items(pos_items, args.train_ratio, args.val_ratio, args.seed)
    n_train, n_val, n_test = split_items(neg_items, args.train_ratio, args.val_ratio, args.seed)

    rng = random.Random(args.seed)
    train = p_train + n_train
    val = p_val + n_val
    test = p_test + n_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    write_list(out_dir / "train.txt", train)
    write_list(out_dir / "val.txt", val)
    write_list(out_dir / "test.txt", test)
    write_list(out_dir / "train_pos_only.txt", p_train)
    write_list(out_dir / "val_pos_only.txt", p_val)
    write_list(out_dir / "val_neg_only.txt", n_val)

    print("=" * 90)
    print("Bijie list generation finished.")
    print(f"landslide images: {len(land_images)}")
    print(f"valid positive items: {len(pos_items)}")
    print(f"non-landslide images: {len(neg_items)}")
    print("-" * 90)
    print(f"missing positive masks: {missing_pos_mask}")
    print(f"empty masks skipped: {empty_masks}")
    print(f"missing positive topo: {missing_pos_topo}")
    print(f"missing negative topo: {missing_neg_topo}")
    print("-" * 90)
    print(f"train: {len(train)} | pos={len(p_train)} neg={len(n_train)}")
    print(f"val:   {len(val)} | pos={len(p_val)} neg={len(n_val)}")
    print(f"test:  {len(test)} | pos={len(p_test)} neg={len(n_test)}")
    print(f"saved to: {out_dir}")
    print("=" * 90)
    if missing_pos_topo > 0 or missing_neg_topo > 0:
        print("WARNING: Some samples have no topo path. Lines with third column 'none' will not use real DEM.")


if __name__ == "__main__":
    main()
