"""Flags images that contain a sticker / colour marker and saves a new CSV with the flags added.

Reads df_merged.csv (left untouched), looks at every image once, and writes sticker_check.csv containing all the
original columns plus:
    sticker_blob_frac   size of the largest sticker-coloured blob as a share of the image (0 to 1)
    sticker_pixel_frac  share of the image made of sticker-coloured pixels in total
    has_sticker         1 if sticker_blob_frac >= THRESHOLD
    is_downsampled      1 if the image name ends in "_downsampled" (a possible source marker worth checking too)

It then prints how the flag lines up with diagnosis / malignancy / id ranges, and saves two contact sheets of
thumbnails so you can SEE what was flagged and what was a near miss, and tune THRESHOLD if needed.

How a sticker is detected: the image is shrunk to ~256 px and converted to HSV. Pixels count as "sticker-coloured"
if they are very saturated and bright (skin and most lesions aren't) or have a green / blue / purple / pink hue
(not skin tones). The largest connected group of such pixels is measured. It's a colour rule, so check the
contact sheets: beige/skin-toned stickers are missed and a vividly coloured lesion could be flagged.

Run:  python find_stickers.py        (edit the settings below first)
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
INPUT_CSV = "../data/processed/df_merged.csv"
OUTPUT_CSV = "../data/processed/sticker_check.csv"
THRESHOLD = 0.015            # flag when the largest sticker-coloured blob covers >= this share of the image
MAX_SIDE = 256               # images are shrunk to this size before analysis (speed)
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)
SHEET_DIR = "."              # where the two contact sheets are saved
SHEET_COUNT = 24             # thumbnails per contact sheet
SEED = 42


def sticker_scores(path):
    """Returns (largest_blob_fraction, total_pixel_fraction) for one image, or (nan, nan) if unreadable."""
    try:
        img = Image.open(path)
        img.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))   # lets JPEG decode at reduced size: much faster
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        hsv = np.asarray(img.convert("HSV")).astype(float) / 255.0
    except Exception:
        return np.nan, np.nan
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    # very saturated AND very bright (brown/orange lesions are saturated but darker), or non-skin hues
    mask = ((s > 0.70) & (v > 0.80)) | ((s > 0.35) & (v > 0.25) & (h > 75) & (h < 330))
    labels, n = ndi.label(mask)
    biggest = np.bincount(labels.ravel())[1:].max() / mask.size if n else 0.0
    return float(biggest), float(mask.mean())


def _score_one(args):
    image_dir, name = args
    return sticker_scores(Path(image_dir) / f"{name}.jpg")


def contact_sheet(names, scores, image_dir, path, cols=6, thumb=200):
    rows = int(np.ceil(len(names) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, score) in enumerate(zip(names, scores)):
        try:
            im = Image.open(Path(image_dir) / f"{name}.jpg")
            im.draft("RGB", (thumb * 2, thumb * 2))
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), "gray")
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + thumb, y + 12], fill="black")
        draw.text((x + 2, y), f"{name[5:]} {score:.3f}", fill="white")
    sheet.save(path, quality=90)
    print(f"Saved {path}")


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"{len(df)} images to check with {NUM_WORKERS} worker(s)...")
    jobs = [(IMAGE_DIR, n) for n in df["image"].astype(str)]
    out = []
    with ProcessPoolExecutor(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.map(_score_one, jobs, chunksize=64), 1):
            out.append(res)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    scores = np.array(out, dtype=float)

    result = df.copy()
    result["sticker_blob_frac"] = scores[:, 0]
    result["sticker_pixel_frac"] = scores[:, 1]
    result["has_sticker"] = (result["sticker_blob_frac"] >= THRESHOLD).astype(int)
    result["is_downsampled"] = result["image"].astype(str).str.contains("_downsampled").astype(int)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")

    unreadable = int(result["sticker_blob_frac"].isna().sum())
    if unreadable:
        print(f"WARNING: {unreadable} image(s) could not be read (check IMAGE_DIR); they are marked has_sticker = 0.")

    n_flag = int(result["has_sticker"].sum())
    print(f"\nFlagged {n_flag} of {len(result)} images ({n_flag / len(result):.2%}) at threshold {THRESHOLD}")

    print("\nFlagged images by diagnosis:")
    print(pd.crosstab(result["diagnosis"], result["has_sticker"]).rename(columns={0: "no sticker", 1: "sticker"}))
    print("\nMalignant share (BinaryTarget) by sticker flag:")
    print(result.groupby("has_sticker")["BinaryTarget"].agg(["count", "mean"]).round(3))
    print("\nMalignant share by '_downsampled' in the name:")
    print(result.groupby("is_downsampled")["BinaryTarget"].agg(["count", "mean"]).round(3))
    print("\nSticker share by '_downsampled':")
    print(result.groupby("is_downsampled")["has_sticker"].agg(["count", "mean"]).round(4))

    # Are the flagged images clustered in image-id ranges (a sign they come from one source)?
    ids = pd.to_numeric(result["image"].astype(str).str.extract(r"ISIC_(\d+)")[0], errors="coerce")
    if ids.notna().any():
        bins = (ids // 1000 * 1000).astype("Int64")
        per_bin = result.assign(id_bin=bins).groupby("id_bin")["has_sticker"].agg(["sum", "count"])
        top = per_bin[per_bin["sum"] > 0].sort_values("sum", ascending=False).head(8)
        top["share"] = (top["sum"] / top["count"]).round(3)
        print("\nImage-id ranges (blocks of 1000) with the most flagged images:")
        print(top.rename(columns={"sum": "flagged", "count": "images_in_block"}).to_string())

    # Contact sheets to check the rule by eye
    rng = np.random.default_rng(SEED)
    flagged = result[result["has_sticker"] == 1]
    if len(flagged):
        pick = flagged.iloc[rng.choice(len(flagged), min(SHEET_COUNT, len(flagged)), replace=False)]
        contact_sheet(pick["image"].tolist(), pick["sticker_blob_frac"].tolist(), IMAGE_DIR,
                      Path(SHEET_DIR) / "sticker_sheet_flagged.jpg")
    near = result[(result["has_sticker"] == 0) & (result["sticker_blob_frac"] >= THRESHOLD * 0.3)]
    if len(near):
        pick = near.sort_values("sticker_blob_frac", ascending=False).head(SHEET_COUNT)
        contact_sheet(pick["image"].tolist(), pick["sticker_blob_frac"].tolist(), IMAGE_DIR,
                      Path(SHEET_DIR) / "sticker_sheet_near_misses.jpg")
    print("\nLook at the two sheets: flagged images should show stickers/markers; near misses are the "
          "highest-scoring images that were NOT flagged (lower THRESHOLD if real stickers appear there).")


if __name__ == "__main__":   # required on Windows because of worker processes
    import sys
    if len(sys.argv) > 1:     # quick check of single images:  python find_stickers.py ISIC_0000017 path/to/other.jpg
        for arg in sys.argv[1:]:
            p = Path(arg) if Path(arg).suffix else Path(IMAGE_DIR) / f"{arg}.jpg"
            blob, pix = sticker_scores(p)
            print(f"{p}: largest blob {blob:.4f}, total {pix:.4f} -> "
                  f"{'STICKER' if blob >= THRESHOLD else 'no sticker'} (threshold {THRESHOLD})")
    else:
        main()