"""One-time preprocessing: DullRazor-style hair removal.

Reads every image listed in df_merged.csv, removes dark hairs, and writes the
cleaned image to OUT_DIR under the SAME filename. Afterwards, point
LOCAL_IMAGE_DIR in config.py at OUT_DIR and nothing else in the pipeline
changes (dataset.py just looks up f"{image}.jpg").

Method (a common re-implementation of DullRazor, Lee et al. 1997):
  1. grayscale
  2. morphological BLACK-HAT with thin line-shaped kernels at several angles
     (black-hat = closing - original, so it lights up dark thin structures
     like hairs; the max over angles catches hairs running in any direction)
  3. threshold -> hair mask, slightly dilated to cover the hair's soft edges
  4. fill the masked pixels from their neighbors (the original paper uses
     bilinear interpolation; cv2.inpaint is the usual practical substitute)

Run from src/ like your other scripts:  python preprocess_hair.py

IMPORTANT: apply the same preprocessing to EVERY split (train, val, and any
test set you evaluate on later). Models trained on hair-free images should be
evaluated on hair-free images.
"""

import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
OUT_DIR = Path("../data/processed/images_hairfree_512")
WORK_SIZE = 512          # images are resized to WORK_SIZE x WORK_SIZE first (your
                         # pipeline already squashes to a square, so this matches it).
                         # Doing it first also makes hair thickness, in pixels,
                         # comparable across images that start at different sizes.
                         # NOTE: this caps the largest usable IMAGE_SIZE at 512.
KERNEL_LEN = 17          # line length in pixels; must exceed ~1.5x hair thickness
                         # but stay well below the size of real lesion structures
THRESH = 10              # black-hat response (0-255) above which a pixel is "hair"
ANGLES = (0, 45, 90, 135)
INPAINT_RADIUS = 3
OVERWRITE = False        # False = skip images that already have an output (resumable)
N_WORKERS = max(1, (os.cpu_count() or 2) - 2)
PREVIEW_N = 8            # random before/mask/after rows saved for visual QA


def line_kernel(length, angle_deg):
    """A 1-pixel-thick line structuring element of the given length/angle."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    p1 = (int(round(c - c * dx)), int(round(c - c * dy)))
    p2 = (int(round(c + c * dx)), int(round(c + c * dy)))
    cv2.line(k, p1, p2, 1, 1)
    return k


def detect_hair(img_bgr, kernel_len=KERNEL_LEN, thresh=THRESH, angles=ANGLES):
    """Returns a uint8 mask (255 = hair) the same height/width as the image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blackhat = np.zeros_like(gray)
    for angle in angles:
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line_kernel(kernel_len, angle))
        blackhat = np.maximum(blackhat, bh)
    _, mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    # Cover the anti-aliased edges of each hair so no dark fringe survives
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def remove_hair(img_bgr, **kwargs):
    """Returns (cleaned_bgr, mask)."""
    mask = detect_hair(img_bgr, **kwargs)
    cleaned = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return cleaned, mask


def load_resized(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    shrinking = (h * w) > (WORK_SIZE * WORK_SIZE)
    return cv2.resize(img, (WORK_SIZE, WORK_SIZE),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def process_one(image_id):
    """Worker: returns (image_id, status, fraction_of_pixels_masked)."""
    out_path = OUT_DIR / f"{image_id}.jpg"
    if out_path.exists() and not OVERWRITE:
        return image_id, "skipped", float("nan")
    img = load_resized(Path(LOCAL_IMAGE_DIR) / f"{image_id}.jpg")
    if img is None:
        return image_id, "unreadable", float("nan")
    cleaned, mask = remove_hair(img)
    cv2.imwrite(str(out_path), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return image_id, "done", float((mask > 0).mean())


def save_preview(image_ids, path):
    """Rows of [original | hair mask | cleaned] for eyeballing the settings."""
    rows = []
    for image_id in image_ids:
        img = load_resized(Path(LOCAL_IMAGE_DIR) / f"{image_id}.jpg")
        if img is None:
            continue
        cleaned, mask = remove_hair(img)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        rows.append(cv2.hconcat([img, mask_bgr, cleaned]))
    if rows:
        grid = cv2.vconcat(rows)
        grid = cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), grid)


if __name__ == "__main__":  # required on Windows for multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_ids = pd.read_csv(PROCESSED_DATA_PATH)["image"].tolist()

    # 1) Preview first, so you can check the settings before the long run
    preview_path = OUT_DIR.parent / "hair_removal_preview.png"
    save_preview(random.Random(0).sample(image_ids, PREVIEW_N), preview_path)
    print(f"Saved preview (original | mask | cleaned): {preview_path}")
    print("Look at it before letting the full run continue; Ctrl+C now if the masks look wrong.\n")

    # 2) Full run, parallel across worker processes
    results = []
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, image_ids, chunksize=32), 1):
            results.append(res)
            if i % 1000 == 0 or i == len(image_ids):
                print(f"  {i}/{len(image_ids)} images processed")

    stats = pd.DataFrame(results, columns=["image", "status", "masked_fraction"])
    stats.to_csv(OUT_DIR.parent / "hair_stats.csv", index=False)
    print("\nStatus counts:\n", stats["status"].value_counts().to_string())
    done = stats[stats["status"] == "done"]["masked_fraction"]
    if len(done):
        print(f"\nMasked-pixel fraction per image: median {done.median():.3%}, "
              f"95th pct {done.quantile(.95):.3%}, max {done.max():.3%}")
        print("Images with >5% of pixels masked (worth eyeballing — possibly over-aggressive):",
              int((done > 0.05).sum()))
    print(f"\nNext: set LOCAL_IMAGE_DIR = \"{OUT_DIR.as_posix()}\" in config.py")