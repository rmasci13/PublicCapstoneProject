"""Standalone sanity check — NOT part of the training pipeline.
Verifies that image -> diagnosis -> target integer -> prediction stays
correctly aligned through the whole pipeline. Run this once whenever you
change data_prep.py, dataset.py, or config.py's LABEL_COLS, to catch any
relabeling bugs before they silently corrupt a training run.
"""

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from config import LABEL_COLS, PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR

RAW_GROUND_TRUTH_CSV = "../data/raw/train/ISIC_2019_Training_GroundTruth.csv"  # adjust to your path


def verify_target_matches_raw_onehot(n=10, random_state=42):
    """Independently re-derives each sampled row's label straight from the
    RAW one-hot ground truth CSV (bypassing data_prep.py entirely) and
    checks it matches what's stored in df_merged's 'diagnosis' and 'target'
    columns. If this ever fails, the bug is in data_prep.py's label
    encoding, not in training or the model."""
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    df_raw = pd.read_csv(RAW_GROUND_TRUTH_CSV)

    sample = df_merged.sample(n=n, random_state=random_state)

    print(f"Checking {n} random rows against the RAW ground truth CSV directly...\n")
    all_ok = True
    for _, row in sample.iterrows():
        raw_row = df_raw[df_raw["image"] == row["image"]].iloc[0]

        # Which column has the 1.0 in the RAW file, independent of any of our code
        raw_diagnosis = raw_row[LABEL_COLS].astype(float).idxmax()
        raw_target = LABEL_COLS.index(raw_diagnosis)

        match = (raw_diagnosis == row["diagnosis"]) and (raw_target == row["target"])
        status = "OK" if match else "MISMATCH!!"
        if not match:
            all_ok = False

        print(
            f"{row['image']:>16} | df_merged: diagnosis={row['diagnosis']:<5} target={row['target']} | "
            f"raw CSV: diagnosis={raw_diagnosis:<5} target={raw_target} | {status}"
        )

    print("\n" + ("All rows matched — labels are correctly encoded." if all_ok
                   else "MISMATCHES FOUND — check data_prep.py's label encoding logic."))
    return all_ok


def visual_spot_check(n=6, random_state=None):
    """Displays n random images from df_merged alongside their recorded
    diagnosis, so you can visually confirm the label attached to a given
    image ID actually corresponds to that image file (catches path
    mix-ups, off-by-one merges, etc. — not clinical correctness, which
    isn't something to verify by eye anyway)."""
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    sample = df_merged.sample(n=n, random_state=random_state)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img_path = f"{LOCAL_IMAGE_DIR}/{row['image']}.jpg"
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{row['image']}\ndiagnosis={row['diagnosis']} (target={row['target']})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def verify_predictions_aligned(model, val_df, image_dir, transform, class_names, device, n=6):
    """Runs the model on n specific images (not a shuffled DataLoader) so
    the image ID, true label, and prediction are guaranteed to correspond
    to the same row — no risk of DataLoader shuffling/batching order
    confusion. Displays each image with its true vs predicted label so you
    can eyeball whether predictions look at least plausible, and confirm
    nothing is silently offset."""
    import torch

    model.eval()
    sample = val_df.sample(n=n, random_state=random.randint(0, 10_000))

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img_path = f"{image_dir}/{row['image']}.jpg"
        img = Image.open(img_path).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(1).item()

        true_label = row["diagnosis"]
        pred_label = class_names[pred_idx]
        correct = "✓" if pred_label == true_label else "✗"

        ax.imshow(img)
        ax.set_title(f"{row['image']}\ntrue={true_label} pred={pred_label} {correct}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def verify_all_image_files_exist(image_dir=LOCAL_IMAGE_DIR):
    """Checks that every image ID in df_merged has a corresponding .jpg
    file on disk. Cheap (existence check only, no image loading) — worth
    running once whenever you change LOCAL_IMAGE_DIR or re-extract the
    image archive, so a missing file surfaces now instead of crashing
    training partway through a long run."""
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)

    missing = [
        img_id for img_id in df_merged["image"]
        if not os.path.exists(os.path.join(image_dir, f"{img_id}.jpg"))
    ]

    print(f"Checked {len(df_merged)} rows against {image_dir}")
    print(f"Missing files: {len(missing)}")
    if missing:
        print("First 20 missing image IDs:", missing[:20])
    else:
        print("All image files present — safe to train without hitting a FileNotFoundError mid-run.")

    return missing


if __name__ == "__main__":
    # Step 1: confirm df_merged's labels match the raw CSV independently
    verify_target_matches_raw_onehot(n=15)

    # Step 2: confirm every image ID actually has a file on disk
    verify_all_image_files_exist()

    # Step 3: eyeball that image files actually match their recorded diagnosis
    visual_spot_check(n=6)

    # Step 4 (optional, requires a trained model in memory):
    # verify_predictions_aligned(model, val_df, LOCAL_IMAGE_DIR, eval_transform,
    #                             class_names=train_dataset.classes, device=DEVICE, n=6)