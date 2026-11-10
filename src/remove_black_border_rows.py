from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm  # pip install tqdm
from config import LOCAL_IMAGE_DIR


def has_black_borders(img_name, threshold=75):
    try:
        img_path = Path(LOCAL_IMAGE_DIR) / f"{img_name}.jpg"
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        h, w = gray.shape
        corners = [
            gray[:h // 12, :w // 12],
            gray[:h // 12, -w // 12:],
            gray[-h // 12:, :w // 12],
            gray[-h // 12:, -w // 12:]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])
        return corner_mean < threshold
    except Exception:
        return False


if __name__ == "__main__":
    df = pd.read_csv('../data/processed/df_merged.csv')

    print("Checking images for dark/black dermatoscope borders...")

    # Enable pandas progress_apply integration
    tqdm.pandas(desc="Processing images")

    # Use progress_apply instead of apply
    df["has_border"] = df["image"].progress_apply(has_black_borders)

    filtered = df.copy()

    filtered = filtered[filtered["has_border"] == False]

    print(f"Total images: {len(df)}")
    print(f"Total images without border: {len(filtered)}")

    filtered.to_csv('../data/processed/df_merged_with_border_info2.csv', index=False)
    print("Done! Saved to df_merged_with_border_info.csv")