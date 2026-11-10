import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import (PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE,
                    MODEL_NAME, DEVICE, CLASSIFICATION_MODE, set_seed)
from dataset import LesionDataset, train_val_split, get_transforms
from models import get_model


def main():
    weights_path = sys.argv[1]

    set_seed()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    stratify_col = "diagnosis" if CLASSIFICATION_MODE == "multiclass" else "risk_level"
    train_df, val_df = train_val_split(df, stratify_col=stratify_col)
    _, eval_transform = get_transforms(IMAGE_SIZE)
    val_ds = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=eval_transform,
                           classification_mode=CLASSIFICATION_MODE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_model(MODEL_NAME, num_classes=len(val_ds.classes))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    z0s, z1s, ys = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch[0].to(DEVICE)).cpu()
            z0s.append(out[:, 0]); z1s.append(out[:, 1]); ys.append(batch[1])
    z0 = torch.cat(z0s).numpy(); z1 = torch.cat(z1s).numpy(); y = torch.cat(ys).numpy()

    print("n:", len(y), "| positives:", int(y.sum()))
    print("AUC using z1 - z0 (what you report now):", round(roc_auc_score(y, z1 - z0), 4))
    print("AUC using z1 only (what the loss trained):", round(roc_auc_score(y, z1), 4))
    print("std z0: %.3f  std z1: %.3f  corr(z0,z1): %.3f" % (z0.std(), z1.std(), np.corrcoef(z0, z1)[0, 1]))


if __name__ == "__main__":
    main()