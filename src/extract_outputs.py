"""Runs your trained CNN and MLP over the images/metadata ONCE and saves, for every image:
  - each model's FEATURES (the numbers just before its final benign/malignant layer)
  - each model's LOGITS   (its final scores)
so fusion experiments can reuse them without re-running images through the CNN.

You normally don't run this directly: run_fusion.py calls it automatically when the feature
files are missing or were made from different runs. To run it on its own, set the two run
paths below (the run folder, or the model_weights.pt inside it; config.json must sit next
to the weights) and run:  python extract_outputs.py

Model names, image size and layer sizes are read from each run's config.json.
Images go through the eval transform (no augmentation), so the output is deterministic.
Needs fusion.py (for FusionDataset) in the same folder.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import LOCAL_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE, DEVICE, set_seed
from dataset import get_transforms
from fusion import FusionDataset
from models import get_model

# ---------------------------------------------------------------------------
# Settings — only used when you run this file directly
# ---------------------------------------------------------------------------
CNN_RUN_DIR = Path("results/2026-10-29_0124_M-EfficientNetB2_Tune_binary_S-size-25331_E-50")   # folder (or .pt file) of the trained image model
MLP_RUN_DIR = Path("results/2026-10-23_0038_M-ComplexMLP_binary_S-size-25331_E-50")   # folder (or .pt file) of the trained metadata model
TRAIN_METADATA_CSV = "../data/processed/train_metadata.csv"   # the pinned split
VAL_METADATA_CSV = "../data/processed/val_metadata.csv"
SPLITS = ["val", "train"]
OUT_DIR = Path("fusion_features")   # one features_<split>.npz is written here per split


def resolve_run_dir(path):
    path = Path(path)
    return path.parent if path.suffix == ".pt" else path


def load_trained(path):
    """Loads a saved run and returns (model, feature_store, config, run_dir, mlp_input_dim).
    A forward hook on the model's last Linear layer captures that layer's INPUT (the features)
    every time the model runs, so one forward pass gives both features and logits."""
    run_dir = resolve_run_dir(path)
    cfg = json.load(open(run_dir / "config.json"))
    state = torch.load(run_dir / "model_weights.pt", map_location="cpu")

    linear_weights = [v for k, v in state.items() if k.endswith("weight") and v.ndim == 2]
    num_classes = linear_weights[-1].shape[0]
    kwargs, input_dim = {}, None
    if cfg.get("use_metadata_only", False):
        input_dim = int(linear_weights[0].shape[1])
        kwargs["input_dim"] = input_dim

    model = get_model(cfg["model"], num_classes=num_classes, **kwargs)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    store = {}
    last_linear.register_forward_hook(
        lambda module, inputs, output: store.__setitem__("features", torch.flatten(inputs[0], 1)))
    return model, store, cfg, run_dir, input_dim


@torch.no_grad()
def extract(df, cnn, cnn_store, mlp, mlp_store, mlp_input_dim, image_size):
    _, eval_transform = get_transforms(image_size)
    ds = FusionDataset(df, LOCAL_IMAGE_DIR, transform=eval_transform)
    if mlp_input_dim is not None and ds.features.shape[1] != mlp_input_dim:
        raise ValueError(f"The MLP expects {mlp_input_dim} metadata features but the metadata CSV gives "
                         f"{ds.features.shape[1]}. Use the same metadata CSVs the MLP was trained on.")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    out = {k: [] for k in ["cnn_features", "cnn_logits", "mlp_features", "mlp_logits"]}
    for images, meta, _, _ in loader:
        out["cnn_logits"].append(cnn(images.to(DEVICE)).float().cpu().numpy())
        out["cnn_features"].append(cnn_store["features"].float().cpu().numpy())
        out["mlp_logits"].append(mlp(meta.to(DEVICE)).float().cpu().numpy())
        out["mlp_features"].append(mlp_store["features"].float().cpu().numpy())
    return ds, {k: np.concatenate(v) for k, v in out.items()}


def sanity_report(split, labels, arrays, name, run_dir):
    """Accuracy from the saved logits. On val this should match the accuracy in the run's own
    classification_report.csv — if not, the weights, transform or split are not what the run used."""
    logits = arrays[f"{name}_logits"]
    if logits.shape[1] != 2:
        return
    pred = logits.argmax(1)
    msg = (f"  {name.upper()} {split}: accuracy={(pred == labels).mean():.4f} "
           f"malignant_recall={(pred[labels == 1] == 1).mean():.4f}")
    report_path = run_dir / "classification_report.csv"
    if split == "val" and report_path.exists():
        saved = float(pd.read_csv(report_path, index_col=0).loc["accuracy", "precision"])
        msg += f" | run's saved accuracy={saved:.4f}"
        if abs(saved - (pred == labels).mean()) > 0.003:
            msg += ("\n  WARNING: this does not match the run's saved accuracy. Check the run folder, that "
                    "dataset.get_transforms() matches how this model was trained, and that the metadata CSVs "
                    "are the split it was validated on.")
    print(msg)


def save_split(split, ds, arrays, info, out_dir):
    df = ds.df
    needed = ["age_is_missing", "sex_unknown", "anatom_site_general_unknown"]
    complete = (df[needed].sum(axis=1) == 0).values if all(c in df.columns for c in needed) \
        else np.ones(len(df), dtype=bool)
    path = Path(out_dir) / f"features_{split}.npz"
    np.savez_compressed(
        path,
        image=df["image"].astype(str).values,
        diagnosis=df["diagnosis"].astype(str).values,
        lesion_id_filled=df["lesion_id_filled"].astype(str).values if "lesion_id_filled" in df.columns
        else df["image"].astype(str).values,
        label=df["BinaryTarget"].values.astype(int),
        complete_metadata=complete.astype(np.int8),
        info_json=np.array(json.dumps(info)),
        **arrays,
    )
    print(f"  saved {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def run_extraction(cnn_run, mlp_run, splits, out_dir, train_csv, val_csv):
    """Extracts features/logits for the given splits and writes features_<split>.npz into out_dir."""
    set_seed()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cnn, cnn_store, cnn_cfg, cnn_dir, _ = load_trained(cnn_run)
    mlp, mlp_store, mlp_cfg, mlp_dir, mlp_input_dim = load_trained(mlp_run)
    if cnn_cfg.get("use_metadata_only", False):
        raise ValueError(f"{cnn_dir} is a metadata-only run; the CNN path needs an image model.")
    if not mlp_cfg.get("use_metadata_only", False):
        raise ValueError(f"{mlp_dir} is an image run; the MLP path needs a metadata-only model.")
    if not cnn_cfg.get("transform_has_CropBlackBorder", False):
        print("WARNING: this CNN run's config doesn't say it was trained with CropBlackBorder, but "
              "dataset.get_transforms() applies it. If the run was trained without it, features will be mismatched.")

    image_size = int(cnn_cfg.get("image_size", IMAGE_SIZE))
    info = {"cnn_run": str(cnn_dir.resolve()), "cnn_model": cnn_cfg["model"], "mlp_run": str(mlp_dir.resolve()),
            "mlp_model": mlp_cfg["model"], "image_size": image_size}
    csvs = {"train": train_csv, "val": val_csv}

    for split in splits:
        print(f"\nExtracting {split} ...")
        df = pd.read_csv(csvs[split])
        ds, arrays = extract(df, cnn, cnn_store, mlp, mlp_store, mlp_input_dim, image_size)
        labels = ds.df["BinaryTarget"].values.astype(int)
        print(f"  {len(ds)} images | CNN features: {arrays['cnn_features'].shape[1]} per image | "
              f"MLP features: {arrays['mlp_features'].shape[1]} per image")
        sanity_report(split, labels, arrays, "cnn", cnn_dir)
        sanity_report(split, labels, arrays, "mlp", mlp_dir)
        save_split(split, ds, arrays, {**info, "split": split}, out_dir)


if __name__ == "__main__":  # required on Windows because of DataLoader workers
    run_extraction(CNN_RUN_DIR, MLP_RUN_DIR, SPLITS, OUT_DIR, TRAIN_METADATA_CSV, VAL_METADATA_CSV)
    print("\nDone.")
