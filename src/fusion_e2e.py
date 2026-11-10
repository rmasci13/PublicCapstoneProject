"""Standalone end-to-end fusion: your trained CNN (images) + your trained metadata MLP, joined by a small
classification head and trained together to predict benign / malignant.

Run:  python fusion_e2e.py      (edit the SETTINGS block below first; nothing else needs touching)

How it works
  - One row of train_metadata.csv / val_metadata.csv gives BOTH inputs for an image: the image file (via its
    `image` id) goes to the CNN, the metadata columns go to the MLP, so the two can never get out of step.
  - Each encoder is your own model from models.py with its final classification layer removed, so it outputs
    its feature vector (CNN 1408 numbers for EfficientNetB2, MLP 128 for ComplexMLP). Your trained weights
    are loaded first.
  - FusionModel normalises each feature vector, joins them, and a small head predicts the two classes.
  - Training: the encoders stay frozen for FREEZE_EPOCHS so only the new head learns, then everything
    fine-tunes (encoders at much smaller learning rates). Class-weighted, label-smoothed cross-entropy like
    train.py; the best epoch by MONITOR is restored at the end.
  - At the end it evaluates the val set and writes everything to results/<timestamp>_FUSION-E2E_.../ :
    confusion matrix (csv + png), classification report with F2, per-diagnosis table, AUC and recall at fixed
    specificity, cutoff table, per_image_predictions_val.csv (same columns multiclass_analysis.py reads),
    loss curve, history, config.json and model_weights.pt.

It reuses models.py (get_model), dataset.py (get_transforms, MetadataLesionDataset) and config.py exactly as
your other scripts do, so images are processed the same way the CNN was trained.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from config import HAS_CROP
from dataset import MetadataLesionDataset, get_transforms
from models import get_model

# ============================================================================
#                               SETTINGS  (edit these)
# ============================================================================
# --- encoders: which model, and where its trained weights are -------------------------------------
CNN_MODEL = "EfficientNetB2_Tune"                    # name in models.py
CNN_WEIGHTS = None      # run folder or .pt file; None = start from the ImageNet weights
MLP_MODEL = "ComplexMLP"                             # name in models.py
MLP_WEIGHTS = None      # run folder or .pt file; None = random init

# --- data ---------------------------------------------------------------------------------------------
IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
TRAIN_METADATA_CSV = "../data/processed/train_metadata.csv"
VAL_METADATA_CSV = "../data/processed/val_metadata.csv"
IMAGE_SIZE = 350                                     # must match the CNN's training size
NUM_WORKERS = 4

# --- head ----------------------------------------------------------------------------------------------
HEAD_HIDDEN = 256                                    # hidden units in the fusion head (0 = a single linear layer)
HEAD_DROPOUT = 0.4

# --- training ------------------------------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 32
FREEZE_EPOCHS = 0                                    # epochs with both encoders frozen (0 = fine-tune from the start)
LR_HEAD = 5e-5                                       # new head + the two feature-normalisation layers
LR_CNN = 1e-5                                        # CNN once unfrozen
LR_MLP = 7.5e-5                                        # MLP once unfrozen
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True
WEIGHT_POWER = 2                                     # balanced weights ** this (same idea as POWER in config.py)
USE_SCHEDULER = True                                 # ReduceLROnPlateau on val loss, like train.py
EARLY_STOPPING_PATIENCE = 7                          # None = always run all EPOCHS
MONITOR = "val_loss"                                 # "val_loss" or "val_macro_recall": what picks the best epoch
SEED = 42

# --- reporting -----------------------------------------------------------------------------------------
CUTOFF = 0.5                                         # P(malignant) cutoff for the confusion matrix / report
SPEC_LEVELS = (0.80, 0.85, 0.90)                     # "malignant recall at this benign recall"
RESULTS_ROOT = "results"

# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINARY_CLASSES = ["Benign", "Malignant/Pre-Malignant"]       # class 0, class 1
REPORT_ORDER = ["Malignant/Pre-Malignant", "Benign"]         # display order, same as metrics.py
RISK_GROUPED_ORDER = ["MEL", "BCC", "AK", "SCC", "NV", "BKL", "DF", "VASC"]


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
#                                  DATASET
# ============================================================================
class FusionDataset(Dataset):
    """Returns (image, metadata_vector, label, diagnosis) for one CSV row.
    Metadata columns are chosen by MetadataLesionDataset, i.e. exactly as when the MLP was trained."""

    def __init__(self, df, image_dir, transform):
        meta = MetadataLesionDataset(df, classification_mode="binary")
        self.df = meta.df
        self.features = meta.features            # (N, n_features) float tensor
        self.labels = meta.labels                # (N,) long tensor (BinaryTarget)
        self.feature_cols = meta.feature_cols
        self.images = self.df["image"].astype(str).tolist()
        self.diagnoses = self.df["diagnosis"].astype(str).tolist()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, f"{self.images[idx]}.jpg")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.features[idx], self.labels[idx], self.diagnoses[idx]


# ============================================================================
#                                   MODEL
# ============================================================================
def resolve_weights(path):
    """Accepts a run folder or a .pt file; returns (path_to_pt, run_dir)."""
    path = Path(path)
    if path.suffix == ".pt":
        return path, path.parent
    return path / "model_weights.pt", path


def strip_head(model):
    """Replaces the model's last Linear layer with Identity so the model outputs its feature vector."""
    name, _ = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, nn.Identity())


def load_encoder(model_name, weights, example_input, expected_input_dim=None):
    """Builds a models.py model, loads its trained weights, strips its final layer.
    Returns (encoder, feature_dim)."""
    kwargs = {"input_dim": expected_input_dim} if expected_input_dim is not None else {}
    model = get_model(model_name, num_classes=2, **kwargs)
    if weights:
        pt, run_dir = resolve_weights(weights)
        if not pt.exists():
            raise FileNotFoundError(f"No weights at {pt}. Set the *_WEIGHTS path at the top of fusion_e2e.py.")
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            if cfg.get("model") not in (None, model_name):
                raise ValueError(f"{run_dir} was trained as {cfg['model']}, but the setting says {model_name}.")
        model.load_state_dict(torch.load(pt, map_location="cpu"))   # fails loudly on any size mismatch
        print(f"  loaded {model_name} weights from {pt}")
    else:
        print(f"  {model_name}: no weights given, using its default initialisation")
    strip_head(model)
    model.eval()
    with torch.no_grad():
        dim = int(model(example_input).shape[1])
    return model, dim


class FusionModel(nn.Module):
    """image -> image_encoder -> features ┐
       metadata -> meta_encoder -> features ┴-> normalise each, join -> head -> 2 class scores"""

    def __init__(self, image_encoder, image_dim, meta_encoder, meta_dim,
                 hidden=HEAD_HIDDEN, dropout=HEAD_DROPOUT, num_classes=2):
        super().__init__()
        self.image_encoder = image_encoder
        self.meta_encoder = meta_encoder
        self.image_norm = nn.LayerNorm(image_dim)
        self.meta_norm = nn.LayerNorm(meta_dim)
        joined = image_dim + meta_dim
        if hidden:
            self.head = nn.Sequential(nn.Linear(joined, hidden), nn.SiLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, num_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(joined, num_classes))
        self.encoders_frozen = False

    def forward(self, image, meta):
        img_f = self.image_norm(self.image_encoder(image))
        meta_f = self.meta_norm(self.meta_encoder(meta))
        return self.head(torch.cat([img_f, meta_f], dim=1))

    def set_encoders_frozen(self, frozen):
        self.encoders_frozen = frozen
        for enc in (self.image_encoder, self.meta_encoder):
            for p in enc.parameters():
                p.requires_grad = not frozen
        self.train(self.training)   # re-apply train/eval mode to the encoders

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:     # frozen encoders stay in eval mode (no dropout, BatchNorm stats fixed)
            self.image_encoder.eval()
            self.meta_encoder.eval()
        return self


# ============================================================================
#                              TRAIN / EVAL LOOPS
# ============================================================================
def recall_per_class(labels, preds):
    return np.array([(preds[labels == c] == c).mean() if (labels == c).any() else np.nan for c in (0, 1)])


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    probs, preds, labels, diagnoses = [], [], [], []
    for images, meta, y, dx in loader:
        images, meta, y = images.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)
        logits = model(images, meta)
        loss_sum += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
        diagnoses.extend(dx)
    labels, preds = np.concatenate(labels), np.concatenate(preds)
    recalls = recall_per_class(labels, preds)
    return {"loss": loss_sum / n, "acc": float((preds == labels).mean()), "macro_recall": float(np.nanmean(recalls)),
            "malignant_recall": float(recalls[1]), "probs": np.concatenate(probs), "preds": preds,
            "labels": labels, "diagnoses": diagnoses}


def make_criterion(train_labels):
    weight = None
    if USE_CLASS_WEIGHTS:
        counts = np.bincount(train_labels, minlength=2).astype(float)
        w = (len(train_labels) / (2 * counts)) ** WEIGHT_POWER
        print(f"Class weights (benign, malignant): {np.round(w, 3).tolist()}")
        weight = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)


def train(model, train_loader, val_loader, criterion):
    if MONITOR not in ("val_loss", "val_macro_recall"):
        raise ValueError("MONITOR must be 'val_loss' or 'val_macro_recall'")
    model.to(DEVICE)
    new_params = (list(model.head.parameters()) + list(model.image_norm.parameters())
                  + list(model.meta_norm.parameters()))
    optimizer = torch.optim.AdamW(
        [{"params": new_params, "lr": LR_HEAD},
         {"params": model.image_encoder.parameters(), "lr": LR_CNN},
         {"params": model.meta_encoder.parameters(), "lr": LR_MLP}], weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
                 if USE_SCHEDULER else None)

    if FREEZE_EPOCHS > 0:
        model.set_encoders_frozen(True)
        print(f"Encoders frozen for the first {FREEZE_EPOCHS} epoch(s); only the head learns.")

    def n_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable parameters now: CNN {n_trainable(model.image_encoder):,} | "
          f"MLP {n_trainable(model.meta_encoder):,} | head + norms {n_trainable(model.head) + n_trainable(model.image_norm) + n_trainable(model.meta_norm):,}")

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_macro_recall",
                               "val_malignant_recall", "lr_head", "epoch_time", "encoders_frozen"]}
    best_score, best_epoch, best_state, stale = float("inf"), None, None, 0

    for epoch in range(1, EPOCHS + 1):
        if model.encoders_frozen and epoch > FREEZE_EPOCHS:
            model.set_encoders_frozen(False)
            stale = 0   # don't let the frozen phase count against patience
            print(f"--- epoch {epoch}: encoders unfrozen, fine-tuning everything ---")
        model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        for images, meta, y, _ in train_loader:
            images, meta, y = images.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images, meta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        val = evaluate(model, val_loader, criterion)
        if scheduler is not None:
            scheduler.step(val["loss"])
        epoch_time = time.time() - t0

        score = val["loss"] if MONITOR == "val_loss" else -val["macro_recall"]
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        for k, v in [("train_loss", loss_sum / total), ("train_acc", correct / total), ("val_loss", val["loss"]),
                     ("val_acc", val["acc"]), ("val_macro_recall", val["macro_recall"]),
                     ("val_malignant_recall", val["malignant_recall"]), ("lr_head", optimizer.param_groups[0]["lr"]),
                     ("epoch_time", epoch_time), ("encoders_frozen", int(model.encoders_frozen))]:
            history[k].append(v)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={loss_sum / total:.4f} train_acc={correct / total:.3f} | "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.3f} val_recall={val['macro_recall']:.3f} "
              f"mal_recall={val['malignant_recall']:.3f} | lr_head={optimizer.param_groups[0]['lr']:.2e} | "
              f"{epoch_time:.0f}s | best epoch {best_epoch} (monitor={MONITOR})")

        if EARLY_STOPPING_PATIENCE is not None and stale >= EARLY_STOPPING_PATIENCE and not model.encoders_frozen:
            print(f"Early stopping: {MONITOR} hasn't improved for {EARLY_STOPPING_PATIENCE} epochs "
                  f"(best epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    print(f"Restored weights from epoch {best_epoch} (monitor={MONITOR}).")
    history["meta"] = {"best_epoch": best_epoch, "epochs_run": len(history["train_loss"])}
    return history


# ============================================================================
#                                  REPORTING
# ============================================================================
def recall_at_spec(y, p, spec):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[np.searchsorted(fpr, 1 - spec, side="right") - 1])


def plot_per_diagnosis(per_dx, title, save_path):
    """Heatmap of how each true diagnosis was predicted (same look as multiclass_analysis.plot_summary):
    counts and share per row, a black line between the malignant and benign diagnoses."""
    counts = per_dx[["predicted_Malignant", "predicted_Benign"]].values
    fractions = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fractions, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nMalignant", "Predicted\nBenign"])
    ax.set_yticks(range(len(per_dx)))
    ax.set_yticklabels(per_dx["diagnosis"])
    ax.set_ylabel("True diagnosis")
    ax.set_title(title)
    for i in range(counts.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{counts[i, j]}\n({fractions[i, j]:.0%})", ha="center", va="center", fontsize=8)
    n_malignant = int((per_dx["true_group"] == REPORT_ORDER[0]).sum())
    ax.axhline(n_malignant - 0.5, color="black", linewidth=2)
    fig.colorbar(im, label="Share of that diagnosis")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(res, val_df, out_dir):
    y, p = res["labels"], res["probs"]
    preds = (p >= CUTOFF).astype(int)
    order_idx = [BINARY_CLASSES.index(n) for n in REPORT_ORDER]

    cm = confusion_matrix(y, preds, labels=order_idx)
    cm_df = pd.DataFrame(cm, index=REPORT_ORDER, columns=REPORT_ORDER)
    print(f"\nConfusion matrix (rows = true, columns = predicted, cutoff {CUTOFF:.2f}):\n{cm_df}")
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    rep = classification_report(y, preds, labels=order_idx, target_names=REPORT_ORDER, zero_division=0,
                                output_dict=True)
    f2 = fbeta_score(y, preds, labels=order_idx, beta=2, average=None, zero_division=0)
    for name, v in zip(REPORT_ORDER, f2):
        rep[name]["f2"] = float(v)
    rep["macro avg"]["f2"] = float(fbeta_score(y, preds, labels=order_idx, beta=2, average="macro", zero_division=0))
    print("\nClassification report:")
    print(classification_report(y, preds, labels=order_idx, target_names=REPORT_ORDER, zero_division=0))
    print("F2 (recall weighted 2x precision): " + ", ".join(f"{n}={rep[n]['f2']:.3f}" for n in REPORT_ORDER))
    pd.DataFrame(rep).T.to_csv(out_dir / "classification_report.csv")

    auc = float(roc_auc_score(y, p))
    spec_rows = {f"malignant_recall@benign_recall_{s:.2f}": recall_at_spec(y, p, s) for s in SPEC_LEVELS}
    print(f"\nAUC = {auc:.4f} | " + " | ".join(f"{k.split('@')[1]}: {v:.3f}" for k, v in spec_rows.items()))

    # per-diagnosis table (malignant dx: share flagged = recall; benign dx: share called benign)
    dx = np.array(res["diagnoses"])
    malignant_dx = set(val_df.loc[val_df["BinaryTarget"] == 1, "diagnosis"].astype(str))
    rows = []
    for d in RISK_GROUPED_ORDER:
        m = dx == d
        if not m.any():
            continue
        flagged = int((preds[m] == 1).sum())
        is_mal = d in malignant_dx
        rows.append({"diagnosis": d, "true_group": REPORT_ORDER[0] if is_mal else REPORT_ORDER[1], "n": int(m.sum()),
                     "predicted_Malignant": flagged, "predicted_Benign": int(m.sum()) - flagged,
                     "correct_rate": (flagged if is_mal else int(m.sum()) - flagged) / int(m.sum()),
                     "mean_prob_malignant": float(p[m].mean())})
    per_dx = pd.DataFrame(rows)
    per_dx.to_csv(out_dir / "per_diagnosis_val.csv", index=False)
    print("\nPer diagnosis (correct_rate = recall for malignant types, specificity for benign types):")
    print(per_dx.round(3).to_string(index=False))

    # cutoff table
    trows = []
    for t in np.round(np.arange(0.20, 0.81, 0.05), 2):
        pr = (p >= t).astype(int)
        tp, fn = int(((pr == 1) & (y == 1)).sum()), int(((pr == 0) & (y == 1)).sum())
        fp, tn = int(((pr == 1) & (y == 0)).sum()), int(((pr == 0) & (y == 0)).sum())
        rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        trows.append({"cutoff": float(t), "malignant_recall": rec, "malignant_precision": prec,
                      "benign_recall": tn / max(tn + fp, 1), "F2": 5 * prec * rec / max(4 * prec + rec, 1e-12),
                      "accuracy": (tp + tn) / len(y), "missed_malignant_FN": fn, "false_alarms_FP": fp})
    pd.DataFrame(trows).to_csv(out_dir / "threshold_overall.csv", index=False)

    # per-image predictions (same columns multiclass_analysis.py's helpers read)
    out = val_df[["image", "diagnosis"] + (["risk_level"] if "risk_level" in val_df.columns else [])
                 + ["BinaryTarget"]].copy()
    out["pred"], out["prob_malignant"] = preds, p
    out.to_csv(out_dir / "per_image_predictions_val.csv", index=False)

    title = out_dir.name
    fig, ax = plt.subplots(figsize=(8, 6))   # same look as metrics.save_run's confusion matrix
    im = ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(REPORT_ORDER); ax.set_yticklabels(REPORT_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_per_diagnosis(per_dx, f"Fusion {CNN_MODEL}+{MLP_MODEL} — binary predictions by true diagnosis (val)",
                       out_dir / "per_diagnosis_summary.png")

    return {"auc": auc, "accuracy": res["acc"], "macro_recall": res["macro_recall"],
            "malignant_recall": res["malignant_recall"], **spec_rows}


def plot_history(history, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(ep, history["train_loss"], marker="o", markersize=3, label="Train loss")
    a1.plot(ep, history["val_loss"], marker="o", markersize=3, label="Val loss")
    a2.plot(ep, history["val_acc"], marker="o", markersize=3, label="Val acc")
    a2.plot(ep, history["val_macro_recall"], marker="o", markersize=3, label="Val macro recall")
    a2.plot(ep, history["val_malignant_recall"], marker="o", markersize=3, label="Val malignant recall")
    frozen = sum(history["encoders_frozen"])
    for a in (a1, a2):
        if frozen:
            a.axvline(frozen + 0.5, color="gray", linestyle=":", label="encoders unfrozen")
        a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=0.3)
    a1.set_title("Loss"); a2.set_title("Validation accuracy / recall")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ============================================================================
#                                    MAIN
# ============================================================================
def main():
    set_seed()
    print(f"Device: {DEVICE} | CropBlackBorder in transforms: {HAS_CROP} (config.HAS_CROP)")
    train_df, val_df = pd.read_csv(TRAIN_METADATA_CSV), pd.read_csv(VAL_METADATA_CSV)
    if "lesion_id_filled" in train_df.columns:
        overlap = set(train_df["lesion_id_filled"]) & set(val_df["lesion_id_filled"])
        assert not overlap, f"Lesion leakage: {len(overlap)} lesions appear in both train and val"
    print(f"Train: {len(train_df)} images ({train_df['BinaryTarget'].mean():.1%} malignant) | "
          f"Val: {len(val_df)} ({val_df['BinaryTarget'].mean():.1%} malignant)")

    train_tf, eval_tf = get_transforms(IMAGE_SIZE)
    train_ds = FusionDataset(train_df, IMAGE_DIR, train_tf)
    val_ds = FusionDataset(val_df, IMAGE_DIR, eval_tf)
    n_meta = len(train_ds.feature_cols)
    print(f"Metadata features per image: {n_meta}")
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                       persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("Building encoders:")
    cnn, cnn_dim = load_encoder(CNN_MODEL, CNN_WEIGHTS, torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    mlp, mlp_dim = load_encoder(MLP_MODEL, MLP_WEIGHTS, torch.zeros(2, n_meta), expected_input_dim=n_meta)
    print(f"  CNN features: {cnn_dim} | MLP features: {mlp_dim}")
    model = FusionModel(cnn, cnn_dim, mlp, mlp_dim)

    criterion = make_criterion(train_df["BinaryTarget"].values.astype(int))
    history = train(model, train_loader, val_loader, criterion)
    res = evaluate(model, val_loader, criterion)

    run_name = re.sub(r'[:\\/*?"<>|]', "-", f"{datetime.now():%Y-%m-%d_%H%M}_FUSION-E2E_{CNN_MODEL}+{MLP_MODEL}")
    out_dir = Path(RESULTS_ROOT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_report(res, val_ds.df, out_dir)
    meta = history.pop("meta")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_history(history, out_dir / "loss_curve.png")
    torch.save(model.state_dict(), out_dir / "model_weights.pt")
    settings = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, float, bool, tuple, type(None)))}
    json.dump({**settings, **meta, **summary, "metadata_features": train_ds.feature_cols},
              open(out_dir / "config.json", "w"), indent=2, default=str)
    print(f"\nSaved everything to {out_dir}/")


if __name__ == "__main__":   # required on Windows because of DataLoader workers
    main()