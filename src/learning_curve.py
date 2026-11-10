"""Learning curve by training-set size (NOT the loss-vs-epoch curve).

Question it answers: if you had more (or less) labeled data, how would
validation performance change?  It trains the same model on growing slices of
the training pool and plots ROC AUC against training-set size.

Design rules (these are what make the curve trustworthy):
  1. ONE fixed validation set, identical to the one main.py uses (same split
     call, same default seed).  Only the TRAINING part is subsampled.
  2. Training subsets are NESTED (each larger subset contains the smaller one)
     and built from WHOLE lesion groups, so no lesion is ever split.
  3. The metric is ROC AUC, which does not depend on a decision cutoff.  Your
     class weights shift the default 0.50 cutoff as data size changes, so F1 or
     recall at 0.50 would drift for reasons unrelated to data volume.
  4. The training score is computed AFTER training, in eval mode, with the eval
     transform (no augmentation).  history's train_acc is not comparable.

Cost: roughly sum(FRACTIONS) x len(SEEDS) times one full-data training run.
Results are saved after every run, and RESUME_DIR lets you restart a crashed
or interrupted job without redoing finished runs.

Run from src/ like your other scripts:  python learning_curve_by_size.py

Assumes: binary mode, a standard 2-logit cross-entropy head (softmax over the two
outputs gives P(malignant)), and train() with the signature used in main.py.
"""

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # write PNGs only; no window needed
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR,
    USE_SCHEDULER, USE_CLASS_WEIGHTS, MODEL_NAME, CLASSIFICATION_MODE,
    EARLY_STOPPING_PATIENCE, DEVICE, set_seed,
)
from dataset import LesionDataset, train_val_split, get_transforms
from models import get_model
from train import train

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
FRACTIONS = [0.1, 0.25, 0.5, .75, 1.0]   # share of the training pool used; 1.0 = everything
SEEDS = [42]                        # >1 seed gives std-over-seeds bands (costs one extra pass per seed)
TRAIN_LR = LR                       # override here if this model uses a different LR
TRAIN_SCORE_MAX_IMAGES = 5000       # training AUC is measured on at most this many training images
N_BOOTSTRAP = 500                   # validation-lesion bootstrap resamples for the CI band
RESUME_DIR = None                   # e.g. Path("results/datasize_curve_EfficientNetB2_Tune_2026-10-12_1530")


# ---------------------------------------------------------------------------
# Pure helpers (no torch) — data subsets, bootstrap, summary, plot
# ---------------------------------------------------------------------------
def nested_group_subsets(train_df, fractions, seed=42, group_col="lesion_id_filled"):
    """Nested training subsets (each larger one contains the smaller) made of WHOLE lesion groups."""
    rng = np.random.default_rng(seed)
    groups = np.array(train_df[group_col].unique(), dtype=object)  # plain numpy: pandas string arrays don't shuffle reliably
    rng.shuffle(groups)
    cum = train_df.groupby(group_col).size().reindex(groups).cumsum().values
    out = {}
    for f in fractions:
        k = int(np.searchsorted(cum, f * len(train_df), side="left")) + 1
        out[f] = train_df[train_df[group_col].isin(set(groups[:k]))]
    return out


def bootstrap_auc_ci(y, p, group_ids, n_boot=N_BOOTSTRAP, seed=0):
    """95% CI for AUC from resampling whole LESION GROUPS of the validation set.
    (Photos of one lesion are correlated, so resampling single images would give a band that is too narrow.)
    This captures validation-set sampling noise only, not training-run randomness."""
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(np.asarray(group_ids))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(results):
    g = results.groupby("fraction")
    s = pd.DataFrame({
        "n_train": g["n_train"].mean(),
        "train_auc": g["train_auc"].mean(),
        "train_auc_std": g["train_auc"].std(),     # NaN when only one seed
        "val_auc": g["val_auc"].mean(),
        "val_auc_std": g["val_auc"].std(),         # NaN when only one seed
        "val_ci_low": g["val_ci_low"].mean(),
        "val_ci_high": g["val_ci_high"].mean(),
        "best_epoch": g["best_epoch"].mean(),
    }).sort_index()
    s["val_auc_gain_vs_previous"] = s["val_auc"].diff()
    return s


def plot_learning_curve(results, title, path):
    s = summarize(results)
    n_seeds = results["seed"].nunique()
    x = s.index.values
    fig, ax = plt.subplots(figsize=(8, 5.5))
    (l_tr,) = ax.plot(x, s["train_auc"], marker="o", label="Training AUC (eval mode, no augmentation)")
    (l_va,) = ax.plot(x, s["val_auc"], marker="o", label="Validation AUC")
    if n_seeds > 1:
        ax.fill_between(x, s["train_auc"] - s["train_auc_std"], s["train_auc"] + s["train_auc_std"],
                        color=l_tr.get_color(), alpha=0.15)
        ax.fill_between(x, s["val_auc"] - s["val_auc_std"], s["val_auc"] + s["val_auc_std"],
                        color=l_va.get_color(), alpha=0.15)
        note = f"bands: ±1 std over {n_seeds} seeds"
    else:
        ax.fill_between(x, s["val_ci_low"], s["val_ci_high"], color=l_va.get_color(), alpha=0.15)
        note = "band: 95% bootstrap CI over validation lesions (validation noise only; single seed)"
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0%}\n(n={int(n):,})" for f, n in zip(x, s["n_train"])])
    ax.set_xlabel("Fraction of training pool used (number of training images)")
    ax.set_ylabel("ROC AUC (malignant vs benign)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.01, note, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Torch-dependent helpers
# ---------------------------------------------------------------------------
def load_fixed_split():
    """Full data, split exactly like main.py (same call, default seed) so the validation set
    matches your normal runs and the 100% point is comparable to them."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    train_df, val_df = train_val_split(df, stratify_col="risk_level")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_loader(df, transform, shuffle):
    ds = LesionDataset(df, LOCAL_IMAGE_DIR, transform=transform, classification_mode="binary")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True)


@torch.no_grad()
def predict_probs(model, loader):
    """(true labels, P(malignant)) in loader order. Uses batch[0], batch[1] so it still works
    if your dataset also returns extra items (e.g. diagnosis labels)."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        logits = model(images.to(DEVICE, non_blocking=True))
        ps.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
        ys.append(np.asarray(labels))
    return np.concatenate(ys), np.concatenate(ps)


def main():
    assert CLASSIFICATION_MODE == "binary", "This script is written for binary mode (P(malignant) from a 2-logit head)."
    train_df, val_df = load_fixed_split()
    train_tf, eval_tf = get_transforms(IMAGE_SIZE)

    out_dir = Path(RESUME_DIR) if RESUME_DIR else Path("results") / f"datasize_curve_{MODEL_NAME}_{datetime.now():%Y-%m-%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done = {(round(float(r["fraction"]), 6), int(r["seed"])) for r in rows}

    settings = {"model": MODEL_NAME, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                "lr": TRAIN_LR, "fractions": FRACTIONS, "seeds": SEEDS, "use_scheduler": USE_SCHEDULER,
                "use_class_weights": USE_CLASS_WEIGHTS, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "n_train_pool": len(train_df), "n_val": len(val_df), "image_dir": LOCAL_IMAGE_DIR}
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))

    print(f"Training pool: {len(train_df)} images | fixed validation set: {len(val_df)} images")
    print(f"Planned cost ~ {sum(FRACTIONS) * len(SEEDS):.2f}x one full-data run. Output: {out_dir}\n")

    yv_expected = val_df["BinaryTarget"].values
    for seed in SEEDS:
        subsets = nested_group_subsets(train_df, FRACTIONS, seed=seed)
        for frac in FRACTIONS:
            if (round(frac, 6), seed) in done:
                print(f"[skip] fraction {frac:.2f}, seed {seed} already finished")
                continue
            sub = subsets[frac]
            print(f"\n=== fraction {frac:.2f} | seed {seed} | {len(sub)} training images "
                  f"({sub['lesion_id_filled'].nunique()} lesions, {sub['BinaryTarget'].mean():.1%} malignant) ===")
            t0 = time.time()
            set_seed(seed)
            train_loader = make_loader(sub, train_tf, shuffle=True)
            val_loader = make_loader(val_df, eval_tf, shuffle=False)
            model = get_model(MODEL_NAME, num_classes=2)
            history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=TRAIN_LR,
                            use_scheduler=USE_SCHEDULER, use_class_weights=USE_CLASS_WEIGHTS,
                            early_stopping_patience=EARLY_STOPPING_PATIENCE)  # restores best-val-loss weights

            yv, pv = predict_probs(model, val_loader)
            assert np.array_equal(yv, yv_expected), "validation labels out of order — predictions would be misaligned"
            val_auc = roc_auc_score(yv, pv)
            ci_lo, ci_hi = bootstrap_auc_ci(yv, pv, val_df["lesion_id_filled"].values, seed=seed)

            eval_sub = sub if len(sub) <= TRAIN_SCORE_MAX_IMAGES else sub.sample(n=TRAIN_SCORE_MAX_IMAGES, random_state=seed)
            ytr, ptr = predict_probs(model, make_loader(eval_sub.reset_index(drop=True), eval_tf, shuffle=False))
            train_auc = roc_auc_score(ytr, ptr)

            meta = history.get("meta", {})
            rows.append({
                "fraction": frac, "seed": seed, "n_train": len(sub),
                "n_lesions": int(sub["lesion_id_filled"].nunique()),
                "malignant_share": float(sub["BinaryTarget"].mean()),
                "train_auc": train_auc, "val_auc": val_auc, "val_ci_low": ci_lo, "val_ci_high": ci_hi,
                "best_epoch": meta.get("best_epoch"), "actual_epochs": meta.get("actual_epochs"),
                "best_val_loss": meta.get("best_val_loss"), "minutes": round((time.time() - t0) / 60, 1),
            })
            pd.DataFrame(rows).to_csv(results_path, index=False)               # save after every run
            np.savez_compressed(out_dir / f"val_probs_f{frac:.2f}_s{seed}.npz", y=yv, p=pv)
            print(f"-> train AUC {train_auc:.4f} | val AUC {val_auc:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f}) "
                  f"| best epoch {meta.get('best_epoch')} | {rows[-1]['minutes']} min")

            del model, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    plot_learning_curve(results, f"{MODEL_NAME}: AUC vs training-set size", out_dir / "data_size_learning_curve.png")
    s = summarize(results)
    s.round(4).to_csv(out_dir / "summary.csv")
    print("\n" + s.round(4).to_string())
    print(f"\nSaved: {out_dir / 'data_size_learning_curve.png'}, summary.csv, results.csv")
    print("Reading it: compare the validation gain between the last two sizes with the band width. "
          "A gain well inside the band means the curve has flattened.")


if __name__ == "__main__":      # required on Windows because of DataLoader workers
    main()