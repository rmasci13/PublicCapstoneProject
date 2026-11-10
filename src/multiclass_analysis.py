"""Loads a previously trained BINARY model from its saved weights, runs it
over a dataset split in eval mode (no weight updates), and breaks its
predictions down by the 8 TRUE diagnoses to show which kinds of lesions the
model struggles with.

The model itself only outputs Benign vs Malignant/Pre-Malignant, so this is
NOT an 8-way classification — it answers "for each true diagnosis (MEL, NV,
BCC, ...), how often does the model put it on the correct side of the
benign/malignant line?"

It also sweeps the decision threshold: instead of always calling an image
malignant when P(malignant) > 0.5, it shows what would happen to each
diagnosis (and to overall recall/precision/F2) at other cutoffs.

Edit the three settings below, then run: python per_class_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE,
    SAMPLE_SIZE, USE_SMALL_SAMPLE, DEVICE, BINARY_CLASSES, set_seed,
    USE_METADATA_ONLY, LABEL_COLS
)
from dataset import LesionDataset, make_tiny_sample, train_val_split, get_transforms, MetadataLesionDataset
from models import get_model
from metrics import RISK_GROUPED_ORDER
from PIL import Image

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
# Folder of the saved run whose model you want to analyze (contains model_weights.pt)
RUN_DIR = Path(
    "results/2026-11-01_0113_M-EfficientViT_b2_Tune_binary_S-size-25331_E-50")

# Must match the architecture that run was trained with (not necessarily
# whatever MODEL_NAME is currently set to in config.py)
MODEL_NAME_TO_LOAD = "EfficientViT_b2_Tune"

# "val" = images the model never trained on (honest read of its weaknesses)
# "train" = images it trained on (will look optimistic; useful for seeing
# what it still can't fit even after seeing them)
SPLIT_TO_EVALUATE = "val"

# Cutoffs tried in the threshold sweep (image is called Malignant when
# P(malignant) >= cutoff). 0.50 is what the model's normal predictions use.
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
# Subset of cutoffs shown in the printed tables (everything is still saved)
PRINT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def load_split(split):
    """Rebuilds the exact same train/val split main.py used. This only
    works if USE_SMALL_SAMPLE / SAMPLE_SIZE in config.py match what the
    saved model was trained with — otherwise 'val' would contain images the
    model actually trained on."""
    if USE_METADATA_ONLY:
        val_df = pd.read_csv('../data/processed/val_metadata.csv')

        return val_df
    else:
        df_merged = pd.read_csv(PROCESSED_DATA_PATH)
        stratify_col = "risk_level"  # binary-mode split, same as main.py

        sample_df = df_merged
        if USE_SMALL_SAMPLE:
            sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, stratify_col=stratify_col)
        train_df, val_df = train_val_split(sample_df, stratify_col=stratify_col)

        print(f"Split sizes: {len(train_df)} train / {len(val_df)} val")
        return val_df if split == "val" else train_df


def load_model(run_dir, model_name, dataset=None):
    if USE_METADATA_ONLY:
        input_dim = len(dataset.feature_cols)
        model = get_model(model_name, num_classes=len(dataset.classes), input_dim=input_dim)
    else:
        model = get_model(model_name, num_classes=len(BINARY_CLASSES))
    state = torch.load(run_dir / "model_weights.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()  # eval mode: dropout off, batchnorm uses running stats
    return model


def predict(model, loader):
    """Returns predicted class (0=Benign, 1=Malignant) and the model's
    probability of Malignant for every image, in loader order."""
    all_preds, all_probs = [], []
    with torch.no_grad():  # no gradients -> weights are never adjusted
        for batch in loader:
            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch
            outputs = model(images.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


def summarize_by_diagnosis(df, preds, probs):
    malignant_dx = set(df.loc[df["BinaryTarget"] == 1, "diagnosis"].unique())

    rows = []
    for dx in RISK_GROUPED_ORDER:
        mask = (df["diagnosis"] == dx).values
        n = int(mask.sum())
        if n == 0:
            continue
        pred_malignant = int((preds[mask] == 1).sum())
        pred_benign = n - pred_malignant
        is_malignant_class = dx in malignant_dx
        correct = pred_malignant if is_malignant_class else pred_benign
        rows.append({
            "diagnosis": dx,
            "true_group": "Malignant/Pre-Malignant" if is_malignant_class else "Benign",
            "n": n,
            "predicted_Malignant": pred_malignant,
            "predicted_Benign": pred_benign,
            # malignant classes: share flagged malignant (recall/sensitivity)
            # benign classes: share called benign (specificity)
            "correct_rate": correct / n,
            "mean_prob_malignant": float(probs[mask].mean()),
        })
    return pd.DataFrame(rows)


def save_per_image_predictions(df, preds, probs, path):
    """One row per image with its true diagnosis and the model's
    probability of Malignant, so the threshold analysis can be redone later
    (or sliced differently) without re-running the model."""
    out = df[["image", "diagnosis", "risk_level", "BinaryTarget"]].copy()
    out["pred"] = preds
    out["prob_malignant"] = probs
    out.to_csv(path, index=False)
    return out


def threshold_sweep_by_diagnosis(df, probs, thresholds=THRESHOLDS):
    """For every cutoff and every true diagnosis: how many images (and what
    share) would be flagged Malignant at that cutoff.
      - malignant diagnoses: this share is recall (higher = fewer misses)
      - benign diagnoses: this share is the false-alarm rate (lower = better)"""
    malignant_dx = set(df.loc[df["BinaryTarget"] == 1, "diagnosis"].unique())
    rows = []
    for t in thresholds:
        flagged = probs >= t
        for dx in RISK_GROUPED_ORDER:
            mask = (df["diagnosis"] == dx).values
            n = int(mask.sum())
            if n == 0:
                continue
            n_flagged = int(flagged[mask].sum())
            rows.append({
                "threshold": float(t),
                "diagnosis": dx,
                "true_group": "Malignant/Pre-Malignant" if dx in malignant_dx else "Benign",
                "n": n,
                "flagged_malignant": n_flagged,
                "share_flagged_malignant": n_flagged / n,
            })
    return pd.DataFrame(rows)


def overall_threshold_metrics(y_true, probs, thresholds=THRESHOLDS):
    """Overall binary recall / precision / F1 / F2 / accuracy (plus raw missed and
    false-alarm counts) at every cutoff."""
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        # F1 score calculation
        f1_denom = precision + recall
        f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0

        # F2 score calculation
        f2_denom = (4 * precision) + recall
        f2 = (5 * precision * recall) / f2_denom if f2_denom > 0 else 0.0

        rows.append({
            "threshold": float(t),
            "malignant_recall": recall,
            "malignant_precision": precision,
            "malignant_f1": f1,
            "malignant_f2": f2,
            "accuracy": (tp + tn) / len(y_true),
            "missed_malignant_FN": fn,
            "false_alarms_FP": fp,
        })
    return pd.DataFrame(rows)


def plot_summary(summary, title, save_path):
    counts = summary[["predicted_Malignant", "predicted_Benign"]].values
    fractions = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fractions, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nMalignant", "Predicted\nBenign"])
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["diagnosis"])
    ax.set_ylabel("True diagnosis")
    ax.set_title(title)
    for i in range(counts.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{counts[i, j]}\n({fractions[i, j]:.0%})", ha="center", va="center", fontsize=8)
    # Divider between the 4 malignant-group rows and the benign-group rows
    n_malignant = int((summary["true_group"] == "Malignant/Pre-Malignant").sum())
    ax.axhline(n_malignant - 0.5, color="black", linewidth=2)
    fig.colorbar(im, label="Share of that diagnosis")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(by_dx, overall, title, save_path):
    """Left: share of each diagnosis flagged Malignant as the cutoff moves
    (solid = truly malignant diagnoses, want high; dashed = truly benign,
    want low). Right: overall recall / precision / F1 / F2 at each cutoff."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for dx in RISK_GROUPED_ORDER:
        sub = by_dx[by_dx["diagnosis"] == dx]
        if sub.empty:
            continue
        is_malignant = sub["true_group"].iloc[0] == "Malignant/Pre-Malignant"
        ax1.plot(sub["threshold"], sub["share_flagged_malignant"],
                 linestyle="-" if is_malignant else "--", marker="o", markersize=3, label=dx)
    ax1.axvline(0.5, color="gray", linewidth=1, linestyle=":")
    ax1.set_xlabel("Cutoff: call Malignant when P(malignant) >= this")
    ax1.set_ylabel("Share of that diagnosis flagged Malignant")
    ax1.set_title("By true diagnosis (solid = malignant, dashed = benign)")
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot metric curves
    ax2.plot(overall["threshold"], overall["malignant_recall"], marker="o", markersize=3, label="Recall")
    ax2.plot(overall["threshold"], overall["malignant_precision"], marker="o", markersize=3, label="Precision")
    ax2.plot(overall["threshold"], overall["malignant_f1"], marker="o", markersize=3, label="F1")
    ax2.plot(overall["threshold"], overall["malignant_f2"], marker="o", markersize=3, label="F2")

    # Locate best thresholds
    best_f1 = overall.loc[overall["malignant_f1"].idxmax()]
    best_f2 = overall.loc[overall["malignant_f2"].idxmax()]

    # Draw vertical reference lines for optimal thresholds
    ax2.axvline(best_f1["threshold"], color="blue", linewidth=1.2, linestyle="--",
                label=f"Best F1 ({best_f1['malignant_f1']:.3f}) at {best_f1['threshold']:.2f}")
    ax2.axvline(best_f2["threshold"], color="green", linewidth=1.2, linestyle="--",
                label=f"Best F2 ({best_f2['malignant_f2']:.3f}) at {best_f2['threshold']:.2f}")

    # Standard 0.5 decision boundary reference
    ax2.axvline(0.5, color="gray", linewidth=1, linestyle=":", label="Default 0.50 cutoff")

    ax2.set_xlabel("Cutoff: call Malignant when P(malignant) >= this")
    ax2.set_title("Overall malignant metrics")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stripplot(pred_path, save_path):
    df = pd.read_csv(pred_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df, x='diagnosis', y='prob_malignant',
        palette='Set2', boxprops=dict(alpha=0.7), ax=ax
    )
    sns.stripplot(
        data=df, x='diagnosis', y='prob_malignant',
        color='black', alpha=0.3, jitter=0.2, size=4, ax=ax
    )

    ax.set_title('Predicted Probability of Malignancy by Diagnosis (Validation Set)')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Probability Malignant')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_diagnosis_confidence_grid(pred_csv_path, image_dir, save_path):
    """Pulls 3 MEL, 3 BKL, and 3 NV samples across malignancy probability

    score tiers (high, borderline 0.40-0.60, low) and plots them in a 3x3 grid.
    """
    df = pd.read_csv(pred_csv_path)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    targets = [("MEL", axes[0]), ("BKL", axes[1]), ("NV", axes[2])]

    for dx, row_axes in targets:
        dx_df = df[df["diagnosis"] == dx].copy()
        if len(dx_df) == 0:
            print(f"Warning: No samples found for diagnosis {dx}")
            continue

        # Define score brackets for your 3 columns
        high_bracket = dx_df[dx_df["prob_malignant"] >= 0.85]
        mid_bracket = dx_df[
            (dx_df["prob_malignant"] >= 0.40)
            & (dx_df["prob_malignant"] <= 0.60)
        ]
        low_bracket = dx_df[dx_df["prob_malignant"] <= 0.15]

        # Randomly sample from brackets (fallback gracefully if a bracket is empty)
        high_sample = (
            high_bracket.sample(1).iloc[0]
            if len(high_bracket) > 0
            else dx_df.sort_values(by="prob_malignant", ascending=False).iloc[0]
        )
        mid_sample = (
            mid_bracket.sample(1).iloc[0]
            if len(mid_bracket) > 0
            else dx_df.iloc[len(dx_df) // 2]
        )
        low_sample = (
            low_bracket.sample(1).iloc[0]
            if len(low_bracket) > 0
            else dx_df.sort_values(by="prob_malignant", ascending=True).iloc[0]
        )

        selected = [
            (
                "High Malignancy Score (False Alarm if Benign)",
                high_sample,
            ),
            (
                "Borderline Score 0.40–0.60 (Uncertainty Zone)",
                mid_sample,
            ),
            (
                "Low Malignancy Score (False Negative if Malignant)",
                low_sample,
            ),
        ]

        for col_idx, (tier_desc, row) in enumerate(selected):
            ax = row_axes[col_idx]
            img_name = row["image"]
            prob = row["prob_malignant"]
            pred = row["pred"]

            # Load image from disk
            img_path = Path(image_dir) / f"{img_name}.jpg"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Image not found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            pred_label = "Malignant" if pred == 1 else "Benign"
            ax.set_title(
                f"{dx} — {tier_desc}\nP(Malignant): {prob:.3f} (Pred: {pred_label})",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle(
        "Visual Audit: Predicted Malignancy Scores for MEL, BKL, and NV",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved malignancy score grid to: {save_path}")


if __name__ == "__main__":  # required on Windows because of DataLoader workers
    set_seed()

    df = load_split(SPLIT_TO_EVALUATE).reset_index(drop=True)
    _, eval_transform = get_transforms(IMAGE_SIZE)  # no augmentation, even for train split
    if USE_METADATA_ONLY:
        dataset = MetadataLesionDataset(df)
    else:
        dataset = LesionDataset(df, LOCAL_IMAGE_DIR, transform=eval_transform, classification_mode="binary")

    # shuffle=False keeps predictions aligned row-for-row with dataset.df
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = load_model(RUN_DIR, MODEL_NAME_TO_LOAD, dataset)
    preds, probs = predict(model, loader)
    assert len(preds) == len(dataset.df), "Prediction count doesn't match dataframe rows"

    # --- Overall binary result (sanity check: on 'val' this should match the run's saved report) ---
    y_true = dataset.df["BinaryTarget"].values
    print(f"\nOverall binary results on the {SPLIT_TO_EVALUATE} split:")
    print(pd.DataFrame(
        confusion_matrix(y_true, preds, labels=[1, 0]),
        index=["True Malignant", "True Benign"], columns=["Pred Malignant", "Pred Benign"],
    ))
    print(classification_report(y_true, preds, labels=[1, 0],
                                target_names=["Malignant/Pre-Malignant", "Benign"], zero_division=0))

    # --- Breakdown by the 8 true diagnoses ---
    summary = summarize_by_diagnosis(dataset.df, preds, probs)
    print(f"Breakdown by true diagnosis ({SPLIT_TO_EVALUATE} split):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    csv_path = RUN_DIR / f"per_diagnosis_{SPLIT_TO_EVALUATE}.csv"
    png_path = RUN_DIR / f"per_diagnosis_{SPLIT_TO_EVALUATE}.png"
    summary.to_csv(csv_path, index=False)
    plot_summary(summary, f"{MODEL_NAME_TO_LOAD} — binary predictions by true diagnosis ({SPLIT_TO_EVALUATE})", png_path)
    print(f"\nSaved: {csv_path}\nSaved: {png_path}")

    # --- Per-image probabilities + threshold sweep ---
    pred_path = RUN_DIR / f"per_image_predictions_{SPLIT_TO_EVALUATE}.csv"
    save_per_image_predictions(dataset.df, preds, probs, pred_path)

    by_dx = threshold_sweep_by_diagnosis(dataset.df, probs)
    overall = overall_threshold_metrics(y_true, probs)

    # Sanity check: the 0.50 cutoff should reproduce the model's own predictions
    at_half = int((probs >= 0.5).sum())
    print(f"\nCutoff 0.50 flags {at_half} images; model's own predictions flag {int((preds == 1).sum())} "
          f"({'matches' if at_half == int((preds == 1).sum()) else 'MISMATCH — check before trusting the sweep'})")

    shown = by_dx[np.isclose(by_dx["threshold"].values[:, None], PRINT_THRESHOLDS).any(axis=1)]
    pivot = (shown.pivot(index="diagnosis", columns="threshold", values="share_flagged_malignant")
                  .reindex(RISK_GROUPED_ORDER))
    print("\nShare of each true diagnosis flagged Malignant at different cutoffs")
    print("(MEL/BCC/AK/SCC: this is recall, want HIGH.  NV/BKL/DF/VASC: this is false-alarm rate, want LOW.)")
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))

    shown_overall = overall[np.isclose(overall["threshold"].values[:, None], PRINT_THRESHOLDS).any(axis=1)]
    print("\nOverall metrics at different cutoffs")
    print(shown_overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best_f1 = overall.loc[overall["malignant_f1"].idxmax()]
    best_f2 = overall.loc[overall["malignant_f2"].idxmax()]

    print(f"\nBest malignant F1: {best_f1['malignant_f1']:.3f} at cutoff {best_f1['threshold']:.2f}")
    print(f"Best malignant F2: {best_f2['malignant_f2']:.3f} at cutoff {best_f2['threshold']:.2f}")

    by_dx.to_csv(RUN_DIR / f"threshold_sweep_by_diagnosis_{SPLIT_TO_EVALUATE}.csv", index=False)
    overall.to_csv(RUN_DIR / f"threshold_sweep_overall_{SPLIT_TO_EVALUATE}.csv", index=False)
    sweep_png = RUN_DIR / f"threshold_sweep_{SPLIT_TO_EVALUATE}.png"
    plot_threshold_sweep(by_dx, overall, f"{MODEL_NAME_TO_LOAD} — threshold sweep ({SPLIT_TO_EVALUATE})", sweep_png)
    stripplot_png = RUN_DIR / f"stripplot.png"
    plot_stripplot(pred_path, stripplot_png)
    print(f"\nSaved: {pred_path}\nSaved: {sweep_png}\n(+ threshold_sweep_*.csv in the same folder)")
    grid_png_path = RUN_DIR / f"diagnosis_confidence_grid_{SPLIT_TO_EVALUATE}.png"
    plot_diagnosis_confidence_grid(pred_path, LOCAL_IMAGE_DIR, grid_png_path)
    print(f"\nSaved: {grid_png_path}")