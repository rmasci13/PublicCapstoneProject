"""Reporting helpers for interpreting model performance beyond raw accuracy —
important given the class imbalance in this dataset."""

import json
import re
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score

from config import DEVICE

# Display order only — malignant/pre-malignant grouped first, benign after.
# This does NOT affect training, target encoding, or model outputs; it's
# purely how rows/columns are arranged in the confusion matrix and report.
RISK_GROUPED_ORDER = ["MEL", "BCC", "AK", "SCC", "NV", "BKL", "DF", "VASC"]
BINARY_ORDER = ["Malignant/Pre-Malignant", "Benign"]


def generate_report(model, val_loader, class_names):
    """Runs the model on val_loader and prints a confusion matrix,
    per-class precision/recall/F1, and the majority-class baseline
    accuracy for comparison. Ordering adapts automatically: 8-class mode
    uses RISK_GROUPED_ORDER, binary mode uses BINARY_ORDER."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Pick display order based on how many classes we actually have —
    # this is what makes the function work unchanged for either mode.
    if len(class_names) == 2:
        ordered_names = BINARY_ORDER
    else:
        ordered_names = RISK_GROUPED_ORDER

    # Map each display-order class name back to its actual integer label
    # (class_names is in encoding order, e.g. LABEL_COLS or BINARY_CLASSES
    # from config.py)
    ordered_labels = [class_names.index(name) for name in ordered_names]

    print(f"\nConfusion matrix (rows=true, cols=predicted) — {ordered_names[0]} first:")
    cm = confusion_matrix(all_labels, all_preds, labels=ordered_labels)
    cm_df = pd.DataFrame(cm, index=ordered_names, columns=ordered_names)
    print(cm_df)

    print("\nClassification report:")
    report_dict = classification_report(
        all_labels, all_preds,
        labels=ordered_labels,
        target_names=ordered_names,
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        all_labels, all_preds,
        labels=ordered_labels,
        target_names=ordered_names,
        zero_division=0,
    ))

    # F2 score — weights recall twice as heavily as precision, matching this
    # project's priority of minimizing missed malignant cases over
    # minimizing false alarms. Computed per-class and added into
    # report_dict so it's saved alongside precision/recall/F1 in the CSV.
    f2_per_class = fbeta_score(all_labels, all_preds, labels=ordered_labels, beta=2, average=None, zero_division=0)
    f2_macro = fbeta_score(all_labels, all_preds, labels=ordered_labels, beta=2, average="macro", zero_division=0)

    print("\nF2 scores (recall weighted 2x precision):")
    for name, f2 in zip(ordered_names, f2_per_class):
        report_dict[name]["f2"] = f2
        print(f"  {name}: {f2:.3f}")
    report_dict["macro avg"]["f2"] = f2_macro
    print(f"  macro avg: {f2_macro:.3f}")

    majority_class_acc = pd.Series(all_labels).value_counts().max() / len(all_labels)
    print(f"\nMajority-class baseline accuracy on this val set: {majority_class_acc:.3f}")

    # Sanity check: each row's total in the (reordered) confusion matrix
    # should equal how many times that class actually appears as a true
    # label — an independent check that the reordering didn't scramble
    # which row belongs to which class.
    true_counts = pd.Series(all_labels).value_counts()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    mismatch_found = False
    for name in ordered_names:
        cm_row_sum = cm_df.loc[name].sum()
        actual_count = true_counts.get(class_to_idx[name], 0)
        if cm_row_sum != actual_count:
            mismatch_found = True
    if mismatch_found:
        print("\nWARNING: confusion matrix row-sum mismatch — reordering logic may be scrambling class identity.")
    else:
        print("\nNo issues found in confusion matrix row-sums")

    return {
        "confusion_matrix": cm,
        "confusion_matrix_df": cm_df,
        "classification_report": report_dict,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "majority_class_acc": majority_class_acc,
    }

def plot_learning_curve(history, run_name, save_path=None):
    """Plots train vs val loss (and accuracy) over epochs. If save_path is
    given, saves the figure there instead of/alongside showing it."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val loss", marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Val acc", marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Learning curve — {run_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved learning curve to: {save_path}")

    plt.close(fig)
    return fig


def save_run(model, run_name, config_dict, history, report_results, results_root="results"):
    """Saves everything about a run to results/<run_name>/ so it can be
    compared against other runs later. run_name should be descriptive,
    e.g. 'baseline_20ep_lr1e-3'."""
    # Windows forbids : \ / * ? " < > | in file/folder names — strip or
    # replace them so any run_name string is safe to use as a folder name.
    safe_name = re.sub(r'[:\\/*?"<>|]', "-", run_name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(results_root) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot — exact hyperparameters used for this run
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Per-epoch history (loss/acc/lr/time)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Learning curve plot (train vs val loss/accuracy over epochs)
    plot_learning_curve(history, run_name, save_path=run_dir / "learning_curve.png")

    # Classification report as CSV (easy to drop into a writeup table)
    pd.DataFrame(report_results["classification_report"]).transpose().to_csv(
        run_dir / "classification_report.csv"
    )

    # Confusion matrix as CSV
    report_results["confusion_matrix_df"].to_csv(run_dir / "confusion_matrix.csv")

    # Confusion matrix as a plotted image, for the writeup
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(report_results["confusion_matrix"], cmap="YlOrRd")
    ax.set_xticks(range(len(report_results["confusion_matrix_df"].columns)))
    ax.set_yticks(range(len(report_results["confusion_matrix_df"].index)))
    ax.set_xticklabels(report_results["confusion_matrix_df"].columns)
    ax.set_yticklabels(report_results["confusion_matrix_df"].index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(run_name)
    for i in range(report_results["confusion_matrix"].shape[0]):
        for j in range(report_results["confusion_matrix"].shape[1]):
            ax.text(j, i, report_results["confusion_matrix"][i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    torch.save(model.state_dict(), run_dir / "model_weights.pt")
    print(f"\nSaved run results to: {run_dir}")
    return run_dir