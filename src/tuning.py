"""Hyperparameter tuning — varies ONE hyperparameter at a time while
holding everything else (data split, model architecture, other settings)
fixed, so any difference in results is attributable to that one variable.

Edit HYPERPARAM_NAME and HYPERPARAM_VALUES below to tune a different
setting (lr, batch_size, dropout, etc. — see note at the bottom on which
ones need extra wiring).
"""

import pandas as pd
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE,
    SAMPLE_SIZE, BATCH_SIZE, EPOCHS, LR, USE_SCHEDULER,
    USE_SMALL_SAMPLE, MODEL_NAME, USE_CLASS_WEIGHTS,
    CLASSIFICATION_MODE, EARLY_STOPPING_PATIENCE,
    set_seed, SEED
)
from dataset import LesionDataset, make_tiny_sample, train_val_split, get_transforms
from models import get_model
from train import train
from metrics import generate_report, save_run

# ---------------------------------------------------------------------------
# What to tune
# ---------------------------------------------------------------------------
HYPERPARAM_NAME = "batch_size"
HYPERPARAM_VALUES = [16, 32, 64, 128, 256]



def load_fixed_split():
    """Loads and splits the data ONCE, outside the tuning loop, so every
    hyperparameter value is tested against the exact same train/val split."""
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    stratify_col = "diagnosis" if CLASSIFICATION_MODE == "multiclass" else "risk_level"

    sample_df = df_merged
    if USE_SMALL_SAMPLE:
        sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, stratify_col=stratify_col)
    train_df, val_df = train_val_split(sample_df, stratify_col=stratify_col, random_state=SEED)

    print(f"Fixed split for tuning: {len(train_df)} train / {len(val_df)} val "
          f"(sample size {len(sample_df)})")
    return train_df, val_df


def build_loaders(train_df, val_df, batch_size=BATCH_SIZE):
    train_transform, eval_transform = get_transforms(IMAGE_SIZE)
    train_dataset = LesionDataset(train_df, LOCAL_IMAGE_DIR, transform=train_transform,
                                   classification_mode=CLASSIFICATION_MODE)
    val_dataset = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=eval_transform,
                                 classification_mode=CLASSIFICATION_MODE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_dataset


def run_one(hyperparam_value, train_df, val_df):
    """Trains and evaluates one model with HYPERPARAM_NAME set to
    hyperparam_value, everything else held at the config.py defaults."""
    set_seed()  # reset before building the model so initial weights match across runs

    # Only batch_size needs its own DataLoader rebuild; everything else
    # (lr, dropout, etc.) is just passed through to train() or get_model().
    batch_size = hyperparam_value if HYPERPARAM_NAME == "batch_size" else BATCH_SIZE
    train_loader, val_loader, train_dataset = build_loaders(train_df, val_df, batch_size=batch_size)

    model_kwargs ={}
    if HYPERPARAM_NAME == "dropout":
        model_kwargs["dropout"] = hyperparam_value
    model = get_model(MODEL_NAME, num_classes=len(train_dataset.classes), **model_kwargs)
    print(model)

    lr = hyperparam_value if HYPERPARAM_NAME == "lr" else LR

    history = train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=lr, use_scheduler=USE_SCHEDULER,
        use_class_weights=USE_CLASS_WEIGHTS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    report_results = generate_report(model, val_loader, class_names=train_dataset.classes)

    run_config = {
        "classification_mode": CLASSIFICATION_MODE,
        "sample_size": len(train_df) + len(val_df), "image_size": IMAGE_SIZE,
        "batch_size": batch_size, "epochs": len(history["train_loss"]), "lr": lr,
        "use_scheduler": USE_SCHEDULER, "model": MODEL_NAME,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "tuning_param": HYPERPARAM_NAME, "tuning_value": hyperparam_value,
    }
    run_dir = save_run(
        model,
        run_name=f"TUNE-{HYPERPARAM_NAME}-{hyperparam_value}_{MODEL_NAME}_{CLASSIFICATION_MODE}",
        config_dict=run_config, history=history, report_results=report_results,
    )

    # Pull out the headline numbers for the summary table
    report = report_results["classification_report"]
    malignant_key = "Malignant/Pre-Malignant" if CLASSIFICATION_MODE == "binary" else None

    return {
        HYPERPARAM_NAME: hyperparam_value,
        "final_epoch": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
        "val_acc": history["val_acc"][-1],
        "accuracy": report["accuracy"],
        "malignant_recall": report[malignant_key]["recall"] if malignant_key else None,
        "malignant_precision": report[malignant_key]["precision"] if malignant_key else None,
        "malignant_f2": report[malignant_key]["f2"] if malignant_key else None,
        "macro_f2": report["macro avg"]["f2"],
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    train_df, val_df = load_fixed_split()

    results = []
    for value in HYPERPARAM_VALUES:
        print(f"\n{'=' * 60}\nTuning {HYPERPARAM_NAME} = {value}\n{'=' * 60}")
        result = run_one(value, train_df, val_df)
        results.append(result)
        print(f"Result: {result}")

    summary_df = pd.DataFrame(results)

    # Sort by malignant_f2 (binary mode) or macro_f2 (multiclass) so the
    # best-performing hyperparameter value is immediately visible at the
    # top, rather than requiring a manual scan of the table.
    sort_col = "malignant_f2" if CLASSIFICATION_MODE == "binary" else "macro_f2"
    summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    summary_path = f"results/tuning_summary_{HYPERPARAM_NAME}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}\nTuning complete. Summary (sorted by {sort_col}, best first):\n{'=' * 60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison summary to: {summary_path}")
    print(f"Best {HYPERPARAM_NAME}: {summary_df.iloc[0][HYPERPARAM_NAME]} "
          f"({sort_col}={summary_df.iloc[0][sort_col]:.3f})")