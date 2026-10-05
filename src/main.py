"""Orchestrates one full run: load data -> sample -> split -> train -> report.
This is the file you actually run."""

import pandas as pd
from torch.utils.data import DataLoader
from verify_labels import verify_predictions_aligned

from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE,
    SAMPLE_SIZE, BATCH_SIZE, EPOCHS, LR, USE_SCHEDULER,
    USE_SMALL_SAMPLE, MODEL_NAME, DEVICE, USE_CLASS_WEIGHTS,
    CLASSIFICATION_MODE, LR_FINETUNE, EARLY_STOPPING_PATIENCE,
    set_seed, DAMPEN_WEIGHTING
)
from dataset import LesionDataset, make_tiny_sample, train_val_split, get_transforms
from models import get_model
from train import train
from metrics import generate_report, save_run

if __name__ == "__main__":
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    sample_df = df_merged

    # Stratify on 'diagnosis' (8-way) for multiclass, 'risk_level' (2-way)
    # for binary — keeps the sample/split balanced on whichever grouping
    # actually matters for the mode you're training in.
    stratify_col = "diagnosis" if CLASSIFICATION_MODE == "multiclass" else "risk_level"

    if USE_SMALL_SAMPLE:
        sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, stratify_col=stratify_col)
    train_df, val_df = train_val_split(sample_df, stratify_col=stratify_col)

    print(f"Mode: {CLASSIFICATION_MODE} | Sample size: {len(sample_df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Class counts (sample, by {stratify_col}):")
    print(sample_df[stratify_col].value_counts())
    print(f"Train class proportions (by {stratify_col}):")
    print(train_df[stratify_col].value_counts(normalize=True))
    print(f"\nVal class proportions (by {stratify_col}):")
    print(val_df[stratify_col].value_counts(normalize=True))

    train_transform, eval_transform = get_transforms(IMAGE_SIZE)
    train_dataset = LesionDataset(train_df, LOCAL_IMAGE_DIR, transform=train_transform,
                                  classification_mode=CLASSIFICATION_MODE)
    val_dataset = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=eval_transform,
                                classification_mode=CLASSIFICATION_MODE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    set_seed()
    model = get_model(MODEL_NAME, num_classes=len(train_dataset.classes))
    print(model)

    effective_lr = LR_FINETUNE if MODEL_NAME == "ResNet18_Tune" else LR

    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=effective_lr, use_scheduler=USE_SCHEDULER,
                    use_class_weights=USE_CLASS_WEIGHTS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    report_results = generate_report(model, val_loader, class_names=train_dataset.classes)

    actual_sample_size = len(sample_df)

    run_config = {
        "classification_mode": CLASSIFICATION_MODE,
        "sample_size": actual_sample_size, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": effective_lr, "use_scheduler": USE_SCHEDULER, "model": MODEL_NAME,
        "use_class_weights": USE_CLASS_WEIGHTS, "weight_balance_dampening": DAMPEN_WEIGHTING
    }
    save_run(
        model=model, run_name=f"M-{MODEL_NAME}_{CLASSIFICATION_MODE}_S-size-{actual_sample_size}_E-{EPOCHS}",
        config_dict=run_config, history=history, report_results=report_results
    )

    """
    verify_predictions_aligned(
        model, val_df, LOCAL_IMAGE_DIR, eval_transform,
        class_names=train_dataset.classes, device=DEVICE, n=6
    )
    """