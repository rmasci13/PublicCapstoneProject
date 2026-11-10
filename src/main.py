"""Orchestrates one full run: load data -> sample -> split -> train -> report.
This is the file you actually run."""

import pandas as pd
from torch.utils.data import DataLoader
from verify_labels import verify_predictions_aligned
from losses import BOOST_MAP
from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE,
    SAMPLE_SIZE, BATCH_SIZE, EPOCHS, LR, USE_SCHEDULER,
    USE_SMALL_SAMPLE, MODEL_NAME, DEVICE, USE_CLASS_WEIGHTS,
    CLASSIFICATION_MODE, LR_FINETUNE, EARLY_STOPPING_PATIENCE,
    set_seed, DAMPEN_WEIGHTING, OPTIMIZER, WEIGHT_SQUARE, LAMBDA_SCHEDULER,
    CUSTOM_LOSS, POWER, USE_METADATA_ONLY, ALPHA, GAMMA, HAS_CROP
)
from dataset import LesionDataset, make_tiny_sample, train_val_split, get_transforms, MetadataLesionDataset
from models import get_model
from train import train
from metrics import generate_report, save_run

if __name__ == "__main__":
    set_seed()
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    sample_df = df_merged

    # Stratify on 'diagnosis' (8-way) for multiclass, 'risk_level' (2-way)
    stratify_col = "diagnosis" if CLASSIFICATION_MODE == "multiclass" else "risk_level"

    if USE_METADATA_ONLY:
        train_df = pd.read_csv('../data/processed/train_metadata.csv')
        val_df = pd.read_csv('../data/processed/val_metadata.csv')

        train_dataset = MetadataLesionDataset(train_df)
        val_dataset = MetadataLesionDataset(val_df)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        input_dim = len(train_dataset.feature_cols)
        model = get_model(MODEL_NAME, num_classes=len(train_dataset.classes), input_dim=input_dim)
    else:
        if USE_SMALL_SAMPLE:
            sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, stratify_col=stratify_col)
        train_df, val_df = train_val_split(sample_df, stratify_col=stratify_col)

        train_transform, eval_transform = get_transforms(IMAGE_SIZE)
        train_dataset = LesionDataset(train_df, LOCAL_IMAGE_DIR, transform=train_transform,
                                      classification_mode=CLASSIFICATION_MODE)
        val_dataset = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=eval_transform,
                                    classification_mode=CLASSIFICATION_MODE)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        model = get_model(MODEL_NAME, num_classes=len(train_dataset.classes))

    print(f"Mode: {CLASSIFICATION_MODE} | Sample size: {len(sample_df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Class counts (sample, by {stratify_col}):")
    print(sample_df[stratify_col].value_counts())
    print(f"Train class proportions (by {stratify_col}):")
    print(train_df[stratify_col].value_counts(normalize=True))
    print(f"\nVal class proportions (by {stratify_col}):")
    print(val_df[stratify_col].value_counts(normalize=True))
    print(model)

    effective_lr = LR

    print(f"Using effective learning rate: {effective_lr}")

    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=effective_lr, use_scheduler=USE_SCHEDULER,
                    use_class_weights=USE_CLASS_WEIGHTS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    report_results = generate_report(model, val_loader, class_names=train_dataset.classes)

    actual_sample_size = len(sample_df)

    run_config = {
        "classification_mode": CLASSIFICATION_MODE,
        "sample_size": actual_sample_size, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": effective_lr, "use_scheduler": USE_SCHEDULER, "model": MODEL_NAME,
        "use_class_weights": USE_CLASS_WEIGHTS, "weight_balance_dampening": DAMPEN_WEIGHTING,
        "processed_data_path": PROCESSED_DATA_PATH, "image_dir": LOCAL_IMAGE_DIR, "optimizer": OPTIMIZER,
        "weight_square": WEIGHT_SQUARE, "lambda_scheduler": LAMBDA_SCHEDULER, "custom_loss": CUSTOM_LOSS,
        "boost_map": BOOST_MAP, "power": POWER, "alpha": ALPHA, "gamma": GAMMA, "use_metadata_only": USE_METADATA_ONLY,
        "transform_has_CropBlackBorder": HAS_CROP
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