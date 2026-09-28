"""Orchestrates one full run: load data -> sample -> split -> train -> report.
This is the file you actually run."""

import pandas as pd
from torch.utils.data import DataLoader
from verify_labels import verify_predictions_aligned

from config import (
    PROCESSED_DATA_PATH, LOCAL_IMAGE_DIR, IMAGE_SIZE,
    SAMPLE_SIZE, BATCH_SIZE, EPOCHS, LR, USE_SCHEDULER,
    USE_SMALL_SAMPLE, MODEL_NAME, DEVICE
)
from dataset import LesionDataset, make_tiny_sample, train_val_split, get_transforms
from models import get_model
from train import train
from metrics import generate_report, save_run

if __name__ == "__main__":
    df_merged = pd.read_csv(PROCESSED_DATA_PATH)
    sample_df = df_merged

    if USE_SMALL_SAMPLE:
        sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE)
    train_df, val_df = train_val_split(sample_df)

    print(f"Sample size: {len(sample_df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print("Class counts (sample):")
    print(sample_df["diagnosis"].value_counts())

    train_transform, eval_transform = get_transforms(IMAGE_SIZE)
    train_dataset = LesionDataset(train_df, LOCAL_IMAGE_DIR, transform=train_transform)
    val_dataset = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(MODEL_NAME, num_classes=len(train_dataset.classes))
    print(model)

    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, use_scheduler=USE_SCHEDULER)
    report_results = generate_report(model, val_loader, class_names=train_dataset.classes)

    run_config = {
        "sample_size": SAMPLE_SIZE, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": LR, "use_scheduler": USE_SCHEDULER, "model": MODEL_NAME,
    }
    save_run(run_name=f"M-{MODEL_NAME}_S-size-{SAMPLE_SIZE}_E-{EPOCHS}", config_dict=run_config, history=history, report_results=report_results)

    """
    verify_predictions_aligned(
        model, val_df, LOCAL_IMAGE_DIR, eval_transform,
        class_names=train_dataset.classes, device=DEVICE, n=6
    )
    """