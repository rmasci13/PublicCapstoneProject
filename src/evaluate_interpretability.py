"""Loads pre-computed predictions from per_image_predictions_{split}.csv, identifies
key test cases (True Positives, False Positives, and border-artifact candidates), and
runs Grad-CAM and LIME strictly on those selected samples. Also generates a 4x3
confidence grid (MEL, BCC, NV, BKL) with Grad-CAM overlays.

Usage: python evaluate_interpretability.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from border_crop import CropBlackBorder

# PyTorch Grad-CAM library imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# LIME imports
from lime import lime_image
from skimage.segmentation import mark_boundaries

from config import (
    LOCAL_IMAGE_DIR, IMAGE_SIZE, DEVICE, BINARY_CLASSES, set_seed,
    USE_METADATA_ONLY
)
from dataset import get_transforms
from models import get_model

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
RUN_DIR = Path(
    "results/BestConfigRuns/BEST_2026-10-31_2326_M-EfficientViT_b2_Tune_binary_S-size-25331_E-50")

MODEL_NAME_TO_LOAD = "EfficientViT_b2_Tune"
SPLIT_TO_EVALUATE = "val"


def load_model(run_dir, model_name):
    """Loads saved model weights for feature attribution mapping."""
    model = get_model(model_name, num_classes=len(BINARY_CLASSES))
    state = torch.load(run_dir / "model_weights.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def has_black_borders(img_path, threshold=35):
    """Fast check: detects if an image contains harsh dark/black dermatoscope borders in corners."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        h, w = gray.shape
        corners = [
            gray[:h//10, :w//10],
            gray[:h//10, -w//10:],
            gray[-h//10:, :w//10],
            gray[-h//10:, -w//10:]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])
        return corner_mean < threshold
    except Exception:
        return False


def get_target_layer(model, model_name):
    """Automatically selects the final convolutional feature layer based on architecture."""
    if "EfficientNet" in model_name:
        return model.features[-1]
    elif "ResNet" in model_name:
        return model.layer4[-1]
    else:
        conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        return conv_layers[-1] if conv_layers else None


def generate_gradcam_grid(model, df_subset, model_name, save_path):
    """Generates a multi-panel Grad-CAM visualization grid for a pre-filtered subset of images."""
    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        print("Warning: Could not automatically locate target layer for Grad-CAM.")
        return

    cam = GradCAM(model=model, target_layers=[target_layer])

    n_samples = min(len(df_subset), 4)
    if n_samples == 0:
        print("No samples available for this category.")
        return

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    _, eval_transform = get_transforms(IMAGE_SIZE)

    for idx, (_, row) in enumerate(df_subset.head(n_samples).iterrows()):
        img_name = row["image"]
        img_path = Path(LOCAL_IMAGE_DIR) / f"{img_name}.jpg"

        orig_img = Image.open(img_path).convert("RGB")
        input_tensor = eval_transform(orig_img).unsqueeze(0).to(DEVICE)

        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Properly crop for display matching the model input
        cropped_img = CropBlackBorder()(orig_img).resize((IMAGE_SIZE, IMAGE_SIZE))
        cropped_np = np.array(cropped_img) / 255.0
        visualization = show_cam_on_image(cropped_np, grayscale_cam, use_rgb=True)

        ax = axes[idx]
        ax.imshow(visualization)
        ax.set_title(
            f"Ground Truth: {row['diagnosis']}\nPred: {'Mal' if row['pred'] == 1 else 'Ben'} "
            f"(P={row['prob_malignant']:.2f})\nBorder: {'Yes' if row['has_border'] else 'No'}",
            fontsize=8
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM grid: {save_path}")


def generate_gradcam_confidence_grid(model, df, model_name, save_path):
    """Generates a 4x3 confidence grid (Rows: MEL, BCC, NV, BKL;
    Columns: High >=0.85, Mid 0.40-0.60, Low <0.20) with Grad-CAM overlays.
    """
    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        print("Warning: Could not automatically locate target layer for Grad-CAM.")
        return

    cam = GradCAM(model=model, target_layers=[target_layer])
    _, eval_transform = get_transforms(IMAGE_SIZE)

    diagnoses = ["MEL", "BCC", "NV", "BKL"]
    fig, axes = plt.subplots(len(diagnoses), 3, figsize=(14, 16))

    for row_idx, dx in enumerate(diagnoses):
        dx_df = df[df["diagnosis"] == dx].copy()
        row_axes = axes[row_idx]

        if len(dx_df) == 0:
            for col_idx in range(3):
                row_axes[col_idx].axis("off")
                row_axes[col_idx].text(0.5, 0.5, f"No samples for {dx}", ha="center", va="center")
            continue

        # Define confidence brackets
        high_bracket = dx_df[dx_df["prob_malignant"] >= 0.85]
        mid_bracket = dx_df[(dx_df["prob_malignant"] >= 0.40) & (dx_df["prob_malignant"] <= 0.60)]
        low_bracket = dx_df[dx_df["prob_malignant"] < 0.20]

        # Graceful fallbacks if a bracket is empty
        high_sample = high_bracket.sample(1).iloc[0] if len(high_bracket) > 0 else dx_df.sort_values(by="prob_malignant", ascending=False).iloc[0]
        mid_sample = mid_bracket.sample(1).iloc[0] if len(mid_bracket) > 0 else dx_df.iloc[len(dx_df) // 2]
        low_sample = low_bracket.sample(1).iloc[0] if len(low_bracket) > 0 else dx_df.sort_values(by="prob_malignant", ascending=True).iloc[0]

        selected_tiers = [
            ("High Malignancy Score (>=0.85)", high_sample),
            ("Borderline Uncertainty (0.40–0.60)", mid_sample),
            ("Low Malignancy Score (<0.20)", low_sample)
        ]

        for col_idx, (tier_desc, row) in enumerate(selected_tiers):
            ax = row_axes[col_idx]
            img_name = row["image"]
            img_path = Path(LOCAL_IMAGE_DIR) / f"{img_name}.jpg"

            if img_path.exists():
                orig_img = Image.open(img_path).convert("RGB")
                input_tensor = eval_transform(orig_img).unsqueeze(0).to(DEVICE)

                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

                # FIX: Crop black borders for the display grid matching the model input
                cropped_img = CropBlackBorder()(orig_img).resize((IMAGE_SIZE, IMAGE_SIZE))
                cropped_np = np.array(cropped_img) / 255.0
                visualization = show_cam_on_image(cropped_np, grayscale_cam, use_rgb=True)

                ax.imshow(visualization)
            else:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center", transform=ax.transAxes)

            pred_label = "Malignant" if row["pred"] == 1 else "Benign"
            ax.set_title(
                f"{dx} — {tier_desc}\nP(Mal): {row['prob_malignant']:.3f} (Pred: {pred_label})",
                fontsize=9
            )
            ax.axis("off")

    fig.suptitle(
        "Grad-CAM Confidence Grid Across Key Diagnoses (MEL, BCC, NV, BKL)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM confidence grid: {save_path}")


def generate_lime_explanation(model, sample_row, save_path):
    """Generates a LIME superpixel attribution explanation for a single targeted image."""
    img_name = sample_row["image"]
    img_path = Path(LOCAL_IMAGE_DIR) / f"{img_name}.jpg"
    orig_img = Image.open(img_path).convert("RGB")

    # Crop black borders for LIME so perturbations apply to the cropped view
    cropped_img = CropBlackBorder()(orig_img).resize((IMAGE_SIZE, IMAGE_SIZE))
    orig_np = np.array(cropped_img)

    def batch_predict(images):
        model.eval()
        _, eval_transform = get_transforms(IMAGE_SIZE)
        batch = torch.stack([eval_transform(Image.fromarray(img.astype('uint8'))) for img in images])
        with torch.no_grad():
            logits = model(batch.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        orig_np,
        batch_predict,
        top_labels=2,
        hide_color=0,
        num_samples=250
    )

    temp, mask = explanation.get_image_and_mask(
        label=1, positive_only=False, num_features=5, hide_rest=False
    )
    img_bound = mark_boundaries(temp / 255.0, mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_bound)
    ax.set_title(
        f"LIME Attribution (Ground Truth: {sample_row['diagnosis']}, P(Mal)={sample_row['prob_malignant']:.2f})",
        fontsize=10
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LIME explanation: {save_path}")


if __name__ == "__main__":
    set_seed()

    # 1. Load pre-computed predictions CSV
    pred_csv_path = RUN_DIR / f"per_image_predictions_{SPLIT_TO_EVALUATE}.csv"
    if not pred_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {pred_csv_path}. Please run multiclass_analysis_2.py first "
            f"to generate per-image predictions."
        )

    print(f"Loading pre-computed predictions from: {pred_csv_path}")
    audit_df = pd.read_csv(pred_csv_path)

    # 2. Tag images with border presence
    print("Checking images for dark/black dermatoscope borders...")
    audit_df["has_border"] = audit_df["image"].apply(
        lambda x: has_black_borders(Path(LOCAL_IMAGE_DIR) / f"{x}.jpg")
    )

    audit_csv_path = RUN_DIR / f"interpretability_audit_{SPLIT_TO_EVALUATE}.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"Saved updated audit log: {audit_csv_path}")

    # 3. Filter specific test subsets
    tp_df = audit_df[(audit_df["BinaryTarget"] == 1) & (audit_df["pred"] == 1)]
    fp_df = audit_df[(audit_df["BinaryTarget"] == 0) & (audit_df["pred"] == 1)]
    border_df = audit_df[audit_df["has_border"] == True]
    borderless_df = audit_df[audit_df["has_border"] == False]

    # 4. Load model weights and generate plots
    if not USE_METADATA_ONLY:
        print("\nLoading model for feature attribution...")
        model = load_model(RUN_DIR, MODEL_NAME_TO_LOAD)

        print("\nGenerating targeted Grad-CAM visualization grids...")
        generate_gradcam_grid(model, tp_df, MODEL_NAME_TO_LOAD, RUN_DIR / f"gradcam_true_positives_{SPLIT_TO_EVALUATE}.png")
        generate_gradcam_grid(model, fp_df, MODEL_NAME_TO_LOAD, RUN_DIR / f"gradcam_false_positives_{SPLIT_TO_EVALUATE}.png")
        generate_gradcam_grid(model, border_df, MODEL_NAME_TO_LOAD, RUN_DIR / f"gradcam_with_borders_{SPLIT_TO_EVALUATE}.png")
        generate_gradcam_grid(model, borderless_df, MODEL_NAME_TO_LOAD, RUN_DIR / f"gradcam_borderless_{SPLIT_TO_EVALUATE}.png")

        print("\nGenerating Grad-CAM 4x3 confidence grid (MEL, BCC, NV, BKL)...")
        generate_gradcam_confidence_grid(model, audit_df, MODEL_NAME_TO_LOAD, RUN_DIR / f"gradcam_confidence_grid_{SPLIT_TO_EVALUATE}.png")

        target_lime_row = fp_df.iloc[0] if len(fp_df) > 0 else audit_df.iloc[0]
        print("\nGenerating targeted LIME explanation...")
        generate_lime_explanation(model, target_lime_row, RUN_DIR / f"lime_sample_{SPLIT_TO_EVALUATE}.png")

    print("\nInterpretability audit script completed successfully!")