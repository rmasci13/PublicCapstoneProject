"""Grad-CAM and LIME for a saved FUSION run (the folder fusion_e2e.py writes).

Same outputs as evaluate_interpretability.py, but for the fusion model, which needs an image AND that image's
metadata. A small wrapper feeds each image's metadata to the model while Grad-CAM / LIME vary the image, so the
heatmaps show what the IMAGE branch contributes to the fused malignant score for that lesion.

Needs in RUN_DIR (all written by fusion_e2e.py): model_weights.pt, config.json, per_image_predictions_val.csv.
Needs fusion_e2e.py and the usual project files (config.py, dataset.py, models.py, border_crop.py) beside it.

Usage: set RUN_DIR below, then  python evaluate_interpretability_fusion.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import mark_boundaries

from border_crop import CropBlackBorder
from config import DEVICE, HAS_CROP, LOCAL_IMAGE_DIR, IMAGE_SIZE as CONFIG_IMAGE_SIZE, set_seed
from dataset import MetadataLesionDataset, get_transforms
from fusion_e2e import FusionModel, strip_head
from models import get_model

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
RUN_DIR = Path("results/BEST-2026-11-08_0142_FUSION-E2E_EfficientNetB2_Tune+ComplexMLP-LinearHead")
SPLIT_TO_EVALUATE = "val"      # fusion runs only save val predictions
LIME_SAMPLES = 1000            # perturbed copies of the image; 250 was too few for stable weights (slower: ~4x)
LIME_SEGMENTS = 40             # approx. number of image regions (SLIC superpixels); fewer = bigger, more stable regions
LIME_HIDE_COLOR = None         # None = hidden regions are filled with their own mean colour; 0 = solid black
LIME_BATCH = 32                # images LIME sends through the model at once (lower if GPU memory is tight)
LIME_SEED = 0                  # fixed seed; change it and re-run to see whether the highlighted regions are stable
LIME_NUM_REGIONS = 5           # how many of the highest-weight regions are drawn (3 is cleaner, the rest are mostly noise)
# Which cases get a LIME picture (one randomly chosen image each, fixed by LIME_SEED). Options:
# "true_positive", "false_positive", "true_negative", "false_negative"
LIME_CASES = ["true_positive", "false_positive", "true_negative"]
SANITY_CHECK_IMAGES = 8        # re-predict this many val images and compare with per_image_predictions_val.csv


# ---------------------------------------------------------------------------
# Loading the saved fusion model
# ---------------------------------------------------------------------------
def build_encoder(model_name, example_input, input_dim=None):
    kwargs = {"input_dim": input_dim} if input_dim is not None else {}
    model = get_model(model_name, num_classes=2, **kwargs)   # weights come from the fusion checkpoint below
    strip_head(model)
    model.eval()
    with torch.no_grad():
        dim = int(model(example_input).shape[1])
    return model, dim


def load_fusion(run_dir):
    cfg = json.load(open(run_dir / "config.json"))
    image_size = int(cfg.get("IMAGE_SIZE", CONFIG_IMAGE_SIZE))
    n_meta = len(cfg["metadata_features"])
    cnn, cnn_dim = build_encoder(cfg["CNN_MODEL"], torch.zeros(2, 3, image_size, image_size))
    mlp, mlp_dim = build_encoder(cfg["MLP_MODEL"], torch.zeros(2, n_meta), input_dim=n_meta)
    model = FusionModel(cnn, cnn_dim, mlp, mlp_dim, hidden=cfg["HEAD_HIDDEN"], dropout=cfg["HEAD_DROPOUT"])
    model.load_state_dict(torch.load(run_dir / "model_weights.pt", map_location="cpu"))   # fails loudly on mismatch
    model.to(DEVICE).eval()
    return model, cfg, image_size


class FixedMetadataModel(nn.Module):
    """Lets Grad-CAM / LIME call model(images) on the fusion model: every image in the batch gets the same
    metadata vector, set with set_meta() before explaining each lesion."""

    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion
        self.meta = None

    def set_meta(self, meta_row):
        self.meta = meta_row.to(DEVICE).float().view(1, -1)

    def forward(self, x):
        return self.fusion(x, self.meta.expand(x.shape[0], -1))


def get_target_layer(image_encoder, model_name):
    """Final convolutional feature layer of the IMAGE encoder (same rule as evaluate_interpretability.py)."""
    if "EfficientNet" in model_name:
        return image_encoder.features[-1]
    if "ResNet" in model_name:
        return image_encoder.layer4[-1]
    convs = [m for m in image_encoder.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else None


def has_black_borders(img_path, threshold=35):
    """Fast check: harsh dark/black dermatoscope borders in the image corners."""
    try:
        gray = np.mean(np.array(Image.open(img_path).convert("RGB")), axis=2)
        h, w = gray.shape
        corners = [gray[:h // 10, :w // 10], gray[:h // 10, -w // 10:], gray[-h // 10:, :w // 10],
                   gray[-h // 10:, -w // 10:]]
        return np.mean([np.mean(c) for c in corners]) < threshold
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def cam_overlay(cam, wrapper, row, meta_lookup, eval_tf, image_size, image_dir):
    orig = Image.open(Path(image_dir) / f"{row['image']}.jpg").convert("RGB")
    wrapper.set_meta(meta_lookup(row["image"]))
    tensor = eval_tf(orig).unsqueeze(0).to(DEVICE)
    gray = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0]
    cropped = CropBlackBorder()(orig).resize((image_size, image_size)) if HAS_CROP else orig.resize((image_size, image_size))
    return show_cam_on_image(np.array(cropped) / 255.0, gray, use_rgb=True)


def generate_gradcam_grid(cam, wrapper, df_subset, meta_lookup, eval_tf, image_size, image_dir, save_path):
    n = min(len(df_subset), 4)
    if n == 0:
        print(f"No samples for {save_path.name}")
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (_, row) in zip(axes, df_subset.head(n).iterrows()):
        ax.imshow(cam_overlay(cam, wrapper, row, meta_lookup, eval_tf, image_size, image_dir))
        ax.set_title(f"Ground Truth: {row['diagnosis']}\nPred: {'Mal' if row['pred'] == 1 else 'Ben'} "
                     f"(P={row['prob_malignant']:.2f})\nBorder: {'Yes' if row['has_border'] else 'No'}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM grid: {save_path}")


def generate_gradcam_confidence_grid(cam, wrapper, df, meta_lookup, eval_tf, image_size, image_dir, save_path):
    """4x3 grid: rows MEL, BCC, NV, BKL; columns High (>=0.85), Borderline (0.40-0.60), Low (<0.20)."""
    diagnoses = ["MEL", "BCC", "NV", "BKL"]
    fig, axes = plt.subplots(len(diagnoses), 3, figsize=(14, 16))
    for r, dx in enumerate(diagnoses):
        dx_df = df[df["diagnosis"] == dx]
        if len(dx_df) == 0:
            for c in range(3):
                axes[r][c].axis("off")
                axes[r][c].text(0.5, 0.5, f"No samples for {dx}", ha="center", va="center")
            continue
        high = dx_df[dx_df["prob_malignant"] >= 0.85]
        mid = dx_df[(dx_df["prob_malignant"] >= 0.40) & (dx_df["prob_malignant"] <= 0.60)]
        low = dx_df[dx_df["prob_malignant"] < 0.20]
        picks = [
            ("High Malignancy Score (>=0.85)", high.sample(1).iloc[0] if len(high) else dx_df.sort_values("prob_malignant", ascending=False).iloc[0]),
            ("Borderline Uncertainty (0.40–0.60)", mid.sample(1).iloc[0] if len(mid) else dx_df.iloc[len(dx_df) // 2]),
            ("Low Malignancy Score (<0.20)", low.sample(1).iloc[0] if len(low) else dx_df.sort_values("prob_malignant").iloc[0]),
        ]
        for c, (tier, row) in enumerate(picks):
            ax = axes[r][c]
            if (Path(image_dir) / f"{row['image']}.jpg").exists():
                ax.imshow(cam_overlay(cam, wrapper, row, meta_lookup, eval_tf, image_size, image_dir))
            else:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{dx} — {tier}\nP(Mal): {row['prob_malignant']:.3f} "
                         f"(Pred: {'Malignant' if row['pred'] == 1 else 'Benign'})", fontsize=9)
            ax.axis("off")
    fig.suptitle("Fusion Grad-CAM Confidence Grid (MEL, BCC, NV, BKL)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM confidence grid: {save_path}")


def generate_lime_explanation(wrapper, row, meta_lookup, eval_tf, image_size, image_dir, save_path, case_label=""):
    orig = Image.open(Path(image_dir) / f"{row['image']}.jpg").convert("RGB")
    cropped = CropBlackBorder()(orig).resize((image_size, image_size)) if HAS_CROP else orig.resize((image_size, image_size))
    wrapper.set_meta(meta_lookup(row["image"]))   # metadata stays fixed while LIME perturbs the image

    def batch_predict(images):   # LIME calls this with up to LIME_BATCH images at a time
        wrapper.eval()
        batch = torch.stack([eval_tf(Image.fromarray(im.astype("uint8"))) for im in images])
        with torch.no_grad():
            return torch.softmax(wrapper(batch.to(DEVICE)), dim=1).cpu().numpy()

    segmenter = SegmentationAlgorithm("slic", n_segments=LIME_SEGMENTS, compactness=10, sigma=1)
    explanation = lime_image.LimeImageExplainer(random_state=LIME_SEED).explain_instance(
        np.array(cropped), batch_predict, top_labels=2, hide_color=LIME_HIDE_COLOR, num_samples=LIME_SAMPLES,
        batch_size=LIME_BATCH, segmentation_fn=segmenter, random_seed=LIME_SEED)
    temp, mask = explanation.get_image_and_mask(label=1, positive_only=False, num_features=LIME_NUM_REGIONS,
                                                hide_rest=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(f"LIME {case_label} — {row['image']}\nGround Truth: {row['diagnosis']}, P(Mal)={row['prob_malignant']:.2f} "
                 f"(green = raises malignant score, red = lowers it)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LIME explanation: {save_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":   # required on Windows because of DataLoader workers / spawn
    set_seed()

    print(f"Loading fusion model from {RUN_DIR}")
    fusion, cfg, image_size = load_fusion(RUN_DIR)
    if "HAS_CROP" in cfg and bool(cfg["HAS_CROP"]) != bool(HAS_CROP):
        print(f"WARNING: this run was trained with HAS_CROP={cfg['HAS_CROP']} but config.py now says {HAS_CROP}. "
              f"Set config.HAS_CROP to match, or the images will be processed differently than in training.")
    image_dir = cfg.get("IMAGE_DIR", LOCAL_IMAGE_DIR)
    wrapper = FixedMetadataModel(fusion).eval()
    _, eval_tf = get_transforms(image_size)

    # Metadata lookup: image id -> that image's metadata vector (same columns the MLP was trained on)
    val_df = pd.read_csv(cfg["VAL_METADATA_CSV"])
    meta_ds = MetadataLesionDataset(val_df, classification_mode="binary")
    assert list(meta_ds.feature_cols) == list(cfg["metadata_features"]), \
        "The metadata columns in the val CSV differ from the ones this fusion run was trained with."
    row_of = {img: i for i, img in enumerate(meta_ds.df["image"].astype(str))}
    meta_lookup = lambda img: meta_ds.features[row_of[str(img)]]

    # 1. Pre-computed predictions
    pred_csv = RUN_DIR / f"per_image_predictions_{SPLIT_TO_EVALUATE}.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Could not find {pred_csv}. fusion_e2e.py writes it at the end of a run.")
    audit_df = pd.read_csv(pred_csv)

    # 2. Sanity check: re-predicting images must reproduce the saved probabilities
    check = audit_df.sample(min(SANITY_CHECK_IMAGES, len(audit_df)), random_state=0)
    diffs = []
    with torch.no_grad():
        for _, r in check.iterrows():
            img = eval_tf(Image.open(Path(image_dir) / f"{r['image']}.jpg").convert("RGB")).unsqueeze(0).to(DEVICE)
            wrapper.set_meta(meta_lookup(r["image"]))
            diffs.append(abs(torch.softmax(wrapper(img), 1)[0, 1].item() - r["prob_malignant"]))
    print(f"Sanity check on {len(diffs)} images: max |P(malignant) here - saved| = {max(diffs):.4f}")
    if max(diffs) > 0.01:
        print("WARNING: probabilities don't match the saved predictions. Check config.HAS_CROP, IMAGE_SIZE, the "
              "metadata CSV and that RUN_DIR is the run you think it is before trusting the explanations.")

    # 3. Tag border images, pick subsets
    print("Checking images for dark/black dermatoscope borders...")
    audit_df["has_border"] = audit_df["image"].apply(lambda x: has_black_borders(Path(image_dir) / f"{x}.jpg"))
    audit_df.to_csv(RUN_DIR / f"interpretability_audit_{SPLIT_TO_EVALUATE}.csv", index=False)
    tp_df = audit_df[(audit_df["BinaryTarget"] == 1) & (audit_df["pred"] == 1)]
    fp_df = audit_df[(audit_df["BinaryTarget"] == 0) & (audit_df["pred"] == 1)]
    border_df = audit_df[audit_df["has_border"]]
    borderless_df = audit_df[~audit_df["has_border"]]
    tn_df = audit_df[(audit_df["BinaryTarget"] == 0) & (audit_df["pred"] == 0)]
    fn_df = audit_df[(audit_df["BinaryTarget"] == 1) & (audit_df["pred"] == 0)]

    # 4. Grad-CAM on the image encoder's last conv layer, LIME on the whole image
    target_layer = get_target_layer(fusion.image_encoder, cfg["CNN_MODEL"])
    if target_layer is None:
        raise RuntimeError("Could not find a convolutional layer in the image encoder for Grad-CAM.")
    cam = GradCAM(model=wrapper, target_layers=[target_layer])
    common = dict(meta_lookup=meta_lookup, eval_tf=eval_tf, image_size=image_size, image_dir=image_dir)

    print("\nGenerating targeted Grad-CAM grids...")
    for name, subset in [("true_positives", tp_df), ("false_positives", fp_df),
                         ("with_borders", border_df), ("borderless", borderless_df)]:
        generate_gradcam_grid(cam, wrapper, subset, save_path=RUN_DIR / f"gradcam_{name}_{SPLIT_TO_EVALUATE}.png", **common)

    print("\nGenerating Grad-CAM 4x3 confidence grid...")
    generate_gradcam_confidence_grid(cam, wrapper, audit_df,
                                     save_path=RUN_DIR / f"gradcam_confidence_grid_{SPLIT_TO_EVALUATE}.png", **common)

    lime_cases = {"true_positive": tp_df, "false_positive": fp_df, "true_negative": tn_df, "false_negative": fn_df}
    for case in LIME_CASES:
        subset = lime_cases[case]
        if len(subset) == 0:
            print(f"\nNo images for LIME case '{case}', skipping.")
            continue
        row = subset.sample(1, random_state=LIME_SEED).iloc[0]
        print(f"\nGenerating LIME explanation for {case}: {row['image']} ({row['diagnosis']}, "
              f"P(Mal)={row['prob_malignant']:.2f})...")
        generate_lime_explanation(wrapper, row, save_path=RUN_DIR / f"lime_{case}_{SPLIT_TO_EVALUATE}.png",
                                  case_label=case.replace("_", " "), **common)

    print("\nInterpretability audit completed successfully!")