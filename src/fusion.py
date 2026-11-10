"""Multimodal fusion: a saved image CNN + a saved metadata MLP -> one binary
(Benign vs Malignant/Pre-Malignant) prediction.

How it works
------------
1. Each trained model is loaded from its run folder (config.json + model_weights.pt).
2. Its final classification layer is removed, so it outputs a feature vector
   instead of class scores (the CNN's image embedding, the MLP's metadata embedding).
3. The two embeddings are normalised, concatenated, and fed to a small new head
   that is trained to make the final prediction.

Encoders can be frozen for the first few epochs (so the new head learns on top of
stable features) and then fine-tuned together at a much lower learning rate.
Nothing here is hard-coded to a particular architecture: the model name, output
size and feature sizes are all read from the saved runs.
"""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from config import DEVICE, WEIGHT_DECAY, DAMPEN_WEIGHTING, WEIGHT_SQUARE, POWER
from dataset import MetadataLesionDataset
from models import get_model


# ---------------------------------------------------------------------------
# Loading the saved encoders
# ---------------------------------------------------------------------------
def _linear_weights(state):
    """(key, tensor) for every 2-D weight (= nn.Linear) in a state dict, in
    registration order. Conv weights are 4-D and norm weights 1-D, so they are skipped."""
    return [(k, v) for k, v in state.items() if k.endswith("weight") and v.ndim == 2]


def strip_head(model):
    """Replaces the model's final nn.Linear (its classification layer) with
    Identity, in place, so the model outputs features rather than class scores.
    Works for any architecture whose last Linear layer is the classifier."""
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError("No nn.Linear layer found to strip — this architecture isn't supported.")
    *path, leaf = last_name.split(".")
    parent = model
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, nn.Identity())
    return last_name


def build_encoder(run_dir, example_input, expect_metadata):
    """Loads a trained run and returns (encoder, feature_dim, run_config).

    example_input: a small real batch (images or metadata rows) used once to
    measure how many features the encoder outputs.
    expect_metadata: True for the MLP run, False for the CNN run — used to catch
    the two RUN_DIRs being swapped or pointed at the wrong kind of run."""
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    state = torch.load(run_dir / "model_weights.pt", map_location="cpu")

    is_metadata_run = bool(cfg.get("use_metadata_only", False))
    if is_metadata_run != expect_metadata:
        want = "a metadata-only (MLP)" if expect_metadata else "an image (CNN)"
        raise ValueError(f"{run_dir} is not {want} run (use_metadata_only={is_metadata_run}).")

    linear = _linear_weights(state)
    num_classes = linear[-1][1].shape[0]  # whatever the run was trained to output
    kwargs = {}
    if is_metadata_run:
        input_dim = linear[0][1].shape[1]
        if input_dim != example_input.shape[1]:
            raise ValueError(
                f"The MLP in {run_dir.name} expects {input_dim} metadata features but the metadata "
                f"CSVs give {example_input.shape[1]}. Use the same train/val metadata CSVs the MLP was trained on."
            )
        kwargs["input_dim"] = input_dim

    model = get_model(cfg["model"], num_classes=num_classes, **kwargs)
    model.load_state_dict(state)
    strip_head(model)
    model.eval()
    with torch.no_grad():
        feature_dim = torch.flatten(model(example_input), 1).shape[1]
    return model, feature_dim, cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, image_encoder, meta_encoder, image_dim, meta_dim,
                 hidden=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.image_encoder = image_encoder
        self.meta_encoder = meta_encoder
        # Normalise each embedding so neither modality dominates just by scale
        self.image_norm = nn.LayerNorm(image_dim)
        self.meta_norm = nn.LayerNorm(meta_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim + meta_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.encoders_frozen = False

    def forward(self, image, meta):
        f_img = torch.flatten(self.image_encoder(image), 1)
        f_meta = torch.flatten(self.meta_encoder(meta), 1)
        return self.head(torch.cat([self.image_norm(f_img), self.meta_norm(f_meta)], dim=1))

    def encoder_parameters(self):
        return list(self.image_encoder.parameters()) + list(self.meta_encoder.parameters())

    def head_parameters(self):
        return (list(self.image_norm.parameters()) + list(self.meta_norm.parameters())
                + list(self.head.parameters()))

    def set_encoders_frozen(self, frozen):
        self.encoders_frozen = frozen
        for p in self.encoder_parameters():
            p.requires_grad = not frozen

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:  # frozen encoders keep dropout off / batchnorm stats fixed
            self.image_encoder.eval()
            self.meta_encoder.eval()
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    """Yields (image, metadata_features, label, diagnosis) for each row.
    df needs the columns: image, diagnosis, BinaryTarget, plus the metadata
    feature columns the MLP was trained on (same CSV schema as val_metadata.csv)."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        # Reuse the metadata dataset so feature selection/order is identical to MLP training
        meta = MetadataLesionDataset(self.df, classification_mode="binary")
        self.features = meta.features
        self.feature_cols = meta.feature_cols
        self.labels = meta.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, f"{row['image']}.jpg")).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.features[idx], self.labels[idx], row["diagnosis"]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def make_criterion(train_labels, use_class_weights):
    """Same loss setup as train.py: label smoothing + optional class weights."""
    if not use_class_weights:
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    if DAMPEN_WEIGHTING:
        weights = np.sqrt(weights)
    elif WEIGHT_SQUARE:
        weights = weights ** POWER
    print(f"Using class weights: {dict(zip(classes, weights.round(3)))}")
    return nn.CrossEntropyLoss(label_smoothing=0.1,
                               weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))


@torch.no_grad()
def predict(model, loader, criterion=None):
    """Returns (P(malignant) per row, true labels, mean loss or None), in loader order."""
    model.eval()
    probs, labels_all, running_loss, total = [], [], 0.0, 0
    for images, meta, labels, _ in loader:
        outputs = model(images.to(DEVICE), meta.to(DEVICE))
        if criterion is not None:
            running_loss += criterion(outputs, labels.to(DEVICE)).item() * labels.size(0)
        total += labels.size(0)
        probs.append(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        labels_all.append(labels.numpy())
    loss = running_loss / total if criterion is not None else None
    return np.concatenate(probs), np.concatenate(labels_all), loss


def train_fusion(model, train_loader, val_loader, epochs, lr_head, lr_encoder,
                 freeze_encoder_epochs=3, use_scheduler=True, use_class_weights=True,
                 early_stopping_patience=None):
    """Trains the fusion model in place and returns a history dict shaped like
    train.py's (so metrics.save_run can plot/save it). Restores the best
    val-loss weights before returning."""
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        [{"params": model.head_parameters(), "lr": lr_head},
         {"params": model.encoder_parameters(), "lr": lr_encoder}],
        weight_decay=WEIGHT_DECAY,
    )
    criterion = make_criterion(train_loader.dataset.labels.numpy(), use_class_weights)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
                 if use_scheduler else None)

    best_val_loss, best_state, best_epoch, stale, running_time = float("inf"), None, None, 0, 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": [], "lr": [], "running_time": []}

    for epoch in range(1, epochs + 1):
        frozen = epoch <= freeze_encoder_epochs
        model.set_encoders_frozen(frozen)
        model.train()
        start = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for images, meta, labels, _ in train_loader:
            images, meta, labels = images.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_probs, val_labels, val_loss = predict(model, val_loader, criterion)
        val_acc = float(((val_probs >= 0.5).astype(int) == val_labels).mean())
        train_loss, train_acc = running_loss / total, correct / total

        current_lr = optimizer.param_groups[0]["lr"]  # head LR
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - start
        running_time += epoch_time

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1

        print(f"Epoch {epoch}/{epochs} [{'encoders frozen' if frozen else 'end-to-end'}] | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | lr={current_lr:.2e} | "
              f"epoch_time={epoch_time:.1f} | best_val_loss={best_val_loss:.4f} (epoch {best_epoch})")

        for key, value in zip(history, [train_loss, train_acc, val_loss, val_acc,
                                        epoch_time, current_lr, running_time]):
            history[key].append(value)

        if early_stopping_patience is not None and stale >= early_stopping_patience:
            print(f"\nEarly stopping: val_loss hasn't improved for {early_stopping_patience} epochs "
                  f"(best was epoch {best_epoch}). Stopping at epoch {epoch}/{epochs}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored model weights from epoch {best_epoch} (best val_loss={best_val_loss:.4f}).")

    actual = len(history["train_loss"])
    history["meta"] = {"requested_epochs": epochs, "actual_epochs": actual,
                       "early_stopped": actual < epochs, "best_epoch": best_epoch,
                       "best_val_loss": best_val_loss}
    return history
