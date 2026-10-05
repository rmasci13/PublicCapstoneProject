"""Core training and evaluation loop. Model-agnostic — works with anything
from models.py."""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import DEVICE, EPOCHS, LR, DAMPEN_WEIGHTING


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, use_scheduler=True,
          use_class_weights=True, early_stopping_patience=None):
    """Trains model in place and returns per-epoch history for plotting/tuning.

    early_stopping_patience: if set (e.g. 7), stops training once val_loss
    hasn't improved for that many consecutive epochs, and restores the
    model's best-val-loss weights before returning (rather than leaving it
    at whatever the last, possibly-overfit, epoch produced). None disables
    early stopping and trains the full epoch count, same as before.
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    running_time = 0

    if use_class_weights:
        target_col = train_loader.dataset.target_col
        train_targets = train_loader.dataset.df[target_col].values
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_targets),
            y=train_targets,
        )
        if DAMPEN_WEIGHTING:
            class_weights = np.sqrt(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"Using class weights: {dict(zip(np.unique(train_targets), class_weights.cpu().numpy().round(3)))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    # Early stopping state
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None
    epochs_without_improvement = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": [], "lr": [], "running_time": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        running_time += epoch_time

        # Track the best model seen so far, regardless of whether early
        # stopping is enabled — costs nothing and means you always have
        # the option to use the best-val-loss weights instead of the
        # final epoch's, even on a full, non-early-stopped run.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={current_lr:.2e} | "
            f"epoch_time={epoch_time:.1f} | "
            f"running_time={running_time:.1f} | "
            f"best_val_loss={best_val_loss:.4f} (epoch {best_epoch})"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(current_lr)
        history["running_time"].append(running_time)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping: val_loss hasn't improved for {early_stopping_patience} epochs "
                  f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f}). Stopping at epoch {epoch}/{epochs}.")
            break

    # Restore the best-val-loss weights (not necessarily the last epoch's)
    # before returning, so the model you get back is the one that
    # generalized best, not whatever overfitting may have happened after.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored model weights from epoch {best_epoch} (best val_loss={best_val_loss:.4f}).")

    return history


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total