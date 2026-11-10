"""Core training and evaluation loop. Model-agnostic — works with anything
from models.py."""

import math
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from losses import MelanomaAwareFocalLoss

from config import (DEVICE, EPOCHS, LR, LR_FINETUNE, DAMPEN_WEIGHTING,
                    WEIGHT_DECAY, WARMUP_EPOCHS, WEIGHT_SQUARE, CUSTOM_LOSS,
                    LAMBDA_SCHEDULER, POWER
                    )


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, use_scheduler=True,
          use_class_weights=True, early_stopping_patience=None):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    running_time = 0

    if use_class_weights and not CUSTOM_LOSS:
        target_col = train_loader.dataset.target_col
        train_targets = train_loader.dataset.df[target_col].values
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_targets),
            y=train_targets,
        )
        if DAMPEN_WEIGHTING:
            class_weights = np.sqrt(class_weights)
        elif WEIGHT_SQUARE:
            class_weights = class_weights ** POWER
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"Using class weights: {dict(zip(np.unique(train_targets), class_weights.cpu().numpy().round(3)))}")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

    # Tests with custom loss function, wasn't effective
    if CUSTOM_LOSS:
        criterion = MelanomaAwareFocalLoss()
        print("Using custom loss")

    # If using a LR scheduler
    scheduler = None
    if use_scheduler:
        if LAMBDA_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )

    # Early stopping state
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None
    epochs_without_improvement = 0

    # Set up history tracking
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": [], "lr": [], "running_time": []}

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            if len(batch) == 3:
                inputs, labels, diagnosis = batch
            else:
                inputs, labels = batch
                diagnosis = None

            if CUSTOM_LOSS and epoch == 1 and total == 0:
                bad = (
                        diagnosis is None
                        or torch.is_tensor(diagnosis)
                        or all(criterion.boost_map.get(str(d), 1.0) == 1.0 for d in diagnosis)
                )
                if bad:
                    print("WARNING: boosts not applying. diagnosis sample:",
                          None if diagnosis is None else diagnosis[:5])

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            if CUSTOM_LOSS:
                loss = criterion(outputs, labels, diagnosis_labels=diagnosis)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate current model on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Check LR scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if LAMBDA_SCHEDULER:
                scheduler.step()
            else:
                scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        running_time += epoch_time

        # Track the best model seen so far
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

    # Restore the best-val-loss weights (not necessarily the last epoch's) before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored model weights from epoch {best_epoch} (best val_loss={best_val_loss:.4f}).")

    actual_epochs = len(history["train_loss"])
    history["meta"] = {
        "requested_epochs": epochs,
        "actual_epochs": actual_epochs,
        "early_stopped": actual_epochs < epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    return history

# Eval loop
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, labels, diagnosis = batch
            else:
                inputs, labels = batch
                diagnosis = None

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            if CUSTOM_LOSS:
                loss = criterion(outputs, labels, diagnosis_labels=diagnosis)
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def get_lr(epoch):
    warmup = WARMUP_EPOCHS
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / (EPOCHS - warmup)
    lr_min = LR_FINETUNE / LR
    return lr_min + 0.5 * (1 - lr_min) * (1 + math.cos(math.pi * t))