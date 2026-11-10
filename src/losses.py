import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ALPHA, GAMMA
)


# Per-diagnosis loss multipliers. With alpha=0.5 these are the only thing
# steering class emphasis, so read them as weights relative to NV (1.0).
# Diagnoses not listed here (DF, VASC) default to 1.0.
BOOST_MAP = {
    "MEL": 2.5,  # Main target: melanoma recall (false negatives)
    "BCC": 1.6,
    "AK": 1.4,
    "SCC": 1.4,
    "NV": 1.4,   # Already plentiful; gets plenty of gradient from volume
    "BKL": 1.6,  # Hardest benign class (highest false-positive rate)
}


class MelanomaAwareFocalLoss(nn.Module):
  """Binary focal loss with per-diagnosis boost multipliers.

  Per-sample loss = alpha_t * (1 - p_t)^gamma * BCE * boost[diagnosis]

  Args:
    alpha: Weight on the malignant class (1 - alpha on benign). 0.5 is
      neutral, so the boosts are the only class steering.
    gamma: Focal exponent. Lower to 1.0 if training is slow or unstable.
    reduction: "mean", "sum", or "none".
    boost_map: Dict of diagnosis name -> multiplier. Defaults to BOOST_MAP.
    class_names: Optional list mapping integer diagnosis indices to names.
      Required only if you pass diagnosis_labels as an integer tensor.
  """

  def __init__(
      self,
      alpha=ALPHA,
      gamma=GAMMA,
      reduction="mean",
      boost_map=None,
      class_names=None,
  ):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction
    self.boost_map = dict(BOOST_MAP if boost_map is None else boost_map)
    self.class_names = list(class_names) if class_names is not None else None
    self._checked_boosts = False

  def _to_names(self, diagnosis_labels):
    """Convert whatever the dataloader gives us into a list of strings."""
    if torch.is_tensor(diagnosis_labels):
      if self.class_names is None:
        raise ValueError(
            "diagnosis_labels is a tensor of ints; pass class_names=[...] "
            "to MelanomaAwareFocalLoss so indices can be mapped to names."
        )
      return [self.class_names[int(i)] for i in diagnosis_labels.tolist()]
    return [str(d) for d in diagnosis_labels]

  def _boost_tensor(self, diagnosis_labels, device):
    names = self._to_names(diagnosis_labels)
    multipliers = [self.boost_map.get(n, 1.0) for n in names]

    # One-time sanity check: if every multiplier on the first batch is 1.0,
    # the labels probably don't match the BOOST_MAP keys (e.g. wrong format).
    if not self._checked_boosts:
      self._checked_boosts = True
      if all(m == 1.0 for m in multipliers) and any(
          v != 1.0 for v in self.boost_map.values()
      ):
        warnings.warn(
            "All boost multipliers in the first batch are 1.0. Check that "
            "diagnosis labels match BOOST_MAP keys. "
            f"Example labels seen: {names[:5]}"
        )

    return torch.tensor(multipliers, device=device, dtype=torch.float32)

  def forward(self, inputs, targets, diagnosis_labels=None):
    # Use the malignant logit if the model outputs 2 nodes
    if inputs.ndim == 2 and inputs.size(1) == 2:
      inputs = inputs[:, 1:2]

    targets = targets.float()
    if targets.ndim == 1:
      targets = targets.unsqueeze(1)

    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    # p_t = probability assigned to the true class (numerically stable form)
    p_t = torch.exp(-ce_loss)
    focal_weight = (1 - p_t) ** self.gamma

    alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
    loss = alpha_weight * focal_weight * ce_loss

    if diagnosis_labels is not None:
      boost = self._boost_tensor(diagnosis_labels, inputs.device)
      loss = loss * boost.unsqueeze(1)

    if self.reduction == "mean":
      return loss.mean()
    elif self.reduction == "sum":
      return loss.sum()
    return loss