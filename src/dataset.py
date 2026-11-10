"""Dataset class plus helpers for sampling and splitting the merged dataframe
into train/val sets without lesion leakage."""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from PIL import Image
from losses import MelanomaAwareFocalLoss
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from border_crop import CropBlackBorder


from config import LABEL_COLS, SAMPLE_SIZE, CLASSIFICATION_MODE, TARGET_COL_BY_MODE, BINARY_CLASSES, HAS_CROP


class LesionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, classification_mode=CLASSIFICATION_MODE):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classification_mode = classification_mode
        self.target_col = TARGET_COL_BY_MODE[classification_mode]
        self.classes = BINARY_CLASSES if classification_mode == 'binary' else LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row[self.target_col])
        diagnosis = row['diagnosis']
        return image, label, diagnosis


class MetadataLesionDataset(Dataset):
    def __init__(self, df, classification_mode=CLASSIFICATION_MODE):
        self.df = df.reset_index(drop=True)
        self.classification_mode = classification_mode
        self.target_col = TARGET_COL_BY_MODE[classification_mode]
        self.classes = BINARY_CLASSES if classification_mode == 'binary' else LABEL_COLS

        # Columns to skip (identifiers, target labels, raw text)
        exclude_cols = {
            'image', 'lesion_id', 'lesion_id_filled', 'diagnosis',
            'risk_level', 'target', 'BinaryTarget', 'sex', 'anatom_site_general'
        }

        # Everything remaining in the CSV becomes a feature tensor column
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        self.features = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df[self.target_col].values, dtype=torch.long)
        self.diagnoses = self.df['diagnosis'].values if 'diagnosis' in self.df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.diagnoses is not None:
            diagnosis = self.diagnoses[idx]
            return x, y, diagnosis

        return x, y

def make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, stratify_col="diagnosis", random_state=42):
    """Stratified sample by stratify_col, grouped so no lesion is split
    across the sample. Use stratify_col='diagnosis' for multiclass mode,
    'risk_level' for binary mode."""
    frac = sample_size / len(df_merged)
    sample = df_merged.groupby(stratify_col, group_keys=False).sample(
        frac=frac, random_state=random_state
    )
    return sample.reset_index(drop=True)


def train_val_split(sample_df, val_frac=0.2, stratify_col="diagnosis", random_state=42):
    """Group-aware, stratified split. stratify_col should match whatever
    make_tiny_sample used, so train/val proportions line up with how the
    sample itself was balanced."""
    n_splits = round(1 / val_frac)  # val_frac=0.2 -> 5 folds, take 1 as val
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf.split(sample_df, y=sample_df[stratify_col], groups=sample_df["lesion_id_filled"]))
    train_df = sample_df.iloc[train_idx].reset_index(drop=True)
    val_df = sample_df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df["lesion_id_filled"]) & set(val_df["lesion_id_filled"])
    assert len(overlap) == 0, f"Lesion leakage detected: {len(overlap)} lesions in both train and val"

    return train_df, val_df


def get_transforms(image_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = []
    if HAS_CROP:
        base = [CropBlackBorder(), transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    else:
        base = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]

    train_transform = transforms.Compose(
        base + [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.RandomAffine(degrees=180, translate=(0.05, 0.05), shear=5,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(124, 116, 104)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"),
    ])

    eval_transform = transforms.Compose(
        base + [
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


class PadToSquare:
    """Resizes image keeping aspect ratio so max side equals target_size, then pads the rest with black."""

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        # Find which dimension is larger and scale it to target_size
        scale = self.target_size / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # Resize preserving aspect ratio
        img = TF.resize(img, (new_h, new_w))

        # Calculate padding needed to reach target_size x target_size
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        # Pad with zeros (pure black)
        return TF.pad(img, padding, fill=0, padding_mode='constant')