"""Dataset class plus helpers for sampling and splitting the merged dataframe
into train/val sets without lesion leakage."""

import os
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from PIL import Image

from config import LABEL_COLS, SAMPLE_SIZE, CLASSIFICATION_MODE, TARGET_COL_BY_MODE, BINARY_CLASSES


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
        return image, label


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
    """Returns (train_transform, eval_transform). Augmentation only applies
    to training; val/test stay deterministic (resize + normalize only)."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform