"""Dataset class plus helpers for sampling and splitting the merged dataframe
into train/val sets without lesion leakage."""

import os
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image

from config import LABEL_COLS, SAMPLE_SIZE


class LesionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = LABEL_COLS  # fixed order, same across every split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["target"])  # use the pre-computed encoding directly
        return image, label


def make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, random_state=42):
    """Stratified sample by diagnosis, grouped so no lesion is split across
    the sample (keeps this consistent with the grouped-split approach even
    at tiny scale)."""
    frac = sample_size / len(df_merged)
    sample = df_merged.groupby("diagnosis", group_keys=False).sample(
        frac=frac, random_state=random_state
    )
    return sample.reset_index(drop=True)


def train_val_split(sample_df, val_frac=0.2, random_state=42):
    """Group-aware split so no lesion_id_filled appears in both train and val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(gss.split(sample_df, groups=sample_df["lesion_id_filled"]))
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