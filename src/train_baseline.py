import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
IMAGE_SIZE = 224
SAMPLE_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LesionDataset(Dataset):
    LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    # DF is the metadata and the labels, so need to pair it with the image directory
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = self.LABEL_COLS

    def __len__(self):
        return len(self.df)

    # When getting the item, use the DF for the label and the image column to find the image path in the folder
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["target"])  # use the pre-computed encoding directly
        return image, label


# ---------------------------------------------------------------------------
# Simple Model
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Split Dataset Safely
# ---------------------------------------------------------------------------
def make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE, random_state=42):
    """
    Stratified sample by diagnosis, grouped so no lesion is split across
    the sample
    """
    frac = sample_size / len(df_merged)
    sample = df_merged.groupby("diagnosis", group_keys=False).sample(
        frac=frac, random_state=random_state
    )
    return sample.reset_index(drop=True)


def train_val_split(sample_df, val_frac=0.2, random_state=42):
    from sklearn.model_selection import GroupShuffleSplit

    # GroupShuffleSplit will make sure that the rows sharing the same lesion_id_filled end stay in the same group (either train or val)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(gss.split(sample_df, groups=sample_df["lesion_id_filled"]))

    return sample_df.iloc[train_idx].reset_index(drop=True), sample_df.iloc[val_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"time={epoch_time:.1f}"
        )


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.perf_counter()
    df_merged = pd.read_csv("../data/processed/df_merged.csv")

    sample_df = make_tiny_sample(df_merged, sample_size=SAMPLE_SIZE)
    train_df, val_df = train_val_split(sample_df)

    # Verify there are no common lesion_id's in both training and validation splits
    overlap = set(train_df['lesion_id_filled']) & set(val_df['lesion_id_filled'])
    assert len(overlap) == 0, f"Lesion leakage detected: {len(overlap)} lesions in both train and val"

    print(f"Sample size: {len(sample_df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print("Class counts (sample):")
    print(sample_df["diagnosis"].value_counts())

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LesionDataset(train_df, LOCAL_IMAGE_DIR, transform=transform)
    val_dataset = LesionDataset(val_df, LOCAL_IMAGE_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=len(train_dataset.classes))
    print(model)

    train(model, train_loader, val_loader)

    end_time = time.perf_counter()
    print(f"Training took {end_time - start_time} seconds")