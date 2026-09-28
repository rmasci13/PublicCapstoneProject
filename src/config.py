"""Shared constants used across the project. Import from here rather than
redefining these values in individual modules, so changing a setting means
editing exactly one place."""
import torch

MODEL_NAME_LIST = ["SimpleCNN", "CustomCNN"]

# Paths — adjust these to match your project layout
LOCAL_IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
PROCESSED_DATA_PATH = "../data/processed/df_merged.csv"

# Data / sampling
SAMPLE_SIZE = 100        # total images in sample
IMAGE_SIZE = 224
USE_SMALL_SAMPLE = True

# Training
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
USE_SCHEDULER = True

# Model
MODEL_NAME = MODEL_NAME_LIST[0]

# Fixed label order — must match how 'target' was built in data_prep.py.
# Kept here (not in dataset.py) so any module can reference it without
# importing the dataset class.
LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")