"""Shared constants and set_seed() used across the project. Import from here rather than
redefining these values in individual modules, so changing a setting means
editing exactly one place."""
import torch
import random
import numpy as np

SEED = 42

def set_seed(seed=SEED):
    """Fixes random sources so training is reproducible — same initial
    weights, same batch order, same augmentation randomness, every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

MODEL_NAME_LIST = ["SimpleCNN", "CustomCNN", "ResNet18_Frozen", "ResNet18_Tune"]

# Paths — adjust these to match your project layout
LOCAL_IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
PROCESSED_DATA_PATH = "../data/processed/df_merged.csv"

# Data / sampling
SAMPLE_SIZE = 10000       # total images in sample
IMAGE_SIZE = 224
USE_SMALL_SAMPLE = True
CLASSIFICATION_MODE = 'binary'

# Training
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
LR_FINETUNE = 1e-5
USE_SCHEDULER = True
USE_CLASS_WEIGHTS = True
EARLY_STOPPING_PATIENCE = 7
DAMPEN_WEIGHTING = True

# Model
MODEL_NAME = MODEL_NAME_LIST[1]

# Fixed label order — must match how 'target' was built in data_prep.py.
# Kept here (not in dataset.py) so any module can reference it without
# importing the dataset class.
LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
BINARY_CLASSES = ["Benign", "Malignant/Pre-Malignant"]

TARGET_COL_BY_MODE = {
    "multiclass": 'target',
    "binary": "BinaryTarget"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_TO_FULLNAME = {
    "MEL": "Melanoma",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "SCC": "Squamous Cell Carcinoma",
    "NV": "Melanocytic Nevus",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion"
}