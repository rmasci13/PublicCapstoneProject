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

# ====================================
#         Metadata Config
# ====================================

USE_METADATA_ONLY = False



#                      0            1              2                    3
MODEL_NAME_LIST = ["SimpleCNN", "CustomCNN", "ResNet18_Frozen", "ResNet18_Tune",
#                             4                      5                    6
                   "EfficientNetB0_Frozen", "EfficientNetB0_Tune", "ResNet50_Tune",
#                          7                  8                    9 (260)
                   "ResNet50V2_Tune", "DenseNet121_Tune", "EfficientNetB2_Tune",
#                            10                     11                     12
                   "EfficientViT_b0_Tune", "EfficientViT_b2_Tune", "EfficientNetB4_Tune",
#                       13                 14                      15
                   "VGG19_Tune", "EfficientNetB2_Partial", "FlatLinearBaseline",
#                      16             17
                   "SimpleMLP", "ComplexMLP"]

# Paths — adjust these to match your project layout
LOCAL_IMAGE_DIR = "../data/raw/train/ISIC_2019_Training_Input"
PROCESSED_DATA_PATH = "../data/processed/df_merged_cleaned.csv"

# Data / sampling
SAMPLE_SIZE = 100      # total images in sample
IMAGE_SIZE = 350
USE_SMALL_SAMPLE = False
CLASSIFICATION_MODE = 'binary'

# Training
BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-5
LR_FINETUNE = 5e-6
USE_SCHEDULER = True
USE_CLASS_WEIGHTS = True
DAMPEN_WEIGHTING = False
WEIGHT_SQUARE = True
POWER = 2.15
EARLY_STOPPING_PATIENCE = 7
WEIGHT_DECAY = 5e-2
OPTIMIZER = "AdamW"
LAMBDA_SCHEDULER = False
WARMUP_EPOCHS = 5
CUSTOM_LOSS = False
ALPHA = 0.5
GAMMA = 0
HAS_CROP = True

# Model
MODEL_NAME = MODEL_NAME_LIST[11]

# Fixed label order — must match how 'target' was built in data_prep.py.

LABEL_COLS = []

if PROCESSED_DATA_PATH == "../data/processed/df_merged.csv" or PROCESSED_DATA_PATH == "../data/processed/df_merged_cleaned.csv":
    LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
elif PROCESSED_DATA_PATH == "../data/processed/df_merged_2_classes.csv":
    LABEL_COLS = ["MEL", "BKL"]
elif PROCESSED_DATA_PATH == "../data/processed/df_merged_3_classes.csv":
    LABEL_COLS = ["MEL", "NV", "BKL"]
else:
    LABEL_COLS = ["MEL", "NV", "BCC", "BKL"]

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