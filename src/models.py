"""Model architectures. Add new classes here as you try different
architectures (transfer learning backbones, deeper CNNs, etc.)."""

import timm
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights, efficientnet_b2,
    EfficientNet_B2_Weights, efficientnet_b4, EfficientNet_B4_Weights, vgg19, VGG19_Weights
)
from config import (
    IMAGE_SIZE, CLASSIFICATION_MODE
)

NUM_CLASSES = 2 if CLASSIFICATION_MODE == 'binary' else 8

class SimpleCNN(nn.Module):
    """Deliberately simple: 3 conv blocks + a small classifier head.
    Baseline for sanity-checking the pipeline before scaling up."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56 -> 28

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CustomCNN(nn.Module):
    """More complex CNN architecture. Hyperparameter tuning for this one"""

    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56 -> 28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_resnet18(num_classes, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_efficientnet_b0(num_classes, freeze_backbone=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def get_efficientnet_b2(num_classes, freeze_backbone=True):
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def get_resnet50(num_classes, freeze_backbone=True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_resnet50v2(num_classes, freeze_backbone=True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_densenet121(num_classes, freeze_backbone=True):
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model

def get_efficientvit_b0(num_classes, freeze_backbone=True):
    model = timm.create_model("efficientvit_b0.r224_in1k", pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True


    return model

def get_efficientvit_b2(num_classes, freeze_backbone=True):
    model = timm.create_model("efficientvit_b2.r288_in1k", pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True


    return model

def get_efficientnet_b4(num_classes, freeze_backbone=True):
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def get_vgg19(num_classes, freeze_backbone=True):
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    return model

def get_efficientnet_b2_partial(num_classes, layers_unfrozen=3):
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 4
    for block in model.features[5:]:
        for param in block.parameters():
            param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

class FlatLinearBaseline(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.flatten_dim = in_channels * image_size * image_size
        self.fc = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class SimpleMLP(nn.Module):
    """A standard Multilayer Perceptron (MLP) for tabular metadata classification."""

    def __init__(self, input_dim, num_classes=2, hidden_dims=[128, 64], dropout_rate=0.3):
        super().__init__()

        layers = []
        current_dim = input_dim

        # Build hidden layers with ReLU, BatchNorm, and Dropout for regularization
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Final classification head
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """A residual block for tabular data to help gradients flow through deep networks."""

    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # LayerNorm is often more stable than BatchNorm in deep tabular networks
            nn.SiLU(),  # Swish activation: smooth, non-monotonic, prevents dying neurons
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))  # Residual connection

class ComplexMLP(nn.Module):
    """An advanced, deep MLP architecture featuring residual connections and Swish activations."""

    def __init__(self, input_dim, num_classes=2, base_dim=256, num_blocks=4, dropout_rate=0.3):
        super().__init__()

        # 1. Input Projection Layer (Upscale tabular features to a wider latent space)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. Deep Residual Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_dim, dropout_rate) for _ in range(num_blocks)
        ])

        # 3. Advanced Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.LayerNorm(base_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),  # Reduce dropout slightly at the final head
            nn.Linear(base_dim // 2, num_classes)
        )

    def forward(self, x):
        # Handle 1D inputs safely for LayerNorm if batch size is 1 during evaluation
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.input_layer(x)

        for block in self.res_blocks:
            out = block(out)

        return self.classifier(out)


# Registry — add each new model class here so it can be selected by
# name from config.py (MODEL_NAME) without editing main.py every time.
MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "CustomCNN": CustomCNN,
    "ResNet18_Frozen": lambda num_classes: get_resnet18(num_classes, freeze_backbone=True),
    "ResNet18_Tune": lambda num_classes: get_resnet18(num_classes, freeze_backbone=False),
    "EfficientNetB0_Frozen": lambda num_classes: get_efficientnet_b0(num_classes, freeze_backbone=True),
    "EfficientNetB0_Tune": lambda num_classes: get_efficientnet_b0(num_classes, freeze_backbone=False),
    "ResNet50_Tune": lambda num_classes: get_resnet50(num_classes, freeze_backbone=False),
    "ResNet50V2_Tune": lambda num_classes: get_resnet50v2(num_classes, freeze_backbone=False),
    "DenseNet121_Tune": lambda num_classes: get_densenet121(num_classes, freeze_backbone=False),
    "EfficientNetB2_Tune": lambda num_classes: get_efficientnet_b2(num_classes, freeze_backbone=False),
    "EfficientViT_b0_Tune": lambda num_classes: get_efficientvit_b0(num_classes, freeze_backbone=False),
    "EfficientViT_b2_Tune": lambda num_classes: get_efficientvit_b2(num_classes, freeze_backbone=False),
    "EfficientNetB4_Tune": lambda num_classes: get_efficientnet_b4(num_classes, freeze_backbone=False),
    "VGG19_Tune": lambda num_classes: get_vgg19(num_classes, freeze_backbone=False),
    "EfficientNetB2_Partial": lambda num_classes: get_efficientnet_b2(num_classes),
    "FlatLinearBaseline": FlatLinearBaseline,
    "SimpleMLP": lambda num_classes, input_dim=21: SimpleMLP(input_dim=input_dim, num_classes=num_classes),
    "ComplexMLP": lambda num_classes, input_dim=21: ComplexMLP(input_dim=input_dim, num_classes=num_classes),
}

def get_model(name, num_classes, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)