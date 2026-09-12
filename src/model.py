# Terra Incognita — model definitions.
# Pretrained ResNet-18 backbone + 3-class damage head.
import torch
import torch.nn as nn
import torchvision

# xView2 4-class scheme collapsed to 3: 0=undamaged, 1=damaged, 2=destroyed.
NUM_CLASSES = 3


def make_model(pretrained: bool = True, n_classes: int = NUM_CLASSES, device="cpu"):
    """ResNet-18(ImageNet) with the final fc swapped for a 3-way head.

    Input is a POST-disaster RGB patch only (keeps conv1 at 3 channels).
    pretrained download failure -> falls back to random init (offline mode).
    """
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
    except Exception as e:  # offline: no ImageNet weights available
        print(f"[model] pretrained weights unavailable ({e}); using random init")
        model = torchvision.models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    # TODO(hackathon): optionally lower backbone LR (set lr_downstream < lr_head).
    return model.to(device)