import torch.nn as nn
import torchvision.models as models

# ResNet backbone with regression head
def build_resnet(model_name="resnet50", pretrained=True, dropout=0.3):
    if model_name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = net.fc.in_features
    elif model_name == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = net.fc.in_features
    else:
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = net.fc.in_features

    # Replace classification head with regression head
    net.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )
    return net
