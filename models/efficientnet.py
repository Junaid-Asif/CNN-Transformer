import torch.nn as nn
import torchvision.models as models

# EfficientNet backbone with regression head
def build_efficientnet(model_name="efficientnet_b0", pretrained=True, dropout=0.3):
    builder = getattr(models, model_name)
    weights_cls = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        "efficientnet_b2": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
    }
    weights = weights_cls.get(model_name, None) if pretrained else None
    net = builder(weights=weights)

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )
    return net
