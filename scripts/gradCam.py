import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from scripts.utils import load_config, get_device
from models.custom_cnn import CustomCNN
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet

# Simple Grad-CAM implementation focusing on last conv feature map layer
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, scores):
        # scores: scalar output for backward
        grads = self.gradients  # [B, C, H, W]
        acts = self.activations # [B, C, H, W]
        weights = grads.mean(dim=(2,3), keepdim=True)  # global average pooling on grads
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        # Normalize to [0,1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

def get_target_layer(model):
    # Try to return last conv layer depending on architecture
    if isinstance(model, CustomCNN):
        return model.features[-1].block[1]  # BatchNorm after last conv; better: last Conv2d
    # ResNet: layer4's last conv
    if hasattr(model, "layer4"):
        return list(model.layer4.children())[-1].conv2
    # EfficientNet: last features conv
    if hasattr(model, "features"):
        return model.features[-1][0]
    raise ValueError("Could not find target layer for GradCAM")

def overlay_cam_on_image(image, cam_tensor):
    # image: PIL Image, cam_tensor: [1,1,H,W]
    img_np = np.array(image)
    cam = cam_tensor.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)
    return overlay

def main():
    cfg = load_config()
    device = get_device()

    # Build model and load checkpoint
    name = cfg["model_name"]
    pretrained = cfg.get("pretrained", True)
    dropout = cfg.get("dropout", 0.3)
    if name == "custom_cnn":
        model = CustomCNN(dropout=dropout)
    elif "resnet" in name:
        model = build_resnet(model_name=name, pretrained=pretrained, dropout=dropout)
    else:
        model = build_efficientnet(model_name=name, pretrained=pretrained, dropout=dropout)

    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{name}_best.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    # Transform for single image
    transform = T.Compose([
        T.Resize((cfg["image_size"], cfg["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=cfg["normalize_mean"], std=cfg["normalize_std"])
    ])

    # Choose an image (change path as needed)
    test_csv = os.path.join(cfg["processed_dir"], cfg["train_csv"])
    import pandas as pd
    df = pd.read_csv(test_csv)
    img_path = df.iloc[0]["image_path"]
    original = Image.open(img_path).convert("RGB")
    x = transform(original).unsqueeze(0).to(device)

    # Grad-CAM setup
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Forward and backward to get gradients with respect to output
    output = model(x)  # shape [1,1]
    score = output.squeeze()  # scalar
    model.zero_grad()
    score.backward()  # gradients flow to target layer

    cam = gradcam.generate(score.unsqueeze(0))
    overlay = overlay_cam_on_image(original, cam)

    os.makedirs(cfg["gradcam_dir"], exist_ok=True)
    out_file = os.path.join(cfg["gradcam_dir"], f"{name}_gradcam.jpg")
    cv2.imwrite(out_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM: {out_file}")

if __name__ == "__main__":
    main()
