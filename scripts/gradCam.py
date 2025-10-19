import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from utils import get_device
from models.custom_cnn import CustomCNN
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet
import config as cfg


# ---------- Grad-CAM Core ----------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self):
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def get_target_layer(model):
    """Identify the final convolutional layer for Grad-CAM."""
    if isinstance(model, CustomCNN):
        return model.features[-1].block[0]  # last Conv2d
    if hasattr(model, "layer4"):
        return list(model.layer4.children())[-1].conv2
    if hasattr(model, "features"):
        return model.features[-1][0]
    raise ValueError("Could not find target layer for GradCAM")


def overlay_cam_on_image(image, cam_tensor):
    img_np = np.array(image)
    cam = cam_tensor.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)
    return overlay


# ---------- Main ----------

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Build and load model
    name = cfg.MODEL_NAME
    pretrained = cfg.PRETRAINED
    dropout = cfg.DROPOUT

    if name == "custom_cnn":
        model = CustomCNN(dropout=dropout)
    elif "resnet" in name:
        model = build_resnet(model_name=name, pretrained=pretrained, dropout=dropout)
    else:
        model = build_efficientnet(model_name=name, pretrained=pretrained, dropout=dropout)

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{name}_best.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    # Load one sample image from test set
    test_csv = os.path.join(cfg.PROCESSED_DIR, "test.csv")
    import pandas as pd
    df = pd.read_csv(test_csv)
    img_path = df.iloc[0]["image_path"]
    original = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD)
    ])
    x = transform(original).unsqueeze(0).to(device)

    # Grad-CAM setup
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Forward + backward
    output = model(x)
    score = output.squeeze()
    model.zero_grad()
    score.backward()

    cam = gradcam.generate()
    overlay = overlay_cam_on_image(original, cam)

    os.makedirs(cfg.GRADCAM_DIR, exist_ok=True)
    out_file = os.path.join(cfg.GRADCAM_DIR, f"{name}_gradcam.jpg")
    cv2.imwrite(out_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved Grad-CAM visualization to: {out_file}")


if __name__ == "__main__":
    main()
