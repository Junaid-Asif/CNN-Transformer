import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from tqdm import tqdm

from dataset import TransformerHealthDataset
from augment import build_transforms
from utils import get_device
from models.custom_cnn import CustomCNN
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet
import config as cfg


# ---------- Model Loader ----------

def load_model(ckpt_path):
    """Rebuild and load model weights from checkpoint."""
    name = cfg.MODEL_NAME
    pretrained = cfg.PRETRAINED
    dropout = cfg.DROPOUT

    if name == "custom_cnn":
        model = CustomCNN(dropout=dropout)
    elif "resnet" in name:
        model = build_resnet(model_name=name, pretrained=pretrained, dropout=dropout)
    elif "efficientnet" in name:
        model = build_efficientnet(model_name=name, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {name}")

    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    return model


# ---------- Evaluation ----------

def evaluate_test(ckpt_path):
    device = get_device()
    print(f"Using device: {device}")

    # Transforms (same as training)
    _, _, test_t = build_transforms(
        image_size=cfg.IMAGE_SIZE,
        mean=cfg.NORMALIZE_MEAN,
        std=cfg.NORMALIZE_STD,
        augment_cfg=cfg.AUGMENT
    )

    # Load dataset
    test_csv = os.path.join(cfg.PROCESSED_DIR, "test.csv")
    test_ds = TransformerHealthDataset(test_csv, transform=test_t)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # Load model
    model = load_model(ckpt_path).to(device)
    model.eval()
    criterion = nn.L1Loss()

    losses, preds_list, targets_list = [], [], []
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Testing", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses.append(loss.item() * imgs.size(0))
            preds_list.extend(outputs.squeeze(1).cpu().tolist())
            targets_list.extend(targets.squeeze(1).cpu().tolist())

    # Metrics
    avg_loss = sum(losses) / len(test_loader.dataset)
    preds_scaled = [p * 100.0 for p in preds_list]
    targets_scaled = [t * 100.0 for t in targets_list]
    mae = mean_absolute_error(targets_scaled, preds_scaled)
    r2 = r2_score(targets_scaled, preds_scaled)

    # Save results
    os.makedirs(cfg.METRICS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.METRICS_DIR, f"{cfg.MODEL_NAME}_test_metrics.csv")
    pd.DataFrame({"pred": preds_scaled, "target": targets_scaled}).to_csv(out_path, index=False)

    print(f"\n✅ Evaluation Complete:")
    print(f"Test Loss: {avg_loss:.4f} | Test MAE (0–100): {mae:.2f} | R²: {r2:.3f}")
    print(f"Predictions saved to: {out_path}")


# ---------- Entry ----------

if __name__ == "__main__":
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MODEL_NAME}_best.pth")
    evaluate_test(ckpt)
