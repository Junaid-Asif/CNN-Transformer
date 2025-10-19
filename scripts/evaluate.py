import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from tqdm import tqdm
from scripts.dataset import TransformerHealthDataset
from scripts.augment import build_transforms
from scripts.utils import load_config, get_device
from models.custom_cnn import CustomCNN
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet

# Load model weights and evaluate on test set
def load_model(cfg, ckpt_path):
    # Build same architecture
    name = cfg["model_name"]
    pretrained = cfg.get("pretrained", True)
    dropout = cfg.get("dropout", 0.3)
    if name == "custom_cnn":
        model = CustomCNN(dropout=dropout)
    elif "resnet" in name:
        model = build_resnet(model_name=name, pretrained=pretrained, dropout=dropout)
    elif "efficientnet" in name:
        model = build_efficientnet(model_name=name, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_name: {name}")

    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    return model

def evaluate_test(cfg, ckpt_path):
    device = get_device()
    _, _, test_t = build_transforms(cfg["image_size"], cfg["normalize_mean"], cfg["normalize_std"], cfg.get("augment", {}))
    test_ds = TransformerHealthDataset(os.path.join(cfg["processed_dir"], cfg["train_csv"])
), transform=test_t)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    model = load_model(cfg, ckpt_path).to(device)
    model.eval()
    criterion = nn.L1Loss()

    losses, preds_list, targets_list = [], [], []
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Test", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses.append(loss.item() * imgs.size(0))
            preds_list.extend(outputs.squeeze(1).cpu().tolist())
            targets_list.extend(targets.squeeze(1).cpu().tolist())

    avg_loss = sum(losses) / len(test_loader.dataset)
    preds_scaled = [p * 100.0 for p in preds_list]
    targets_scaled = [t * 100.0 for t in targets_list]
    mae = mean_absolute_error(targets_scaled, preds_scaled)
    r2 = r2_score(targets_scaled, preds_scaled)

    # Save metrics
    os.makedirs(cfg["metrics_dir"], exist_ok=True)
    out_path = os.path.join(cfg["metrics_dir"], f"{cfg['model_name']}_test_metrics.csv")
    pd.DataFrame({"pred": preds_scaled, "target": targets_scaled}).to_csv(out_path, index=False)

    print(f"Test Loss: {avg_loss:.4f} | Test MAE (0-100): {mae:.2f} | R2: {r2:.3f}")
    print(f"Saved predictions: {out_path}")

if __name__ == "__main__":
    cfg = load_config()
    ckpt = os.path.join(cfg["checkpoint_dir"], f"{cfg['model_name']}_best.pth")
    evaluate_test(cfg, ckpt)
