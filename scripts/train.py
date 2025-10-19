import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# above line lets us import from project root when running: `python scripts/train.py`

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from dataset import TransformerHealthDataset
from augment import build_transforms
from utils import set_seed, get_device, save_checkpoint

from models.custom_cnn import CustomCNN
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet

import config as cfg


# ---------- Model / Optim / Sched helpers ----------

def build_model():
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

    if cfg.FREEZE_BACKBONE and name != "custom_cnn":
        # Freeze all but final head
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze classifier/fc
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True

    return model


def get_optimizer(model_params):
    lr = cfg.LEARNING_RATE
    wd = cfg.WEIGHT_DECAY
    opt = cfg.OPTIMIZER.lower()
    if opt == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)
    elif opt == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        return torch.optim.Adam(model_params, lr=lr, weight_decay=wd)


def get_scheduler(optimizer):
    sched = cfg.SCHEDULER
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    return None


# ---------- Train / Eval loops ----------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device).unsqueeze(1)  # shape (B,1)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds_list, targets_list = [], [], []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses.append(loss.item() * imgs.size(0))
            preds_list.extend(outputs.squeeze(1).cpu().tolist())
            targets_list.extend(targets.squeeze(1).cpu().tolist())

    avg_loss = sum(losses) / len(loader.dataset)
    preds_scaled = [p * 100.0 for p in preds_list]
    targets_scaled = [t * 100.0 for t in targets_list]
    mae = mean_absolute_error(targets_scaled, preds_scaled)
    return avg_loss, mae, preds_scaled, targets_scaled


# ---------- Main ----------

def main():
    set_seed(cfg.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Transforms
    train_t, val_t, _ = build_transforms(
        image_size=cfg.IMAGE_SIZE,
        mean=cfg.NORMALIZE_MEAN,
        std=cfg.NORMALIZE_STD,
        augment_cfg=cfg.AUGMENT
    )

    # Datasets / Loaders
    train_csv = os.path.join(cfg.PROCESSED_DIR, "train.csv")
    val_csv   = os.path.join(cfg.PROCESSED_DIR, "val.csv")

    train_ds = TransformerHealthDataset(train_csv, transform=train_t)
    val_ds   = TransformerHealthDataset(val_csv,   transform=val_t)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_memory)

    # Model / Optim / Sched
    model = build_model().to(device)
    criterion = nn.L1Loss()  # MAE for regression
    optimizer = get_optimizer(model.parameters())
    scheduler = get_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler() if (cfg.MIXED_PRECISION and device.type == "cuda") else None

    best_mae = math.inf
    patience = cfg.EARLY_STOPPING_PATIENCE
    no_improve = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler:
            if cfg.SCHEDULER == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE (0-100): {val_mae:.2f}")

        # Early stopping on MAE
        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MODEL_NAME}_best.pth")
            save_checkpoint(model, optimizer, epoch, best_mae, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
