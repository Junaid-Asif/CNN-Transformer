import os
import random
import yaml
import torch
import numpy as np

def load_config(path="C:\\Users\\User\\OneDrive\\Documents\\Tranformers_health_index\\configs\\default_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, best_metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)
