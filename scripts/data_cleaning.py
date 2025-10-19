import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, META_FILE

def clean_and_split(test_size=0.15, val_size=0.15, seed=42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(META_FILE)
    assert "image_path" in df.columns
    assert "health_index" in df.columns

    df = df.dropna(subset=["image_path", "health_index"])
    abs_paths = []
    for p in tqdm(df["image_path"], desc="Verifying image paths"):
        abs_p = os.path.join(RAW_DIR, p)
        abs_paths.append(abs_p if os.path.exists(abs_p) else None)
    df["abs_image_path"] = abs_paths
    df = df.dropna(subset=["abs_image_path"])
    df["health_index"] = df["health_index"].clip(lower=0, upper=100)

    bins = np.linspace(0, 100, 11)
    df["bin"] = np.digitize(df["health_index"], bins)
    bin_counts = df["bin"].value_counts()
    valid_bins = bin_counts[bin_counts >= 2].index
    df = df[df["bin"].isin(valid_bins)]

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["bin"])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["bin"])

    train_df[["abs_image_path", "health_index"]].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df[["abs_image_path", "health_index"]].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df[["abs_image_path", "health_index"]].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("✅ Cleaned and saved train/val/test splits.")

if __name__ == "__main__":
    clean_and_split()
