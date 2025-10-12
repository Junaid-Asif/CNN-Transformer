import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Simple, safe cleaning to produce train/val/test CSVs with image paths and targets
# Assumes a metadata CSV in data/raw/metadata.csv with columns:
# - image_path (relative to data/raw)
# - health_index (float target 0-1 or 0-100)
# Adjust column names as needed.

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
META_FILE = os.path.join(RAW_DIR, "metadata.csv")

def clean_and_split(test_size=0.15, val_size=0.15, seed=42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load metadata
    df = pd.read_csv(META_FILE)

    # Basic checks
    assert "image_path" in df.columns, "metadata.csv must have 'image_path'"
    assert "health_index" in df.columns, "metadata.csv must have 'health_index'"

    # Drop rows with missing paths or targets
    df = df.dropna(subset=["image_path", "health_index"])

    # Resolve image absolute paths and filter existing images
    abs_paths = []
    for p in tqdm(df["image_path"], desc="Verifying image paths"):
        abs_p = os.path.join(RAW_DIR, p)
        abs_paths.append(abs_p if os.path.exists(abs_p) else None)
    df["abs_image_path"] = abs_paths
    df = df.dropna(subset=["abs_image_path"])

    # Remove obvious outliers in health_index (optional clipping to [0, 100])
    df["health_index"] = df["health_index"].clip(lower=0, upper=100)

    # Stratify is tricky for regression; we bin the target to keep distribution
    bins = np.linspace(0, 100, 11)
    df["bin"] = np.digitize(df["health_index"], bins)

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["bin"]
    )

    # Train/val split
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df["bin"]
    )

    # Save CSVs
    out_cols = ["abs_image_path", "health_index"]
    train_df[out_cols].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "train.csv"), index=False
    )
    val_df[out_cols].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "val.csv"), index=False
    )
    test_df[out_cols].rename(columns={"abs_image_path": "image_path"}).to_csv(
        os.path.join(PROCESSED_DIR, "test.csv"), index=False
    )

    print("Saved cleaned splits in data/processed")

if __name__ == "__main__":
    clean_and_split()
