import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import load_config

cfg = load_config()

RAW_DIR = cfg["raw_dir"]
PROCESSED_DIR = cfg["processed_dir"]
META_FILE = os.path.join(RAW_DIR, "metadata.csv")
# MAIN FUNCTION OF THS SCRIPT
def clean_and_split(test_size=0.15, val_size=0.15, seed=42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(META_FILE)
    assert cfg["image_col"] in df.columns
    assert cfg["target_col"] in df.columns

    df = df.dropna(subset=[cfg["image_col"], cfg["target_col"]])
    abs_paths = []
    for p in tqdm(df[cfg["image_col"]], desc="Verifying image paths"):
        abs_p = os.path.join(RAW_DIR, p)
        abs_paths.append(abs_p if os.path.exists(abs_p) else None)
    df["abs_image_path"] = abs_paths
    df = df.dropna(subset=["abs_image_path"])
    df[cfg["target_col"]] = df[cfg["target_col"]].clip(lower=0, upper=100)

    # Bin health index into 10 groups
    bins = np.linspace(0, 100, 11)
    df["bin"] = np.digitize(df[cfg["target_col"]], bins)

    # Remove bins with <2 samples
    bin_counts = df["bin"].value_counts()
    valid_bins = bin_counts[bin_counts >= 2].index
    df = df[df["bin"].isin(valid_bins)]

    # Split safely
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["bin"])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["bin"])

    out_cols = ["abs_image_path", cfg["target_col"]]
    train_df[out_cols].rename(columns={"abs_image_path": cfg["image_col"]}).to_csv(
        os.path.join(PROCESSED_DIR, cfg["train_csv"]), index=False)
    val_df[out_cols].rename(columns={"abs_image_path": cfg["image_col"]}).to_csv(
        os.path.join(PROCESSED_DIR, cfg["val_csv"]), index=False)
    test_df[out_cols].rename(columns={"abs_image_path": cfg["image_col"]}).to_csv(
        os.path.join(PROCESSED_DIR, cfg["test_csv"]), index=False)

    print("✅ Cleaned and saved train/val/test splits.")


if __name__ == "__main__":
    clean_and_split()
