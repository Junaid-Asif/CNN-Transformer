# Transformer Health Index (Power/Distribution) — Vision Regression

This repository predicts a transformer health index from images using a custom CNN and compares against pretrained ResNet/EfficientNet backbones. Includes data cleaning, augmentation/scaling, training, evaluation, Grad-CAM, and hyperparameter tuning.

## Steps

1. Prepare raw data
   - Place images under `data/raw/` and a CSV `data/raw/metadata.csv` with:
     - `image_path`: relative path to image (e.g., `images/tx_001.jpg`)
     - `health_index`: numeric target (0–100)

2. Clean and split
   ```bash
   python scripts/data_cleaning.py
