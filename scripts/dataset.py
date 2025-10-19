import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Simple dataset reading images and targets from CSV
class TransformerHealthDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_col="image_path", target_col="health_index"):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_col = image_col
        self.target_col = target_col

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.image_col]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = float(row[self.target_col]) / 100.0
        return img, target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.image_col]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = float(row[self.target_col])
        # Scale target to 0-1 for regression consistency (optional)
        target_scaled = target / 100.0
        return img, target_scaled
