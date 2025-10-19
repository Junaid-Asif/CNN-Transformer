import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class TransformerHealthDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = float(row["health_index"]) / 100.0
        return img, target

    def __len__(self):
        return len(self.df)
