"""Shared pieces copied from edaV2.ipynb. Edit the two paths if your data moves."""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

META = r"D:\D DRIVE DOWNLOADS\zr7vgbcyr2-1\metadata.csv"
IMG_ROOT = r"D:\D DRIVE DOWNLOADS\zr7vgbcyr2-1\images"

FEATURES = ['grew', 'itch', 'changed', 'elevation', 'bleed', 'region', 'hurt', 'age']
TARGET = 'diagnostic'


def build_image_index(root_dir):
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                index[fname] = os.path.join(dirpath, fname)
    return index


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_index, label_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.le = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.img_index[row['img_id']]).convert('RGB')
        label = self.le.transform([row[TARGET]])[0]
        if self.transform:
            image = self.transform(image)
        return image, label


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
