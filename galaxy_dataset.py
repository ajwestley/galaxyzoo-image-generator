import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class RawImageDataset(Dataset):
    def __init__(self, folder_path, image_dims):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_dims = image_dims

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        tensor = pil_to_tensor(image, self.image_dims)
        return tensor


def pil_to_tensor(image, image_dims):
    image = image.resize(image_dims)
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def get_dataloader(image_folder, image_dims, batch_size, n=50000, seed=42):

    random.seed(seed)

    # Create dataset
    full_dataset = RawImageDataset(image_folder, image_dims)

    # Select 50k random images
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    subset_indices = all_indices[:n]

    # Subset + DataLoader
    dataset = Subset(full_dataset, subset_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return loader
