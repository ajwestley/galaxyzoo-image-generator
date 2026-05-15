import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset

class LatentIndicesDataset(Dataset):
    """
    Load pre-computed latent indices on-the-fly.
    Minimal memory footprint, perfect for large datasets.
    """
    def __init__(self, indices_dir):
        self.indices_dir = Path(indices_dir)
        
        # Get all .pt files and sort by name (so they're in order)
        self.indices_files = sorted(
            self.indices_dir.glob('*.pt'),
            key=lambda x: int(x.stem)
        )
        
        print(f"Found {len(self.indices_files)} latent indices")
    
    def __getitem__(self, idx):
        # Load from disk (very fast with SSD)
        indices_path = self.indices_files[idx]
        indices = torch.load(indices_path)  # (H, W) of integers
        return indices.long()
    
    def __len__(self):
        return len(self.indices_files)

def get_dataloader(data_folder, batch_size, n=50000, seed=42):

    random.seed(seed)

    # Create dataset
    full_dataset = LatentIndicesDataset(data_folder)

    # Select 50k random images
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    subset_indices = all_indices[:n]

    # Subset + DataLoader
    dataset = Subset(full_dataset, subset_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return loader