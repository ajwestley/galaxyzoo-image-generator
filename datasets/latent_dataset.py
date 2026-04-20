import torch
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