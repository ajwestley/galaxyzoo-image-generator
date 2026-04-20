import torch
import torch.nn as nn
import torch.nn.functional as F
from Pixel_CNN.aux_layers import MaskConv2D, MaskedResBlock

class PixelCNN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        num_res: int, 
        filters: int
    ) -> None:
        super(PixelCNN, self).__init__()
        self.input_layer = MaskConv2D('A', in_channels, filters, kernel_size=7, padding=3)
        
        self.residual_blocks = nn.ModuleList([MaskedResBlock(filters, filters) for _ in range(num_res)])
        
        self.final_mask = MaskConv2D('B', filters, filters, kernel_size=1)
        # self.output_mask = MaskConv2D('B', filters, out_channels, kernel_size=1)
        
        self.output = nn.Conv2d(filters, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        x = F.one_hot(x, num_classes=self.output.out_channels)
        x = x.permute(0, 3, 1, 2).float()
        
        x = self.input_layer(x)
        x = self.relu(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.final_mask(x)
        x = self.relu(x)
        
        return self.output(x)
    
    @torch.no_grad()
    def sample(self, grid_size, device):
        self.eval()
        
        H, W = grid_size
        
        # Start with zeros (or any constant index)
        z = torch.zeros((1, H, W), dtype=torch.long, device=device)
        
        for i in range(H):
            for j in range(W):
                
                # Forward pass
                logits = self(z)  # (1, K, H, W)
                
                # Get logits for current position
                logits_ij = logits[:, i, j]  # (K,)
                
                # Convert to probabilities
                probs = F.softmax(logits_ij, dim=0)
                
                # Sample from categorical distribution
                z[0, i, j] = torch.multinomial(probs, num_samples=1)
        
        return z  # (1, H, W)