import torch
import torch.nn as nn
import torch.nn.functional as F
from Pixel_CNN.aux_layers import MaskConv2D, MaskedResBlock

class PixelCNN(nn.Module):
    """
    A PixelCNN for the purpose of sampling a VQ-VAE latent space.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        num_res: int, 
        filters: int
    ) -> None:
        """
        Initialise the model.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            num_res (int): The number of residual blocks.
            filters (int): The number of filters in each convolutional layer.
        """
        super(PixelCNN, self).__init__()
        self.embedding = nn.Embedding(in_channels, filters)
        self.input_layer = MaskConv2D('A', filters, filters, kernel_size=7, padding=3)
        
        self.residual_blocks = nn.ModuleList([MaskedResBlock(filters, filters) for _ in range(num_res)])
        
        self.final_mask = MaskConv2D('B', filters, filters, kernel_size=1)
        self.output = nn.Conv2d(filters, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        """
        Perform one pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, height, width) containing integer indices corresponding to the VQ-VAE embeddings.

        Returns:
            torch.Tensor: The output tensor of shape (batch, out_channels, height, width) containing the logits for each embedding index at each pixel location.
        """
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.input_layer(x)
        x = self.relu(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.final_mask(x)
        x = self.relu(x)
        
        return self.output(x)
    
    @torch.no_grad()
    def sample(self, grid_size: tuple[int, int], device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from the learnt pixel distribution to generate an image.

        Args:
            grid_size (tuple[int, int]): The size of the image grid to generate.
            device (torch.device): The device on which to perform the sampling.

        Returns:
            torch.Tensor: The generated image tensor of shape (1, height, width).
        """
        self.eval()
        
        H, W = grid_size
        
        z = torch.zeros((1, H, W), dtype=torch.long, device=device)
        
        for i in range(H):
            for j in range(W):
                
                # Forward pass
                logits = self(z)
                
                # Get logits for current position
                logits_ij = logits[0, :, i, j] / temperature
                
                # Convert to probabilities
                probs = F.softmax(logits_ij, dim=0)
                
                # Sample from categorical distribution
                z[0, i, j] = torch.multinomial(probs, num_samples=1)
        
        return z