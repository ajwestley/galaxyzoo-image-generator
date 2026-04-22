import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Any

class MaskConv2D(nn.Module):
    """
    A 2D convolutional layer that applies a causal mask to its kernel.
    
    This is used in autoregressive models (like PixelCNN) where the model
    must predict pixels in a specific order without "cheating" by looking
    at pixels it's supposed to predict.
    
    The mask ensures the receptive field only includes:
    - All pixels ABOVE the current pixel (previous rows)
    - Pixels to the LEFT of the current pixel (same row, but left side)
    - Optionally, the current pixel itself (if mask_type="B")
    """
    def __init__(
        self, 
        mask_type: Literal["A", "B"],
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        **kwargs: Any
    ) -> None:
        """
        Args:
            mask_type: "A" = don't include current pixel, "B" = include current pixel
            in_channels: Number of input channels (e.g., 3 for RGB)
            out_channels: Number of output filters
            kernel_size: Size of the convolutional kernel (default 3x3)
            **kwargs: Other Conv2D arguments (padding, stride, etc.)
        """
        super().__init__()
        self.mask_type = mask_type
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )
        
        self._register_mask()
    
    def _register_mask(self) -> None:
        """
        Create the causal mask and register it as a buffer.
        
        The mask has shape (out_channels, in_channels, kernel_height, kernel_width).
        We set certain positions to 0 to "block" connections that would violate
        the autoregressive property.
        """
        kernel_shape = self.conv.weight.shape
        
        mask = torch.zeros(kernel_shape)
        
        kernel_h, kernel_w = kernel_shape[2], kernel_shape[3]
        
        mask[:, :, :kernel_h // 2, :] = 1.0
        
        mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.0
        

        if self.mask_type == "B":
            mask[:, :, kernel_h // 2, kernel_w // 2] = 1.0
        
        self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply mask to kernel, then convolve.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor after masked convolution
        """
        weight = self.conv.weight * self.mask
        
        return F.conv2d(
            x,
            weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

class MaskedResBlock(nn.Module):
    ''''''
    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters//2, kernel_size=1)
        self.pixel_conv = MaskConv2D(mask_type='B', in_channels=filters//2, out_channels=filters//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters//2, out_channels=filters, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, res: torch.Tensor):
        x = self.conv1(res)
        x = self.relu(x)
        
        x = self.pixel_conv(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        return res + x
    