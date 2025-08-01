import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16
from torchvision.transforms import Normalize

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(weights=True).features[:16].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)
        return F.mse_loss(self.vgg(x), self.vgg(y))
