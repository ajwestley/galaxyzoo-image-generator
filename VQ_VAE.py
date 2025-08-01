import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from loss_functions import VGGPerceptualLoss

perceptual_loss = VGGPerceptualLoss()

class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder network: Converts images to feature maps
        self.encoder = nn.Sequential(
            # Input: 3x128x128 -> 32x64x64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 32x64x64 -> 64x32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 256x8x8 -> 128x4x4
            nn.Conv2d(256, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        
        # Fully connected layers for mean and log variance of the latent distribution
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        
        # Decoder network: Reconstructs images from latent representations
        self.decoder = nn.Sequential(
            # Input: 128x4x4 -> 256x8x8
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 32x64x64 -> 3x128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output normalized images in [-1, 1] range
        )
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape [B, 3, 128, 128]
        
        Returns:
            x_recon: reconstructed image
            vq_loss: codebook + commitment loss
            perplexity: for monitoring codebook usage
            z_e: encoder output before quantization
            z_q: quantized latent
        """
        # 1. Encode
        z_e = self.encoder(x)  # shape [B, D, H, W]

        # 2. Vector Quantization
        z_q, vq_loss, perplexity, encoding_indices = self.vq(z_e)

        # 3. Decode
        x_recon = self.decoder(z_q)  # shape [B, 3, 128, 128]

        return x_recon, vq_loss, perplexity, z_e, z_q, encoding_indices

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        z: shape [B, C, H, W]
        """
        # Flatten z -> [BHW, C]
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat_z = z_perm.view(-1, self.embedding_dim)  # [BHW, C]

        # Compute distances to codebook entries: [BHW, K]
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )  # ||z - e||^2

        # Find closest embeddings
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot [BHW, K]

        # Quantize: lookup in codebook
        quantized = encodings @ self.embeddings.weight  # [BHW, C]
        B, H, W, C = z_perm.shape
        quantized = quantized.view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        # Compute loss terms
        codebook_loss = F.mse_loss(quantized.detach(), z, reduction='mean')
        commitment_loss = F.mse_loss(quantized, z.detach(), reduction='mean')
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity
        avg_probs = encodings.detach().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Actual embedding weights (not learnable by gradient)
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", self.embedding.clone())

    def forward(self, z):
        # Reshape input
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat_z = z_perm.view(-1, self.embedding_dim)  # [BHW, C]

        # Compute distances and encoding indices
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(dist, dim=1)  # [BHW]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)  # [BHW, K]

        # Quantize
        quantized = encodings @ self.embedding  # [BHW, C]
        quantized = quantized.view(*z_perm.shape).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        if self.training:
            # EMA updates
            ema_cluster_size = encodings.sum(0)
            ema_embedding_sum = encodings.T @ flat_z

            # Laplace smoothing of cluster size
            self.ema_cluster_size.mul_(self.decay).add_(ema_cluster_size * (1 - self.decay))
            self.ema_embedding.mul_(self.decay).add_(ema_embedding_sum * (1 - self.decay))

            # Normalize to get updated embeddings
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding = self.ema_embedding / cluster_size.unsqueeze(1)

        # Loss
        commitment_loss = F.mse_loss(quantized.detach(), z, reduction='mean')
        loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices



def construct_vae(device, embedding_dim=256, num_embeddings=512, commitment=0.25):
    """ 
    Constructs a Variational Autoencoder (VAE) model.
    
    Args:
        device (torch.device): Device to run the model on (CPU or GPU)
        latent_dim (int): Dimensionality of the latent space
    
    Returns:
        VAE: Configured Variational Autoencoder model
    """
    return VQVAE(embedding_dim, num_embeddings, commitment).to(device)

def optimizer(vae, learning_rate=3e-4):
    """
    Constructs an optimizer for the VAE.
    
    Args:
        vae (VAE): The Variational Autoencoder model
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    return torch.optim.Adam(vae.parameters(), lr=learning_rate)

def train(vae, device, train_loader, optimizer, lmd=0.1):
    """
    Trains the VQ-VAE for one epoch.
    """
    vae.train()
    train_loss = 0.0
    rec_loss = 0.0
    quant_loss = 0.0

    used_indices = set()

    for _, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_recon, vq_loss, _, _, _, encoding_indices = vae(data)

        # Loss
        recon_loss = F.mse_loss(x_recon, data, reduction='mean')
        loss = recon_loss + vq_loss #+ lmd * perceptual_loss(x_recon, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        rec_loss += recon_loss.item() * data.size(0)
        quant_loss += vq_loss.item() * data.size(0)

        # Track code usage
        used_indices.update(encoding_indices.view(-1).tolist())

    avg_total_loss = train_loss / len(train_loader.dataset)
    avg_rec_loss = rec_loss / len(train_loader.dataset)
    avg_vq_loss = quant_loss / len(train_loader.dataset)
    
    return avg_total_loss, avg_rec_loss, avg_vq_loss, len(used_indices)
