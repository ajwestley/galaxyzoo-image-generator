import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder network: Converts images to feature maps
        self.encoder = nn.Sequential(
            # 128 → 64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            # 64 → 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),

            # 32 → 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),

            # 16 → 16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),

            # 16 → 16
            nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )

        
        # Fully connected layers for mean and log variance of the latent distribution
        self.vq = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        
        # Decoder network: Reconstructs images from latent representations
        self.decoder = nn.Sequential(
            # 16 → 16
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),

            # 16 → 16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),

            # 16 → 32
            PixelShuffleBlock(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),

            # 32 → 64
            PixelShuffleBlock(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            # 64 → 128
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
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

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook and EMA buffers: [K, D]
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_w", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))

    def forward(self, inputs):
        """
        inputs: [B, C, H, W]
        returns: quantized, loss, perplexity, encoding_indices
        """
        B, C, H, W = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Distances to embeddings: [N, K]
        with torch.no_grad():  # distances don't need grad
            e2 = (self.embedding ** 2).sum(dim=1)
            x2 = (flat_input ** 2).sum(dim=1, keepdim=True)
            xe = flat_input @ self.embedding.t()
            distances = x2 - 2 * xe + e2.unsqueeze(0)

            # Nearest code indices
            encoding_indices = torch.argmin(distances, dim=1)

        # Quantize by lookup and reshape back to [B,C,H,W]
        quantized = self.embedding.index_select(0, encoding_indices)
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # EMA updates
        if self.training:
            with torch.no_grad():
                # Counts per code: [K]
                counts = torch.bincount(encoding_indices, minlength=self.num_embeddings).to(self.cluster_size.dtype)

                # Sums of assigned vectors per code: [K, D]
                sums = torch.zeros_like(self.ema_w)
                sums.index_add_(0, encoding_indices, flat_input)
                
                # EMA update
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(sums, alpha=1 - self.decay)

                # Normalization to avoid shrinking when codes are rare
                n = self.cluster_size.sum()
                # Re-normalize cluster_size so sums/size ~ running mean
                cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                # Update codebook
                self.embedding.copy_(self.ema_w / cluster_size.clamp_min(self.eps).unsqueeze(1))

        # Commitment loss only (no codebook loss with EMA)
        commit_loss = F.mse_loss(quantized.detach(), inputs, reduction="mean")
        loss = self.commitment_cost * commit_loss

        # Straight-through: gradients w.r.t. inputs
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (usage metric) without one-hot
        with torch.no_grad():
            probs = torch.bincount(encoding_indices, minlength=self.num_embeddings).float()
            probs = probs / probs.sum().clamp_min(self.eps)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return quantized, loss, perplexity, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)
    
class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * upscale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)