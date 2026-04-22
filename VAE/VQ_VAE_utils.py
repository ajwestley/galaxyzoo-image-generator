import torch
import torch.nn.functional as F
import tqdm
import lpips
from VAE.VQ_VAE import VQVAE

perceptual_loss = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')

def construct_vae(device, embedding_dim=256, num_embeddings=512, commitment=0.25):
    """Constructs a VQ-VAE model."""
    return VQVAE(embedding_dim, num_embeddings, commitment).to(device)

def optimizer(vae, learning_rate=3e-4):
    """Constructs an optimizer for the VQ-VAE."""
    return torch.optim.Adam(vae.parameters(), lr=learning_rate)

def train(vae, device, train_loader, optimizer, commitment_cost, lmd_lpips=0.1, lmd_l1 = 0.05):
    """Trains the VQ-VAE for one epoch."""
    vae.vq.commitment_cost = commitment_cost
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
        recon_loss = F.l1_loss(x_recon, data, reduction='mean')
        l1_loss = F.l1_loss(x_recon, data, reduction='mean')
        lpips_loss = perceptual_loss(x_recon, data).mean()
        loss = recon_loss + vq_loss + lmd_lpips * lpips_loss + lmd_l1 * l1_loss
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
