import torch
from matplotlib.pyplot import imsave

def reconstruct_from_indices(indices_grid, vqvae_model, device='cuda'):
    """
    Quick reconstruction: indices → latent → image
    """
    indices_grid = indices_grid.to(device)
    
    with torch.no_grad():
        # Lookup
        latent_vectors = torch.nn.functional.embedding(
            indices_grid,
            vqvae_model.vq.embedding
        )
        
        # Reshape
        latent = latent_vectors.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        images = vqvae_model.decoder(latent)
    
    return images

def save_image(filepath: str, image: torch.Tensor, img_dim: int = 128, img_channels: int = 3):
    img = image.to('cpu').detach().numpy().reshape(img_channels, img_dim, img_dim).transpose((1, 2, 0))
    imsave(filepath, img)