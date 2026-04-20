from argparse import ArgumentParser
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib.pyplot import imsave
from VQ_VAE import VQVAE
from generator_utils import load_model

DEFAULT_MODEL_PATH = 'models/galaxy_vae.pth'
DEFAULT_FREQ_PATH = 'models/frequencies.pkl'
DEFAULT_DIM = 64
DEFAULT_NUM = 1024
DEFAULT_COMMITMENT = 1

def generate_image(vae: VQVAE, frequencies, temperature: float = 1.0):
    """
    Generate images by sampling from the VQ-VAE latent space.
    
    Args:
        vae: Trained VQ-VAE model
        num_images: Number of images to generate
        temperature: Controls sampling randomness
                    - temperature < 1.0: more conservative (favors common codes)
                    - temperature = 1.0: uniform sampling
                    - temperature > 1.0: more creative/random
        device: 'cuda' or 'cpu'
    
    Returns:
        images: Tensor of shape [num_images, 3, 128, 128]
    """
    vae.eval()
    
    with torch.no_grad():
        num_embeddings = vae.vq.num_embeddings
        latent_h, latent_w = 16, 16
        
        # Create logits from log probabilities, apply temperature
        logits = torch.log(frequencies + 1e-10)  # [num_embeddings]
        logits = logits / temperature
        
        # Expand to spatial dimensions
        logits = logits.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, num_embeddings]
        logits = logits.expand(1, latent_h, latent_w, -1)  # [B, H, W, num_embeddings]
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        probs_flat = probs.reshape(-1, num_embeddings)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1)
        sampled_indices = sampled_flat.reshape(1, latent_h, latent_w)
        
        # Decode
        flat_indices = sampled_indices.reshape(-1)
        latent_codes = vae.vq.embedding[flat_indices]
        latent_codes = latent_codes.reshape(1, latent_h, latent_w, -1)
        latent_codes = latent_codes.permute(0, 3, 1, 2).contiguous()
        
        images = vae.decoder(latent_codes)
        
    return images.detach().numpy()[0]

def save_image(img: np.ndarray, filename: str = 'output.png'):
    imsave(filename, img.transpose((1, 2, 0)))

if __name__ == '__main__':
    parser = ArgumentParser(
            prog='Galaxy image generator',
            description='Generate images of galaxies using a pretrained VQ-VAE.'
        )
    
    parser.add_argument('-d', '--embedding_dim', default=DEFAULT_DIM)
    parser.add_argument('-n', '--num_embeddings', default=DEFAULT_NUM)
    parser.add_argument('-c', '--commitment', default=DEFAULT_COMMITMENT)
    parser.add_argument('-t', '--temperature', default=1.5)
    parser.add_argument('-o', '--output', default='output.png')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    model_file = DEFAULT_MODEL_PATH
    embedding_dim = int(args.embedding_dim)
    num_embeddings = int(args.num_embeddings)
    commitment = float(args.commitment)
    temperature = float(args.temperature)
    outfile = args.output
    verbose = bool(args.verbose)
    
    with open(DEFAULT_FREQ_PATH, 'rb') as file:
        frequencies = pickle.load(file)
    
    model = load_model(model_file, embedding_dim, num_embeddings, commitment, verbose)
    image = generate_image(model, frequencies, temperature)
    save_image(image, outfile)