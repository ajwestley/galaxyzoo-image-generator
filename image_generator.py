import torch
from matplotlib.pyplot import imsave
from utils.general import load_models_from_json, available_device
class ImageGenerator:
    """
    A composite class to combine the VQ-VAE and PixelCNN for image generation.
    """
    def __init__(self, config_file: str) -> None:
        """
        Load an ImageGenerator from a config file.

        Args:
            config_file (str): A JSON file containing the model configs.
        """
        vae, cnn = load_models_from_json(config_file)
        
        self.vqvae = vae
        self.pixelcnn = cnn
        self.device = available_device()
        
        self.chans = vae.encoder[0].in_channels
        self.image_dims = self.vqvae.image_dims
        
        sample_input = torch.randn(1, self.chans, self.image_dims[0], self.image_dims[1]).to(self.device)
        with torch.no_grad():
            self.latent_dims = vae.encoder(sample_input).shape[1:]
    
    def generate_image(self, output_file: str, temperature: float = 1.0):
        """
        Generate and save an image.

        Args:
            output_file (str): The file path to save the image to.
            temperature (float): The temperature for sampling from the PixelCNN.
        """
        z = self.pixelcnn.sample((self.latent_dims[1], self.latent_dims[2]), self.device, temperature)
        z = self.__reconstruct_from_indices(z)
        self.__save_image(output_file, z)
    
    def __reconstruct_from_indices(self, indices_grid: torch.Tensor) -> torch.Tensor:
        indices_grid = indices_grid.to(self.device)
        with torch.no_grad():
            latent_vectors = torch.nn.functional.embedding(
                indices_grid,
                self.vqvae.vq.embedding
            )
            latent = latent_vectors.permute(0, 3, 1, 2).contiguous()
            images = self.vqvae.decoder(latent)
        return images
    
    def __save_image(self, filepath: str, image: torch.Tensor):
        self.vqvae.decoder.state_dict()
        img = image.to('cpu').detach().numpy().reshape(self.chans, self.image_dims[0], self.image_dims[1]).transpose((1, 2, 0))
        imsave(filepath, img)
