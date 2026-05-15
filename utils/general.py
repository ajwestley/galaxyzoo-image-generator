import torch
import json
from collections import OrderedDict

from VAE.VQ_VAE import VQVAE
from Pixel_CNN.pixel_cnn import PixelCNN
from utils import exceptions

def available_device() -> torch.device:
    """
    Gets the most convenient available device.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_from_json(filepath: str) -> tuple[VQVAE, PixelCNN]:
    
    try:
        with open(filepath) as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        exceptions.show_error(f"Config file not found at '{filepath}'.")
    except json.JSONDecodeError:
        exceptions.show_error(f"JSON file '{filepath}' cannot be decoded.")
    
    validate_config(config)
    
    vqvae_path = config['vqvae_path']
    pixelcnn_path = config['pixelcnn_path']
    embedding_dim = config['embedding_dimensions']
    num_embeddings = config['num_embeddings']
    res_blocks = config['residual_blocks']
    filters = config['filters']
    image_dims = (config['image_w'], config['image_h'])
    
    try:
        vqvae_weights = torch.load(vqvae_path, weights_only=True)
    except FileNotFoundError:
        exceptions.show_error(f"VQ-VAE file not found at '{filepath}'.")
    
    try:
        pixelcnn_weights = torch.load(pixelcnn_path, weights_only=True)
    except FileNotFoundError:
        exceptions.show_error(f"PixelCNN file not found at '{filepath}'.")
    
    vqvae = VQVAE(embedding_dim, num_embeddings, image_dims=image_dims)
    pixelcnn = PixelCNN(num_embeddings, num_embeddings, res_blocks, filters)
    
    validate_vqvae_params(vqvae, vqvae_weights)
    validate_pixelcnn_params(pixelcnn, pixelcnn_weights)
    
    vqvae.load_state_dict(vqvae_weights)
    pixelcnn.load_state_dict(pixelcnn_weights)
    
    device = available_device()
    
    return vqvae.to(device), pixelcnn.to(device)
    

def validate_config(config: dict) -> None:
    str_params = ['vqvae_path', 'pixelcnn_path']
    int_params = ['num_embeddings', 'embedding_dimensions', 'residual_blocks', 'filters', 'image_w', 'image_h']
    all_params = str_params + int_params
    
    for p in all_params:
        if p not in config:
            exceptions.show_error(f"Config file: Parameter'{p}' is missing.")
    
    for p in str_params:
        if not isinstance(config[p], str):
            exceptions.show_error(f"Config file: Parameter '{p}' is not a valid string.")
    
    for p in int_params:
        if not isinstance(config[p], int):
            exceptions.show_error(f"Config file: Parameter '{p}' is not a valid integer.")
        elif config[p] < 1:
            exceptions.show_error(f"Config file: Parameter '{p}' must be > 0.")

def validate_vqvae_params(vqvae: VQVAE, weights: OrderedDict):
    weight_num_embeddings, weight_embedding_dim = weights['vq.embedding'].shape
    
    if weight_num_embeddings != vqvae.vq.num_embeddings:
        exceptions.show_error(
            f"The VQ-VAE has been initialised with 'num_embeddings' = {vqvae.vq.num_embeddings}, but the model file requires {weight_num_embeddings}."
        )
    
    if weight_embedding_dim != vqvae.vq.embedding_dim:
        exceptions.show_error(
            f"The VQ-VAE has been initialised with 'embedding_dim' = {vqvae.vq.embedding_dim}, but the model file requires {weight_embedding_dim}."
        )

def validate_pixelcnn_params(pixelcnn: PixelCNN, weights: OrderedDict):
    weight_in_params, weight_filters = weights['embedding.weight'].shape
    weight_out_params = weights['output.weight'].shape[0]
    weight_res_blocks = len([k for k in weights.keys() if 'residual_blocks' in k]) // 7
    
    in_params, filters = pixelcnn.embedding.weight.shape
    out_params = pixelcnn.output.weight.shape[0]
    res_blocks = len(pixelcnn.residual_blocks)
    
    if weight_in_params != weight_out_params:
        exceptions.show_error(
            f"The PixelCNN model file has 'in_channels' = {in_params}, and 'out_channels' = {out_params}. They should be equal."
        )
    
    if weight_in_params != in_params:
        exceptions.show_error(
            f"The PixelCNN has been initialised with 'in_channels' = {in_params}, but the model file requires {weight_in_params}. This corresponds to the 'num_embeddings' parameter in the config file."
        )
    
    if weight_out_params != out_params:
        exceptions.show_error(
            f"The PixelCNN has been initialised with 'out_channels' = {out_params}, but the model file requires {weight_out_params}. This corresponds to the 'num_embeddings' parameter in the config file."
        )
    
    if weight_res_blocks != res_blocks:
        exceptions.show_error(
            f"The PixelCNN has been initialised with 'res_blocks' = {res_blocks}, but the model file requires {weight_res_blocks}."
        )
    
    if weight_filters != filters:
        exceptions.show_error(
            f"The PixelCNN has been initialised with 'filters' = {filters}, but the model file requires {weight_filters}."
        )
