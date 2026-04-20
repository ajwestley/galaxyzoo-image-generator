import re
import torch
from VQ_VAE import VQVAE

def load_model(
    model_file: str, 
    embedding_dim: int, 
    num_embeddings: int, 
    commitment: float,
    verbose: bool = False
    ):
    
    model = VQVAE(embedding_dim, num_embeddings, commitment)
    
    try:
        model.load_state_dict(torch.load(model_file, weights_only=True))
    except FileNotFoundError as e:
        print(f'No model file found at {model_file}.')
        if verbose:
            print(e)
    except RuntimeError as e:
        print('There was an error loading the model')
        error_str = str(e)
        if 'encoder' in error_str:
            dim = re.findall(r'Size\(\[(\d+)\]\)', error_str)[0]
            if int(dim) != embedding_dim:
                print(f'The provided model file requires an embedding dimension of {dim}, but has {embedding_dim}.')
                print(f'Try executing with "-d {dim}"')
        elif 'cluster_size' in error_str:
            num = re.findall(r'Size\(\[(\d+)\]\)', error_str)[0]
            if int(num) != num_embeddings:
                print(f'The provided model file requires {num} embeddings, but has {num_embeddings}.')
                print(f'Try executing with "-n {num}"')
        if verbose:
            print(e)

    return model