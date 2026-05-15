import torch
import torch.nn.functional as F
import tqdm

def optimizer(cnn, learning_rate=3e-4):
    """Constructs an optimizer for the PixelCNN."""
    return torch.optim.Adam(cnn.parameters(), lr=learning_rate)

def train(model, dataloader, optimizer, device):
    """Trains the PixelCNN for a single epoch"""
    model.train()
    
    total_loss = 0.0
    
    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        z = batch.to(device)
        
        optimizer.zero_grad()

        logits = model(z)   # (B, K, H, W)
        
        loss = F.cross_entropy(logits, z)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
