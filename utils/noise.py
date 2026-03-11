import torch

def generate_noise(batch_size, z_dim, device):

    return torch.randn(batch_size, z_dim).to(device)