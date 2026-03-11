import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_loader import get_dataloader
from models.vanilla_gan import Generator, Discriminator
from utils.noise import generate_noise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
lr = 0.0001
epochs = 30
batch_size = 64


def train():

    dataloader = get_dataloader(batch_size)

    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for real, _ in dataloader:

            real = real.view(-1, 784).to(device)
            batch = real.size(0)

            # Train Discriminator

            noise = generate_noise(batch, z_dim, device)
            fake = G(noise)

            D_real = D(real).view(-1)
            loss_real = criterion(D_real, torch.ones_like(D_real)  * 0.9)

            D_fake = D(fake.detach()).view(-1)
            loss_fake = criterion(D_fake, torch.zeros_like(D_fake))

            loss_D = (loss_real + loss_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator

            output = D(fake).view(-1)
            loss_G = criterion(output, torch.ones_like(output))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1}/{epochs}  Loss_D: {loss_D:.4f}  Loss_G: {loss_G:.4f}")

    torch.save(G.state_dict(), "weights/vanilla/generator.pth")
    torch.save(D.state_dict(), "weights/vanilla/discriminator.pth")

    print("Training complete. Weights saved.")


if __name__ == "__main__":
    train()