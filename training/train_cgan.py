import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_loader import get_dataloader
from models.cgan import Generator, Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
epochs = 20
lr = 0.0002
batch_size = 32


def train():

    dataloader = get_dataloader(batch_size)

    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    for epoch in range(epochs):

        for real, labels in dataloader:

            real = real.view(-1, 784).to(device)
            labels = labels.to(device)

            batch = real.size(0)

            noise = torch.randn(batch, z_dim).to(device)

            fake = G(noise, labels)

            # Train Discriminator

            D_real = D(real, labels).view(-1)
            loss_real = criterion(D_real, torch.ones_like(D_real) * 0.85)

            D_fake = D(fake.detach(), labels).view(-1)
            loss_fake = criterion(D_fake, torch.zeros_like(D_fake))

            loss_D = (loss_real + loss_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator

            output = D(fake, labels).view(-1)

            loss_G = criterion(output, torch.ones_like(output))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1}/{epochs} Loss_D:{loss_D:.4f} Loss_G:{loss_G:.4f}")

    torch.save(G.state_dict(), "weights/cgan/generator.pth")
    torch.save(D.state_dict(), "weights/cgan/discriminator.pth")

    print("CGAN Training Complete")


if __name__ == "__main__":
    train()