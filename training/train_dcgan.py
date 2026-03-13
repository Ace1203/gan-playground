import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_loader import get_dataloader
from models.dcgan import Generator, Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
lr = 0.0002
epochs = 100
batch_size = 32


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def train():

    dataloader = get_dataloader(batch_size)

    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)

    # Initialize weights ONCE
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for real, _ in dataloader:

            real = real.to(device)
            batch = real.size(0)

            noise = torch.randn(batch, z_dim, 1, 1).to(device)

            fake = G(noise)

            # Train Discriminator

            D_real = D(real).view(-1)
            loss_real = criterion(D_real, torch.ones_like(D_real) * 0.9)

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

        print(f"Epoch {epoch+1}/{epochs} Loss_D:{loss_D:.4f} Loss_G:{loss_G:.4f}")

    torch.save(G.state_dict(), "weights/dcgan/generator.pth")
    torch.save(D.state_dict(), "weights/dcgan/discriminator.pth")

    print("DCGAN Training Complete")


if __name__ == "__main__":
    train()