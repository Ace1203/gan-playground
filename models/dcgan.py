import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=100, channels=1, features_g=32):
        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose2d(z_dim, features_g*4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g*2, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, channels=1, features_d=32):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(channels, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d, features_d*2, 4, 2, 1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(features_d*2*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)