import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=100, num_classes=10, img_dim=784):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512,1024),
            nn.ReLU(),

            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        label_embedding = self.label_embed(labels)

        x = torch.cat([noise, label_embedding], dim=1)

        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, num_classes=10, img_dim=784):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(img_dim + num_classes, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):

        label_embedding = self.label_embed(labels)

        x = torch.cat([img, label_embedding], dim=1)

        return self.model(x)