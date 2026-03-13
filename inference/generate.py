import torch
import matplotlib.pyplot as plt
import os

from models.vanilla_gan import Generator as VanillaGenerator
from models.dcgan import Generator as DCGANGenerator
from models.cgan import Generator as CGANGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(model_type, digit=None):

    if model_type == "Vanilla GAN":

        z_dim = 100

        G = VanillaGenerator(z_dim).to(device)
        G.load_state_dict(torch.load("weights/vanilla/generator.pth", map_location=device))
        G.eval()

        noise = torch.randn(1, z_dim).to(device)

        with torch.no_grad():
            fake = G(noise)

        img = fake.view(28,28).cpu().numpy()
        path = "outputs/generated_images/vanilla/sample.png"


    elif model_type == "DCGAN":

        z_dim = 100

        G = DCGANGenerator(z_dim).to(device)
        G.load_state_dict(
            torch.load("weights/cgan/generator.pth", map_location=device),
            strict=False
        )
        G.eval()

        noise = torch.randn(1, z_dim, 1, 1).to(device)

        with torch.no_grad():
            fake = G(noise)

        img = fake.squeeze().cpu().numpy()
        path = "outputs/generated_images/dcgan/sample.png"


    elif model_type == "CGAN":

        z_dim = 100

        G = CGANGenerator(z_dim).to(device)
        G.load_state_dict(
            torch.load("weights/cgan/generator.pth", map_location=device),
            strict=False
        )
        G.eval()

        label = torch.tensor([digit]).to(device)

        noise = torch.randn(1, z_dim).to(device)

        with torch.no_grad():
            fake = G(noise, label)

        img = fake.view(28,28).cpu().numpy()
        path = "outputs/generated_images/cgan/sample.png"


    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path)

    return path