import torch
import torchvision.transforms as transforms
from PIL import Image

from models.vanilla_gan import Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect(image_file):

    D = Discriminator().to(device)
    D.load_state_dict(torch.load("weights/vanilla/discriminator.pth", map_location=device))
    D.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    img = Image.open(image_file)
    img = transform(img).view(1, -1).to(device)

    with torch.no_grad():
        output = D(img)

    if output.item() > 0.5:
        return "Real"
    else:
        return "Fake"