import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        # CIFAR-10 images are color images with 3 channels (RGB),
        # unlike the MNIST images (grayscale, 1 channel) used in the practical,
        # so the first conv layer's input channels is changed from 1 to 3.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)
    

def generate_samples(model, inp_imgs, steps, step_size, noise):
    """Langevin Dynamics: starting from random noise images, repeatedly move
    toward lower energy (images the model considers more realistic)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    inp_imgs = inp_imgs.clone()
    inp_imgs.requires_grad = True

    for _ in range(steps):
        inp_imgs.data += noise * torch.randn_like(inp_imgs)
        inp_imgs.data.clamp_(min=-1.0, max=1.0)

        energy = model(inp_imgs)
        grad = torch.autograd.grad(energy.sum(), inp_imgs)[0]
        grad = grad.clamp(-0.03, 0.03)

        inp_imgs.data -= step_size * grad
        inp_imgs.data.clamp_(min=-1.0, max=1.0)

    for p in model.parameters():
        p.requires_grad = True

    return inp_imgs.detach()