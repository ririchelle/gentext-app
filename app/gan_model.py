import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network: maps a random noise vector (shape: batch_size x 100)
    to a 28x28 grayscale image that looks like a handwritten digit.
    """
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        # Fully connected layer: noise vector -> 7*7*128 flat features
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        # ConvTranspose2D: 128 -> 64 channels, 7x7 -> 14x14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # ConvTranspose2D: 64 -> 1 channel, 14x14 -> 28x28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # reshape flat features into (batch, 128, 7, 7)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x  # shape: (batch, 1, 28, 28), pixel values in [-1, 1]


class Discriminator(nn.Module):
    """
    Discriminator network: looks at a 28x28 grayscale image and outputs a
    single probability that the image is real (vs. generated/fake).
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv2D: 1 -> 64 channels, 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        # Conv2D: 64 -> 128 channels, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x  # shape: (batch, 1), probability the image is real