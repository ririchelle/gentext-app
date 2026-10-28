import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.gan_model import Generator, Discriminator

# Device setup (uses Apple Silicon GPU if available, otherwise CPU)
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# Hyperparameters
Z_DIM = 100
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.0002

# MNIST images, normalized to [-1, 1] to match the Generator's Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator = Generator(z_dim=Z_DIM).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
opt_gen = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---- Train Discriminator: learn to tell real from fake ----
        noise = torch.randn(batch_size, Z_DIM).to(device)
        fake_images = generator(noise)

        disc_real = discriminator(real_images)
        loss_real = criterion(disc_real, real_labels)

        disc_fake = discriminator(fake_images.detach())
        loss_fake = criterion(disc_fake, fake_labels)

        loss_disc = loss_real + loss_fake

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ---- Train Generator: try to fool the Discriminator ----
        disc_fake_for_gen = discriminator(fake_images)
        loss_gen = criterion(disc_fake_for_gen, real_labels)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"D loss: {loss_disc.item():.4f} G loss: {loss_gen.item():.4f}")

# Save only the Generator's weights — that's all the API needs to generate new digits
torch.save(generator.state_dict(), "app/gan_generator_weights.pth")
print("Training complete. Generator weights saved to app/gan_generator_weights.pth")