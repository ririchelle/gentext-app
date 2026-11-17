import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from app.diffusion_model import DiffusionModel, offset_cosine_diffusion_schedule

# ---- Hyperparameters ----
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 10       # kept small since a UNet on CIFAR-10 is much heavier than the energy model
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Data: CIFAR-10, normalized to [-1, 1] ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = DiffusionModel(num_channels=NUM_CHANNELS).to(DEVICE)
optimizer = optim.AdamW(model.network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.L1Loss()


def train_step(model, optimizer, loss_fn, images):
    noises = torch.randn_like(images)
    diffusion_times = torch.rand((images.shape[0], 1, 1, 1), device=images.device)
    noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)
    noisy_images = signal_rates * images + noise_rates * noises

    pred_noises, pred_images = model.denoise(noisy_images, noise_rates, signal_rates, use_ema=False)

    loss = loss_fn(pred_noises, noises)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.update_ema()

    return loss.item()


if __name__ == "__main__":
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            loss = train_step(model, optimizer, loss_fn, images)
            total_loss += loss

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] loss={loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] finished, avg loss = {avg_loss:.4f} ===")

    torch.save(model.state_dict(), "app/diffusion_model_weights.pth")
    print("Training complete. Weights saved to app/diffusion_model_weights.pth")