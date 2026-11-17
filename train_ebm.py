import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from app.energy_model import EnergyModel

# ---- Hyperparameters ----
BATCH_SIZE = 128
EPOCHS = 6          # keep this small so training finishes in a reasonable time
STEPS = 40          # number of Langevin Dynamics steps used to generate each "fake" sample
STEP_SIZE = 10
NOISE = 0.005
ALPHA = 0.1         # regularization strength, keeps energy values from exploding
BUFFER_SIZE = 8192  # size of the replay buffer of past generated images
LR = 0.0001

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

model = EnergyModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.0, 0.999))


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
    model.train()

    return inp_imgs.detach()


class Buffer:
    """Keeps a pool of past generated images so sampling doesn't always have
    to start from pure random noise -- this makes training more efficient
    and more stable."""

    def __init__(self, model, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.model = model
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.examples = torch.rand((buffer_size, 3, 32, 32)) * 2 - 1

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = max(1, int(self.batch_size * 0.05))
        n_old = self.batch_size - n_new

        rand_imgs = torch.rand((n_new, 3, 32, 32)) * 2 - 1
        idx = torch.randint(0, self.examples.shape[0], (n_old,))
        old_imgs = self.examples[idx]

        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).to(DEVICE)
        inp_imgs = generate_samples(self.model, inp_imgs, steps=steps,
                                     step_size=step_size, noise=noise)

        replace_idx = torch.randint(0, self.examples.shape[0], (self.batch_size,))
        self.examples[replace_idx] = inp_imgs.detach().cpu()

        return inp_imgs


def train_step(model, optimizer, buffer, real_imgs):
    real_imgs = real_imgs.to(DEVICE)
    real_imgs = real_imgs + NOISE * torch.randn_like(real_imgs)
    real_imgs = real_imgs.clamp(-1.0, 1.0)

    fake_imgs = buffer.sample_new_exmps(steps=STEPS, step_size=STEP_SIZE, noise=NOISE)

    real_out = model(real_imgs)
    fake_out = model(fake_imgs)

    cdiv_loss = real_out.mean() - fake_out.mean()
    reg_loss = ALPHA * (real_out ** 2 + fake_out ** 2).mean()
    loss = cdiv_loss + reg_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()

    return loss.item(), cdiv_loss.item(), reg_loss.item()


if __name__ == "__main__":
    buffer = Buffer(model)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            loss, cdiv, reg = train_step(model, optimizer, buffer, real_imgs)
            total_loss += loss

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"loss={loss:.4f} cdiv={cdiv:.4f} reg={reg:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] finished, avg loss = {avg_loss:.4f} ===")

    torch.save(model.state_dict(), "app/energy_model_weights.pth")
    print("Training complete. Weights saved to app/energy_model_weights.pth")