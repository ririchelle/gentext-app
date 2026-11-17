import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


# ---------------- Diffusion schedule: converts "progress" (0 to 1) into signal/noise ratios ----------------
def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


# ---------------- Sinusoidal embedding: turns a noise-level scalar into a vector ----------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer(
            "angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x):
        # x has shape (batch, 1, 1, 1): the noise variance for each image in the batch
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


# ---------------- UNet building blocks ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.residual_proj(x)
        x = swish(self.norm1(x))
        x = self.conv1(x)
        x = swish(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Progressively shrinks the image (downsampling) while saving intermediate
    feature maps into the skips list, to be reused later during upsampling."""

    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        blocks = []
        for i in range(block_depth):
            blocks.append(ResidualBlock(in_channels if i == 0 else out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """Progressively enlarges the image (upsampling) and concatenates the
    saved skip connections back in to recover detail."""

    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        blocks = []
        for i in range(block_depth):
            prev_channels = in_channels if i == 0 else out_channels
            blocks.append(ResidualBlock(prev_channels + out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        x = self.upsample(x)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


# ---------------- Full UNet ----------------
class UNet(nn.Module):
    def __init__(self, num_channels=3, num_frequencies=16, block_depth=2):
        super().__init__()
        embed_channels = num_frequencies * 2  # sin half + cos half

        self.initial_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies)

        self.down1 = DownBlock(32 + embed_channels, 32, block_depth)
        self.down2 = DownBlock(32, 64, block_depth)
        self.down3 = DownBlock(64, 96, block_depth)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(128, 96, block_depth)
        self.up2 = UpBlock(96, 64, block_depth)
        self.up3 = UpBlock(64, 32, block_depth)

        self.final_conv = nn.Conv2d(32, num_channels, kernel_size=1)
        # Zero-initialize the final layer: at the start of training the predicted
        # noise is 0, which makes early training more stable.
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, noisy_images, noise_variances):
        x = self.initial_conv(noisy_images)

        emb = self.embedding(noise_variances)       # (batch, 1, 1, embed_channels)
        emb = emb.permute(0, 3, 1, 2)                  # (batch, embed_channels, 1, 1)
        emb = F.interpolate(emb, size=x.shape[-2:], mode="nearest")
        x = torch.cat([x, emb], dim=1)

        skips = []
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final_conv(x)


# ---------------- Diffusion model wrapper: trainable network + EMA network ----------------
class DiffusionModel(nn.Module):
    def __init__(self, num_channels=3, ema_decay=0.8):
        super().__init__()
        self.network = UNet(num_channels=num_channels)
        # EMA (exponential moving average) network: a "smoothed" copy of the
        # weights, used for generation since it tends to produce more stable results.
        self.ema_network = UNet(num_channels=num_channels)
        self.ema_network.load_state_dict(self.network.state_dict())
        for p in self.ema_network.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_p.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    def denoise(self, noisy_images, noise_rates, signal_rates, use_ema=True):
        network = self.ema_network if use_ema else self.network
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Start from pure noise and iteratively denoise to produce a generated image."""
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        device = initial_noise.device

        next_noisy_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = torch.ones((num_images, 1, 1, 1), device=device) - step * step_size
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, use_ema=True)
            max_val = pred_images.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            pred_images = pred_images / max_val

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule(next_diffusion_times)

            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=20, image_size=32, device="cpu"):
        initial_noise = torch.randn((num_images, 3, image_size, image_size), device=device)
        return self.reverse_diffusion(initial_noise, diffusion_steps)