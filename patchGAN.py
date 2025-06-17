import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path

# -------------------
#  CONFIG
# -------------------

image_size = 64
latent_dim = 100
base_dim = 64
epochs = 100
batch_size = 4
save_dir = "./outputs"
epochcount = 10
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
#  Custom Dataset
# -------------------

class CustomImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# -------------------
#  Generator
# -------------------

class Generator(nn.Module):
    def __init__(self, latent_dim, base_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_dim * 8, 4, 1, 0),  # 1 -> 4
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_dim * 2, base_dim, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(base_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_dim, 3, 4, 2, 1),  # 32 -> 64
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# -------------------
#  PatchGAN Discriminator
# -------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim * 4, base_dim * 8, 4, 1, 1),  # 8 -> 7
            nn.BatchNorm2d(base_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim * 8, 1, 4, 1, 1)  # Patch output
        )

    def forward(self, x):
        return self.model(x)

# -------------------
#  Training
# -------------------

def train(image_dir):
    dataset = CustomImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(latent_dim, base_dim).to(device)
    D = PatchDiscriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_optim = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # --- Train Discriminator ---
            z = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_images = G(z).detach()
            d_real = D(real_images)
            d_fake = D(fake_images)

            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            d_loss_real = criterion(d_real, real_labels)
            d_loss_fake = criterion(d_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            D.zero_grad()
            d_loss.backward()
            d_optim.step()

            # --- Train Generator ---
            z = torch.randn(b_size, latent_dim, 1, 1, device=device)
            gen_images = G(z)
            output = D(gen_images)
            real_labels_for_generator = torch.ones_like(output)

            g_loss = criterion(output, real_labels_for_generator)

            G.zero_grad()
            g_loss.backward()
            g_optim.step()



            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        if epoch % epochcount == 0:
            # Save model and sample images
            torch.save(G.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth"))
        with torch.no_grad():
           samples = G(fixed_noise).cpu()
           save_image(samples * 0.5 + 0.5, os.path.join(save_dir, f"sample_epoch_{epoch+1}.png"), nrow=4)

if __name__ == "__main__":
    train("C:/Users/david/Downloads/framesmedium/1")
