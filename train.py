import os
import torch
from torchvision.io import read_image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set your device (CUDA if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters and constants
MANUAL_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
SHUFFLE = True

# Data Augmentation & Transformation
# Data Augmentation & Transformation
transform = Compose([
    Resize((150, 150)),
    ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Example: normalizing around 0.5 if training followed similar normalization
])



# Create a custom Dataset to load images
class LandscapeDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot
        self.images = os.listdir(f'{self.dataroot}/color')  # Ensure color images exist
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load the corresponding color and grayscale images
        color_img = read_image(f'{self.dataroot}/color/{img_path}') / 255.0
        gray_img = read_image(f'{self.dataroot}/gray/{img_path}') / 255.0

        # Apply transformations, if any
        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        return gray_img, color_img  # Return grayscale input and color target

# Initialize dataset and dataloaders
dataroot = './landscape_Images'
dataset = LandscapeDataset(dataroot=dataroot, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(MANUAL_SEED))

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Define a more advanced autoencoder model (U-Net based)
class UNetAutoEncoder(nn.Module):
    def __init__(self):
        super(UNetAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder (Upsample)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model, loss function, and optimizer
model = UNetAutoEncoder().to(DEVICE)

# Try using SSIM loss for better visual accuracy, in combination with MSE
criterion = nn.MSELoss()  # Can also explore perceptual loss or SSIM loss
optimizer = optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop with learning rate scheduling
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    for gray_img, color_img in tqdm(trainloader, total=len(trainloader)):
        gray_img = gray_img.to(DEVICE)
        color_img = color_img.to(DEVICE)

        # Forward pass
        output = model(gray_img)

        # Calculate loss
        loss = criterion(output, color_img)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Learning rate adjustment
    scheduler.step(running_loss)

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}')

# After training, save the model
torch.save(model.state_dict(), 'color_autoencoder_unet.pth')
print("Model saved successfully!")
