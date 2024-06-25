# -*- coding: utf-8 -*-
"""DA2 CS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UqaEm83WuUD6yCIQzsy8iAIU5U37yIpv
"""

# from google.colab import drive
# drive.mount('/content/drive')

# # Unzip data
# !unzip /content/drive/MyDrive/generated_images_10Kids_cropped.zip -d my_data

# !pip install wandb -qU
import wandb
wandb.login()

# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
from torchsummary import summary

class FaceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_pairs = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        people_dirs = sorted(os.listdir(self.image_folder))
        for person_dir in people_dirs:
            person_path = os.path.join(self.image_folder, person_dir)
            images = sorted(os.listdir(person_path))
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    self.image_pairs.append((os.path.join(person_path, images[i]), os.path.join(person_path, images[j])))
                    self.labels.append(1)
                    # Add negative samples
                    neg_person = person_dir
                    while neg_person == person_dir:
                        neg_person = people_dirs[torch.randint(len(people_dirs), (1,)).item()]
                    neg_images = sorted(os.listdir(os.path.join(self.image_folder, neg_person)))
                    self.image_pairs.append((os.path.join(person_path, images[i]), os.path.join(self.image_folder, neg_person, neg_images[0])))
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Siamese Network
class TinySiameseNetwork(nn.Module):
    def __init__(self):
        super(TinySiameseNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1), # double to (1,4...)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1), # double to (4,8...)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*28*28, 8), # double to (8*28*28, 16)
            nn.ReLU(),
            nn.Linear(8, 1), # double to (16,1)
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.conv_net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return torch.abs(output1 - output2)

# Training script
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Using tqdm to display the progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for img1, img2, label in train_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(img1, img2).squeeze()  # Squeeze the output to match the label shape
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        # Save the model
        torch.save(model.state_dict(), f'networks/network_epoch{epoch}.pth')

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Hyperparameters and setup
batch_size = 32
learning_rate = 0.005
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Batch size: {batch_size}')
print(f'LR: {learning_rate}')
print(f'Epochs: {epochs}')
print(f'Device: {device}')

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# Load dataset
# image_folder = 'generated_images_10Kids_cropped'  # Update with the path to your dataset

image_folder = 'start_dataset'  # Update with the path to your dataset
dataset = FaceDataset(image_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, and optimizer
model = TinySiameseNetwork().to(device)

summary(model, [(1, 112, 112), (1, 112, 112)])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, train_loader, criterion, optimizer, epochs=epochs)

