import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
from torchsummary import summary
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Input 112*112 * 8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)

        self.fc1 = nn.Linear(24 * 14 * 14, 36)  # Calculating output size after convolutions
        self.fc1komma5 = nn.Linear(36, 28)
        self.fc2 = nn.Linear(28, 16)
        self.fc3 = nn.Linear(16, 1)  # Output should be 1 for binary classification
        self.dropout = nn.Dropout(0.5)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))  # input 112*112 * 8
        x = F.max_pool2d(x, 2)  # 56*56 * 8
        x = F.relu(self.conv2(x))  # 56*56*16
        x = F.max_pool2d(x, 2)  # 28*28*16
        x = F.relu(self.conv3(x))  # 28*28*24
        x = F.max_pool2d(x, 2)  # 14*14*24
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1komma5(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Applying dropout here
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = torch.abs(output1 - output2)
        output = self.fc3(distance)  # Applying sigmoid for binary classification
        return output


class FaceDataset(Dataset):
    def __init__(self, image_folder, people_dirs, transform=None):
        self.image_folder = image_folder
        self.people_dirs = people_dirs
        self.transform = transform
        self.image_pairs = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        for person_dir in self.people_dirs:
            person_path = os.path.join(self.image_folder, person_dir)
            images = os.listdir(person_path)
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    self.image_pairs.append((os.path.join(person_path, images[i]), os.path.join(person_path, images[j])))
                    self.labels.append(1)

                    neg_person = person_dir
                    while neg_person == person_dir:
                        neg_person = random.choice(self.people_dirs)

                    neg_images = os.listdir(os.path.join(self.image_folder, neg_person))
                    random_image_index = random.randrange(start=0, stop=len(neg_images))
                    self.image_pairs.append((os.path.join(person_path, images[i]), os.path.join(self.image_folder, neg_person, neg_images[random_image_index])))
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

def split_dataset(image_folder, train_ratio=0.94, val_ratio=0.05, test_ratio=0.01):
    people_dirs = os.listdir(image_folder)
    random.shuffle(people_dirs)

    train_end = int(train_ratio * len(people_dirs))
    val_end = train_end + int(val_ratio * len(people_dirs))

    train_dirs = people_dirs[:train_end]
    val_dirs = people_dirs[train_end:val_end]
    test_dirs = people_dirs[val_end:]

    return train_dirs, val_dirs, test_dirs

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    accumulation_steps = 4
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (img1, img2, label) in enumerate(train_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(img1, img2).squeeze()
                    loss = criterion(outputs, label)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item() * accumulation_steps
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)
        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        torch.save(model.state_dict(), f'networks/network_epoch{epoch}.pth')

def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in data_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(img1, img2).squeeze()
                loss = criterion(outputs, label)
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
    accuracy = correct / total
    return running_loss / len(data_loader), accuracy

if __name__ == '__main__':
    wandb.login()

    wandb.init(project='face-recognition-philip')

    batch_size = 512
    learning_rate = 0.02
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "device": str(device)
    })

    print(f'Batch size: {batch_size}')
    print(f'LR: {learning_rate}')
    print(f'Epochs: {epochs}')
    print(f'Device: {device}')

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])

    image_folder = 'generated_images_10Kids_cropped'
    train_dirs, val_dirs, test_dirs = split_dataset(image_folder)

    # Normal transform
    transform_normal = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    
    # Data augmentation and normalization
    transform_data_augmentation = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])
    
    val_dataset = FaceDataset(image_folder, val_dirs, transform=transform_normal)  
    test_dataset = FaceDataset(image_folder, test_dirs, transform=transform_normal)
    #train_dataset_data_augmentation = FaceDataset(image_folder, train_dirs, transform=transform_data_augmentation)
    #train_dataset_normal =  FaceDataset(image_folder, train_dirs, transform=transform_normal)
    
    train_dataset = FaceDataset(image_folder, train_dirs, transform=transform_data_augmentation)
    
    cores = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cores, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=cores, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cores, shuffle=False)

    model = SiameseNetwork().to(device)
    summary(model, [(1, 112, 112), (1, 112, 112)])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = torch.amp.GradScaler()

    train(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    torch.save(model.state_dict(), 'networks/final_network.pth')
