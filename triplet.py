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
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network for face recognition.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 12, 41)
        self.fc2 = nn.Linear(41, 32)
        self.fc3 = nn.Linear(32, 16)

        # Total params: 198777
        
    def forward_one(self, x):
        """
        Forward pass for one input in the Siamese Network.
        """
        x = F.relu(self.conv1(x))  # input 8 * 112 * 112
        x = F.max_pool2d(x, 2)  # 8 * 56 * 56
        x = F.relu(self.conv2(x))  # 16 * 52 * 52
        x = F.max_pool2d(x, 2)  # 16 * 26 * 26
        x = F.relu(self.conv3(x))  # 32 * 24 * 24
        x = F.max_pool2d(x, 2)  # 32 * 14 * 14
        
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, input1, input2, input3):
        """
        Forward pass for three inputs in the Siamese Network.
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)
        return output1, output2, output3

class FaceDataset(Dataset):
    """
    Custom Dataset class for face images.
    """
    def __init__(self, image_folder, people_dirs, transform=None):
        self.image_folder = image_folder
        self.people_dirs = people_dirs
        self.transform = transform
        self.triplets = []
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare the triplets of images (anchor, positive, negative).
        """
        for person_dir in self.people_dirs:
            person_path = os.path.join(self.image_folder, person_dir)
            images = os.listdir(person_path)
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    # Anchor and Positive pair
                    anchor = os.path.join(person_path, images[i])
                    positive = os.path.join(person_path, images[j])

                    # Negative samples
                    neg_person = person_dir
                    while neg_person == person_dir:
                        neg_person = random.choice(self.people_dirs)
                    neg_images = os.listdir(os.path.join(self.image_folder, neg_person))
                    negative = os.path.join(self.image_folder, neg_person, random.choice(neg_images))

                    self.triplets.append((anchor, positive, negative))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        """
        anchor_path, positive_path, negative1_path = self.triplets[idx]
        anchor = Image.open(anchor_path).convert('L')
        positive = Image.open(positive_path).convert('L')
        negative = Image.open(negative1_path).convert('L')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

def split_dataset(image_folder, train_ratio=0.94, val_ratio=0.05, test_ratio=0.01):
    """
    Split the dataset into training, validation, and test sets.
    """
    people_dirs = os.listdir(image_folder)
    random.shuffle(people_dirs)

    train_end = int(train_ratio * len(people_dirs))
    val_end = train_end + int(val_ratio * len(people_dirs))

    train_dirs = people_dirs[:train_end]
    val_dirs = people_dirs[train_end:val_end]
    test_dirs = people_dirs[val_end:]

    return train_dirs, val_dirs, test_dirs

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs):
    """
    Train the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (img1, img2, img3) in enumerate(train_loader):
                img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    output1, output2, output3 = model(img1, img2, img3)
                    loss = criterion(output1, output2, output3)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (i + 1))
                pbar.update(1)

        scheduler.step()

        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Save the model
        torch.save(model.state_dict(), f'./networks/triplet/network_epoch{epoch}.pth')

def evaluate(model, data_loader, criterion):
    """
    Evaluate the model.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, img3 in data_loader:
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            with torch.amp.autocast('cuda'):
                output1, output2, output3 = model(img1, img2, img3)
                loss = criterion(output1, output2, output3)
            running_loss += loss.item()

            # Calculate accuracy (you can adjust this based on your task)
            distance_pos = F.pairwise_distance(output1, output2)
            distance_neg = F.pairwise_distance(output1, output3)
            predicted = (distance_pos < distance_neg).float()
            correct += (predicted == 1).sum().item()
            total += img1.size(0)

    accuracy = correct / total
    return running_loss / len(data_loader), accuracy

if __name__ == '__main__':
    wandb.login()

    # Initialize wandb
    wandb.init(project='face-recognition-project')

    # Hyperparameters and setup
    batch_size = 256
    learning_rate = 0.02
    epochs = 25
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

    # Data augmentation and normalization
    transform_data_augmentation = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])
    transform_normal = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    # Load dataset
    image_folder = 'generated_images_10Kids_cropped'  # Update with the path to your dataset
    train_dirs, val_dirs, test_dirs = split_dataset(image_folder)

    train_dataset = FaceDataset(image_folder, train_dirs, transform=transform_data_augmentation)
    val_dataset = FaceDataset(image_folder, val_dirs, transform=transform_normal)
    test_dataset = FaceDataset(image_folder, test_dirs, transform=transform_normal)

    cores = int(os.cpu_count() * 0.75)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cores, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=cores, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cores, shuffle=False, pin_memory=True)

    # Model, loss, and optimizer
    model = SiameseNetwork().to(device)

    criterion = nn.TripletMarginLoss(margin=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs=epochs)

    # Evaluate on test set
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Log final test metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    # Finish wandb run
    wandb.finish()
