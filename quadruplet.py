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
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 12, 41)  # Updated to 32 * 12 * 12
        self.fc1komma5 = nn.Linear(41,32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)

    def forward_one(self, x):
        x = F.relu(self.conv1(x)) # 8 * 112 * 112
        x = F.max_pool2d(x, 2)  # output size: (8, 56, 56)
        x = F.relu(self.conv2(x)) # 16* 52 * 52
        x = F.max_pool2d(x, 2)  # output size: (16, 26, 26)
        x = F.relu(self.conv3(x)) # 32 * 24 * 24
        x = F.max_pool2d(x, 2)  # output size: (32, 12, 12)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1komma5(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, input1, input2, input3, input4):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)
        output4 = self.forward_one(input4)
        return output1, output2, output3, output4
    
class QuadrupletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative1, negative2):
        distance_pos = F.pairwise_distance(anchor, positive)
        distance_neg1 = F.pairwise_distance(anchor, negative1)
        distance_neg2 = F.pairwise_distance(positive, negative2)

        loss = torch.mean(F.relu(distance_pos - distance_neg1 + self.margin)) + \
               torch.mean(F.relu(distance_pos - distance_neg2 + self.margin))
        return loss


class FaceDataset(Dataset):
    def __init__(self, image_folder, people_dirs, transform=None):
        self.image_folder = image_folder
        self.people_dirs = people_dirs
        self.transform = transform
        self.quadruplets = []
        self._prepare_data()

    def _prepare_data(self):
        for person_dir in self.people_dirs:
            person_path = os.path.join(self.image_folder, person_dir)
            images = os.listdir(person_path)
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    # Anchor and Positive pair
                    anchor = os.path.join(person_path, images[i])
                    positive = os.path.join(person_path, images[j])

                    # Negative samples
                    neg_person1 = person_dir
                    while neg_person1 == person_dir:
                        neg_person1 = random.choice(self.people_dirs)
                    neg_images1 = os.listdir(os.path.join(self.image_folder, neg_person1))
                    negative1 = os.path.join(self.image_folder, neg_person1, random.choice(neg_images1))

                    neg_person2 = person_dir
                    while neg_person2 == person_dir or neg_person2 == neg_person1:
                        neg_person2 = random.choice(self.people_dirs)
                    neg_images2 = os.listdir(os.path.join(self.image_folder, neg_person2))
                    negative2 = os.path.join(self.image_folder, neg_person2, random.choice(neg_images2))

                    self.quadruplets.append((anchor, positive, negative1, negative2))

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative1_path, negative2_path = self.quadruplets[idx]
        anchor = Image.open(anchor_path).convert('L')
        positive = Image.open(positive_path).convert('L')
        negative1 = Image.open(negative1_path).convert('L')
        negative2 = Image.open(negative2_path).convert('L')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative1 = self.transform(negative1)
            negative2 = self.transform(negative2)

        return anchor, positive, negative1, negative2

def split_dataset(image_folder, train_ratio=0.94, val_ratio=0.05, test_ratio=0.01):
    people_dirs = os.listdir(image_folder)
    random.shuffle(people_dirs)

    train_end = int(train_ratio * len(people_dirs))
    val_end = train_end + int(val_ratio * len(people_dirs))

    train_dirs = people_dirs[:train_end]
    val_dirs = people_dirs[train_end:val_end]
    test_dirs = people_dirs[val_end:]

    return train_dirs, val_dirs, test_dirs

# Training script with validation
import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (img1, img2, img3, img4) in enumerate(train_loader):
                img1, img2, img3, img4 = img1.to(device), img2.to(device), img3.to(device), img4.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    # Pass all four inputs to the model
                    output1, output2, output3, output4 = model(img1, img2, img3, img4)

                    # Calculate loss
                    loss = criterion(output1, output2, output3, output4)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss/(i+1))
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
        torch.save(model.state_dict(), f'./network_epoch{epoch}.pth')

# Evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, img3, img4 in data_loader:
            img1, img2, img3, img4 = img1.to(device), img2.to(device), img3.to(device), img4.to(device)
            with torch.cuda.amp.autocast():
                output1, output2, output3, output4 = model(img1, img2, img3, img4)
                loss = criterion(output1, output2, output3, output4)
            running_loss += loss.item()

            # Calculate accuracy (you can adjust this based on your task)
            distance_pos = F.pairwise_distance(output1, output2)
            distance_neg = F.pairwise_distance(output3, output4)
            predicted = (distance_pos < distance_neg).float()  # Adjust as per your task
            # Assuming label is 1 for positive pair and 0 for negative pair
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

    # Load dataset

    image_folder = 'generated_images_10Kids_cropped'  # Update with the path to your dataset
    train_dirs, val_dirs, test_dirs = split_dataset(image_folder)

    #train_dataset_data_augmentation = FaceDataset(image_folder, train_dirs, transform=transform_data_augmentation)
    #train_dataset_normal = FaceDataset(image_folder, train_dirs, transform=transform_normal)

    # train_dataset = ConcatDataset([train_dataset_normal, train_dataset_data_augmentation])
    train_dataset = FaceDataset(image_folder, train_dirs, transform=transform_data_augmentation)
    val_dataset = FaceDataset(image_folder, val_dirs, transform=transform_normal)
    test_dataset = FaceDataset(image_folder, test_dirs, transform=transform_normal)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, and optimizer
    model = SiameseNetwork().to(device)
    summary(model, [(1, 112, 112), (1, 112, 112),(1, 112, 112),(1, 112, 112)])
    #criterion = nn.BCEWithLogitsLoss()
    criterion = QuadrupletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

  
    # Train the model
    #train(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
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
