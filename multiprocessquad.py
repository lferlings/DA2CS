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

class QuadrupletNetwork(nn.Module):
    def __init__(self):
        super(QuadrupletNetwork, self).__init__()
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
    def __init__(self, margin1=1.0, margin2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist1 = F.pairwise_distance(anchor, negative1)
        neg_dist2 = F.pairwise_distance(anchor, negative2)
        loss = F.relu(pos_dist - neg_dist1 + self.margin1) + F.relu(pos_dist - neg_dist2 + self.margin2)
        return loss.mean()

class QuadrupletFaceDataset(Dataset):
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
                    anchor_image = images[i]
                    positive_image = images[j]

                    neg_person_1 = person_dir
                    while neg_person_1 == person_dir:
                        neg_person_1 = random.choice(self.people_dirs)

                    neg_images_1 = os.listdir(os.path.join(self.image_folder, neg_person_1))
                    random_image_index_1 = random.randrange(start=0, stop=len(neg_images_1))
                    negative_image_1 = neg_images_1[random_image_index_1]

                    neg_person_2 = person_dir
                    while neg_person_2 == person_dir or neg_person_2 == neg_person_1:
                        neg_person_2 = random.choice(self.people_dirs)

                    neg_images_2 = os.listdir(os.path.join(self.image_folder, neg_person_2))
                    random_image_index_2 = random.randrange(start=0, stop=len(neg_images_2))
                    negative_image_2 = neg_images_2[random_image_index_2]

                    self.quadruplets.append((os.path.join(person_path, anchor_image),
                                             os.path.join(person_path, positive_image),
                                             os.path.join(self.image_folder, neg_person_1, negative_image_1),
                                             os.path.join(self.image_folder, neg_person_2, negative_image_2)))

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path_1, negative_path_2 = self.quadruplets[idx]
        anchor = Image.open(anchor_path).convert('L')
        positive = Image.open(positive_path).convert('L')
        negative_1 = Image.open(negative_path_1).convert('L')
        negative_2 = Image.open(negative_path_2).convert('L')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative_1 = self.transform(negative_1)
            negative_2 = self.transform(negative_2)

        return anchor, positive, negative_1, negative_2

def split_dataset(image_folder, train_ratio=0.94, val_ratio=0.05, test_ratio=0.01):
    people_dirs = os.listdir(image_folder)
    random.shuffle(people_dirs)

    train_end = int(train_ratio * len(people_dirs))
    val_end = train_end + int(val_ratio * len(people_dirs))

    train_dirs = people_dirs[:train_end]
    val_dirs = people_dirs[train_end:val_end]
    test_dirs = people_dirs[val_end:]

    return train_dirs, val_dirs, test_dirs

def train_quadruplet(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    accumulation_steps = 4
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (anchor, positive, negative1, negative2) in enumerate(train_loader):
                anchor, positive, negative1, negative2 = anchor.to(device), positive.to(device), negative1.to(device), negative2.to(device)
                with torch.cuda.amp.autocast():
                    anchor_out, positive_out, negative1_out, negative2_out = model(anchor, positive, negative1, negative2)
                    loss = criterion(anchor_out, positive_out, negative1_out, negative2_out)
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

        # Validation step
        val_accuracy = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Accuracy: {val_accuracy}")
        wandb.log({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader), "val_accuracy": val_accuracy})

        # Save model checkpoint
        torch.save(model.state_dict(), f'networks/network_epoch{epoch}.pth')

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

    wandb.init(project='face-recognition-philip')

    batch_size = 128
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
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])

    image_folder = 'generated_images_10Kids_cropped'
    train_dirs, val_dirs, test_dirs = split_dataset(image_folder)

    transform_normal = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    train_dataset = QuadrupletFaceDataset(image_folder, train_dirs, transform=transform)
    val_dataset = QuadrupletFaceDataset(image_folder, val_dirs, transform=transform_normal)
    test_dataset = QuadrupletFaceDataset(image_folder, test_dirs, transform=transform_normal)

    cores = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cores, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=cores, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cores, shuffle=False, pin_memory=True)

    model = QuadrupletNetwork().to(device)
    summary(model, [(1, 112, 112), (1, 112, 112), (1, 112, 112), (1, 112, 112)])
    criterion = QuadrupletLoss(margin1=1.0, margin2=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    train_quadruplet(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

    test_accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_accuracy}')
    wandb.log({"test_accuracy": test_accuracy})
    torch.save(model.state_dict(), 'networks/final_network.pth')
