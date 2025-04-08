import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn_models import EmotionCNN
from utils import get_transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load transforms
train_transform, test_transform = get_transforms()

# Load dataset
full_dataset = ImageFolder(root='data/train', transform=train_transform)
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model
model = EmotionCNN(num_classes=7).to(device)

# Optimizer & Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Hàm vẽ Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

# Training
best_acc = 0.0
for epoch in range(1, 31):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/30], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_acc = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch}/30] - Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / len(val_loader.dataset)
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch}/30] - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'model_weights.pth')
        print("✅ Saved best model")

# Sau huấn luyện: Vẽ confusion matrix cho tập Validation
plot_confusion_matrix(all_labels, all_preds, classes=class_names, title="Validation Confusion Matrix")

# Vẽ confusion matrix cho tập Train
model.eval()
train_preds = []
train_labels = []
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

plot_confusion_matrix(train_labels, train_preds, classes=class_names, title="Train Confusion Matrix")

print(f"✅ Best Validation Accuracy: {best_acc:.4f}")
