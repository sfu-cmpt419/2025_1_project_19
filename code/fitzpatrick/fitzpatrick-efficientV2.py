import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# for DL training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm  # For EfficientNet-V2

# 1. CONFIG & HYPERPARAMS
CSV_PATH = "../../datasets/fitzpatrick/fitzpatrick-preprocessed.csv"
IMAGE_DIR = "../../datasets/fitzpatrick/images/"
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-5
NUM_CLASSES = 6  # Fitzpatrick: 1..6 -> we will shift to 0..5 internally
IMAGE_SIZE = 224
RANDOM_SEED = 42

# 2. DATASET DEFINITION
class FitzpatrickDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df must have columns:
          - 'fitzpatrick_scale' in [1..6]
          - 'image_hash' (string)
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_1_to_6 = row['fitzpatrick_scale']
        # shift from 1..6 to 0..5
        label = label_1_to_6 - 1

        image_hash = row['image_hash']
        img_path = os.path.join(self.image_dir, image_hash + '.jpg')

        # Read image using Pillow
        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 3. CREATE DATASETS
df = pd.read_csv(CSV_PATH)

# Split into train/val
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
    stratify=df['fitzpatrick_scale'],
    random_state=RANDOM_SEED
)

# Data augmentation transforms (the paper used flips, rotations, etc.)
# Similar to paper, we specifically avoided color-jitter or hue modifications,
# as color is critical for Fitzpatrick classification.
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_dataset = FitzpatrickDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = FitzpatrickDataset(val_df, IMAGE_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 4. CREATE MODEL
# The paper used EfficientNet-V2M. We use timm.create_model to load it.
# timm models: https://huggingface.co/models?library=timm&sort=trending 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
# Replace the classifier head
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# 5. TRAINING LOOP
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
criterion = nn.CrossEntropyLoss()  # single-scale classification

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # Collect predictions for accuracy
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

@torch.no_grad()
def val_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

best_val_acc = 0.0
patience = 10
epochs_no_improve = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = val_one_epoch(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        # save best model weights
        torch.save(model.state_dict(), "best_efficientnetv2m_fitzpatrick.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# 6. LOAD BEST MODEL & EVALUATE
model.load_state_dict(torch.load("best_efficientnetv2m_fitzpatrick.pt"))
val_loss, val_acc = val_one_epoch(model, val_loader, criterion)
print(f"Best Model Validation Accuracy: {val_acc:.4f}")

# Per-class one-vs-all or macro average
def compute_macro_auc(model, loader):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    # Convert to one-hot
    all_targets_onehot = np.zeros((len(all_targets), NUM_CLASSES))
    for i, lbl in enumerate(all_targets):
        all_targets_onehot[i, lbl] = 1
    # softmax across the 6 classes
    probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
    # compute macro average
    try:
        macro_auc = roc_auc_score(all_targets_onehot, probs, average='macro')
    except ValueError:
        macro_auc = np.nan
    return macro_auc

val_auc = compute_macro_auc(model, val_loader)
print(f"Validation Macro-AUC: {val_auc:.4f}")
