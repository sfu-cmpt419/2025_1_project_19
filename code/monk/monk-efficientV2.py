import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# For deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm  # for EfficientNet-V2

# -------------------------------
# 1. CONFIG & HYPERPARAMS
# -------------------------------
CSV_PATH = "../../datasets/monk/monk-preprocessed.csv"
IMAGE_DIR = "../../datasets/monk/images/"
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-5
NUM_CLASSES = 10       # For Monk Skin Tone dataset with 10 classes
IMAGE_SIZE = 224
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-7    # L2 regularization

# -------------------------------
# 2. DATASET DEFINITION
# -------------------------------
class MonkSkinDataset(Dataset):
    """
    Adjust column names to match your CSV:
      - e.g., 'monk_scale' or 'skin_tone_class' for labels
      - 'filename', 'image_path', or similar for image filenames
    
    If your CSV has labels from 1..10,
    do: label = row['skin_tone_class'] - 1.
    Otherwise, if 0..9, use them as is.
    """
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Example: 'label' column in 0..9
        label = row['MST'] -1

        image_filename = row['image_ID'].strip()  # or whatever column your CSV uses
        img_path = os.path.join(self.image_dir, image_filename)

        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------------
# 3. CREATE DATASETS
# -------------------------------
df = pd.read_csv(CSV_PATH)

# Train/Val split (80-20) with stratification
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
    stratify=df['MST'],  # or your label column name
    random_state=RANDOM_SEED
)

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

train_dataset = MonkSkinDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = MonkSkinDataset(val_df, IMAGE_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -------------------------------
# 4. CREATE MODEL
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
# Replace the classifier head for 10 classes
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# -------------------------------
# 5. TRAINING LOOP
# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

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
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

best_val_acc = 0.0
patience = 5  # how many epochs with no improvement before stopping
epochs_no_improve = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = val_one_epoch(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), "best_efficientnetv2m_monk.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# -------------------------------
# 6. LOAD BEST MODEL & EVALUATE
# -------------------------------
model.load_state_dict(torch.load("best_efficientnetv2m_monk.pt"))
val_loss, val_acc = val_one_epoch(model, val_loader, criterion)
print(f"Best Model Validation Accuracy: {val_acc:.4f}")

@torch.no_grad()
def compute_macro_auc(model, loader):
    model.eval()
    all_logits = []
    all_targets = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)

    # One-hot encode the labels
    all_targets_onehot = np.zeros((len(all_targets), NUM_CLASSES))
    for i, lbl in enumerate(all_targets):
        all_targets_onehot[i, lbl] = 1

    # Softmax for class probabilities
    exp_logits = np.exp(all_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Macro-AUC
    try:
        macro_auc = roc_auc_score(all_targets_onehot, probs, average='macro')
    except ValueError:
        macro_auc = np.nan
    return macro_auc

val_auc = compute_macro_auc(model, val_loader)
print(f"Validation Macro-AUC: {val_auc:.4f}")
