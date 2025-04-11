import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# 1. CONFIG & HYPERPARAMS
CSV_PATH = "../../datasets/fitzpatrick/fitzpatrick-preprocessed.csv"
IMAGE_DIR = "../../datasets/fitzpatrick/images/"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NUM_CLASSES = 6  # Fitzpatrick scale: 1..6 -> will shift to 0..5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# 2. DATASET DEFINITION
class FitzpatrickDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        Expects df with:
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
        # shift from [1..6] to [0..5]
        label = row['fitzpatrick_scale'] - 1

        # build image path
        image_hash = row['image_hash']
        img_path = os.path.join(self.image_dir, image_hash + ".jpg")

        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# 3. MAIN FUNCTION
def main():
    df = pd.read_csv(CSV_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        stratify=df['fitzpatrick_scale'],  # ensure balanced splits
        random_state=RANDOM_SEED
    )

    # We avoid color-jitter because color is crucial for Fitzpatrick classification.
    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # creating datsets and loaders
    train_ds = FitzpatrickDataset(train_df, IMAGE_DIR, transform=train_transform)
    val_ds   = FitzpatrickDataset(val_df,   IMAGE_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # create model (ResNet-50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # or older PyTorch versions: model = models.resnet50(pretrained=True)
    
    # replace final FC layer for 6 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    model.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)

    # training loop
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_preds = np.concatenate(all_preds)
        train_targets = np.concatenate(all_targets)
        train_acc = accuracy_score(train_targets, train_preds)

        # validation step
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_targets = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * imgs.size(0)

                preds = torch.argmax(logits, dim=1)
                val_all_preds.append(preds.cpu().numpy())
                val_all_targets.append(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_preds = np.concatenate(val_all_preds)
        val_targets = np.concatenate(val_all_targets)
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_resnet50_fitzpatrick.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 4. Load Best Model & Evaluate
    model.load_state_dict(torch.load("best_resnet50_fitzpatrick.pt"))
    model.eval()

    # final val metrics
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Final Best Model -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # confusion matrix
    all_preds, all_labels = predict_labels(model, val_loader, device)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # macro-avg AUC (one-vs-rest)
    macro_auc = compute_macro_auc(model, val_loader, device, num_classes=NUM_CLASSES)
    print(f"Validation Macro-AUC: {macro_auc:.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def predict_labels(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels

def compute_macro_auc(model, loader, device, num_classes=6):
    """
    For multi-class classification, compute the macro-average ROC AUC
    by one-vs-rest approach.
    """
    from sklearn.metrics import roc_auc_score
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

    # one-hot encoding
    one_hot = np.zeros((len(all_targets), num_classes))
    for i, lbl in enumerate(all_targets):
        one_hot[i, lbl] = 1

    # softmax to get probabilities
    exp_logits = np.exp(all_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # macro-average AUC
    try:
        macro_auc = roc_auc_score(one_hot, probs, average='macro')
    except ValueError:
        macro_auc = float('nan')
    return macro_auc

if __name__ == "__main__":
    main()
