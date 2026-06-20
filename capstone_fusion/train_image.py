import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import f1_score

# ================= CONFIG =================
CSV_FILE = "ifnd_multimodal_clean.csv"
INTERSECTION_CSV = "IFND_intersection.csv"   # <-- NEW

IMAGE_ROOT = "images"

MODEL_DIR = Path("unfiltered_model_visual_gnn_model")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 12
LR = 2e-4
WEIGHT_DECAY = 1e-2
IMG_SIZE = 380

# ================= TRANSFORMS =================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= DATASET =================
class ImageNewsDataset(Dataset):
    def __init__(self, df, tfms):
        self.paths = df["image_path"].values
        self.labels = df["label"].values.astype(int)
        self.tfms = tfms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.tfms(img)
        except:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)

# ================= MODEL =================
class VisualPatchGNN(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 16 nodes

        self.node_proj = nn.Linear(1792, hidden_dim)

        self.gnn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        feat = self.cnn(x)
        feat = self.pool(feat)
        feat = feat.flatten(2).transpose(1, 2)

        nodes = self.node_proj(feat)

        for layer in self.gnn:
            agg = nodes.mean(dim=1, keepdim=True)
            nodes = self.dropout(self.relu(layer(nodes + agg)))

        fused = nodes.mean(dim=1)
        return self.classifier(fused)

# ================= TRAIN / VAL LOOP =================
def run_epoch(model, loader, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    loss_fn = nn.CrossEntropyLoss()

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=train):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = logits.argmax(1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / total, acc, macro_f1

# ================= MAIN =================
def main():
    df = pd.read_csv(CSV_FILE)

    df["label"] = df["label"].map({
        "FALSE": 0, "fake": 0, "Fake": 0,
        "TRUE": 1, "real": 1, "Real": 1
    })

    df = df.dropna(subset=["label", "image_path"])
    df = df.sample(frac=1, random_state=42)

    # -------- NEW: EXCLUDE INTERSECTION FROM TRAIN ONLY --------
    intersect_df = pd.read_csv(INTERSECTION_CSV)

    # ⚠️ CHANGE THESE IF YOUR COLUMN NAMES ARE DIFFERENT
    # If your files use different names, tell me and I'll rewrite this line.
    KEY = "row_id"

    split = int(0.8 * len(df))
    train_df_full, val_df = df[:split], df[split:]

    # Remove intersection samples from TRAINING ONLY
    train_df = train_df_full[~train_df_full[KEY].isin(intersect_df[KEY])]

    print(f"Train before removal: {len(train_df_full)}")
    print(f"Train after removal:  {len(train_df)}")
    print(f"Validation size:      {len(val_df)}")
    # -----------------------------------------------------------

    train_ds = ImageNewsDataset(train_df, train_tfms)
    val_ds = ImageNewsDataset(val_df, val_tfms)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=4)

    model = VisualPatchGNN().to(DEVICE)

    # Freeze backbone for warm-up
    for p in model.cnn.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Unfreeze backbone
        if epoch == 2:
            for p in model.cnn.parameters():
                p.requires_grad = True
            print("🔓 Backbone unfrozen")

        # Learning rate decay (unchanged)
        if epoch == 5:
            for g in optimizer.param_groups:
                g["lr"] = 1e-4
            print("📉 Learning rate reduced to 1e-4")

        tl, ta, tf1 = run_epoch(model, train_loader, optimizer, scaler)
        vl, va, vf1 = run_epoch(model, val_loader)

        print(
            f"Train | Loss: {tl:.4f} | Acc: {ta*100:.2f}% | Macro-F1: {tf1:.4f}"
        )
        print(
            f"Val   | Loss: {vl:.4f} | Acc: {va*100:.2f}% | Macro-F1: {vf1:.4f}"
        )

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("✅ Saved best model")

    print(f"\nTraining completed. Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()