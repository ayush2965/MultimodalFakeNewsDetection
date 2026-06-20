import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIG
# =====================================================
CSV_FILE = "IFND_intersection.csv"

MODEL_PATH = (
    "unfiltered_model_visual_gnn_model/best_model.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
IMG_SIZE = 380

# =====================================================
# TRANSFORMS
# =====================================================
val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASET
# =====================================================
class ImageNewsDataset(Dataset):

    def __init__(self, df, tfms):

        self.paths = df["image_path"].values

        self.labels = (
            df["label"]
            .values
            .astype(int)
        )

        self.tfms = tfms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        try:

            img = Image.open(
                self.paths[idx]
            ).convert("RGB")

            img = self.tfms(img)

        except:

            img = torch.zeros(
                3,
                IMG_SIZE,
                IMG_SIZE
            )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return img, label

# =====================================================
# MODEL
# =====================================================
class VisualPatchGNN(nn.Module):

    def __init__(self, hidden_dim=256):

        super().__init__()

        backbone = models.efficientnet_b4(
            weights="IMAGENET1K_V1"
        )

        self.cnn = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.node_proj = nn.Linear(
            1792,
            hidden_dim
        )

        self.gnn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.4)

        self.classifier = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, x):

        feat = self.cnn(x)

        feat = self.pool(feat)

        feat = feat.flatten(2).transpose(1, 2)

        nodes = self.node_proj(feat)

        for layer in self.gnn:

            agg = nodes.mean(
                dim=1,
                keepdim=True
            )

            nodes = self.dropout(
                self.relu(
                    layer(nodes + agg)
                )
            )

        fused = nodes.mean(dim=1)

        return self.classifier(fused)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_FILE)

df["label"] = df["label"].map({
    "FALSE": 0,
    "fake": 0,
    "Fake": 0,

    "TRUE": 1,
    "real": 1,
    "Real": 1
})

df = df.dropna(
    subset=["label", "image_path"]
)

# =====================================================
# FORCE EXACTLY 663 BATCHES
# =====================================================
df = df.iloc[:5304]

print(f"\nValidation samples: {len(df)}")

val_ds = ImageNewsDataset(
    df,
    val_tfms
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    drop_last=True
)

print(f"Validation batches: {len(val_loader)}")

# =====================================================
# LOAD MODEL
# =====================================================
model = VisualPatchGNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# =====================================================
# EVALUATION
# =====================================================
all_preds = []
all_labels = []

with torch.no_grad():

    for imgs, labels in tqdm(
        val_loader,
        desc="Evaluating"
    ):

        imgs = imgs.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(imgs)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# =====================================================
# METRICS
# =====================================================
acc = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

# =====================================================
# RESULTS
# =====================================================
print("\n========== FINAL EVALUATION ==========\n")

print(f"Samples Used : {len(all_labels)}")

print(f"Accuracy     : {acc:.4f}")

print(f"Precision    : {precision:.4f}")

print(f"Recall       : {recall:.4f}")

print(f"F1-Score     : {f1:.4f}")

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)