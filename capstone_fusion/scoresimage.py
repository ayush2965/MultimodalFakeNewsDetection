import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# IMPORT YOUR EXISTING STUFF
from train_image import (
    VisualPatchGNN,
    ImageNewsDataset,
    val_tfms,
    DEVICE,
    BATCH_SIZE
)

# ================= CONFIG =================
CSV_FILE = "D:\\capstone_fusion\\capstone_fusion\\ifnd_multimodal_clean.csv"
MODEL_PATH = Path("D:\\capstone_fusion\\capstone_fusion\\unfiltered_model_visual_gnn_model\\best_model.pt")

# ================= LOAD DATA =================
df = pd.read_csv(CSV_FILE)

df["label"] = df["label"].map({
    "FALSE": 0, "fake": 0, "Fake": 0,
    "TRUE": 1, "real": 1, "Real": 1
})

df = df.dropna(subset=["label", "image_path"])

# SAME SPLIT USED DURING TRAINING
split = int(0.8 * len(df))
val_df = df[split:]

val_ds = ImageNewsDataset(val_df, val_tfms)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ================= LOAD MODEL =================
model = VisualPatchGNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

# ================= EVALUATION =================
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ================= METRICS =================
acc = accuracy_score(all_labels, all_preds)

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

cm = confusion_matrix(all_labels, all_preds)

# ================= PRINT =================
print("\n========== FINAL MODEL EVALUATION ==========")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)