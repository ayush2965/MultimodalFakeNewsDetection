import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_DIR = "bert_lstm_end2end"
IMAGE_MODEL_PATH = Path("unfiltered_model_visual_gnn_model/best_model.pt")
BATCH_SIZE = 8
MAX_LEN = 256

# ---------------- LABEL MAP (IMAGE MODEL FORMAT) ----------------
LABEL_MAP = {
    "FALSE": 0, "fake": 0, "Fake": 0,
    "TRUE": 1,  "real": 1, "Real": 1
}

# ---------------- TEXT DATASET ----------------
class IFNDDatasetText:
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["Statement"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ---------------- LOAD MODELS ----------------
from train import BertLSTMForSequenceClassification
from train_image import VisualPatchGNN, ImageNewsDataset, val_tfms

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load text model
text_model = BertLSTMForSequenceClassification().to(DEVICE)
text_model.load_state_dict(
    torch.load(Path(TEXT_MODEL_DIR) / "best_model_state_dict.pt", map_location=DEVICE)
)
text_model.eval()

# Load image model
image_model = VisualPatchGNN().to(DEVICE)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
image_model.eval()

# ---------------- LOAD FULL DATASET ----------------
df = pd.read_csv("IFND_intersection.csv")

df = df.dropna(subset=["label", "image_path", "Statement"])
df["label"] = df["label"].map(LABEL_MAP)

print(f"Testing on {len(df)} samples")

# ---------------- CREATE DATALOADERS ----------------
text_dataset = IFNDDatasetText(df, tokenizer, MAX_LEN)
image_dataset = ImageNewsDataset(df, val_tfms)

text_loader = DataLoader(
    text_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

image_loader = DataLoader(
    image_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------- LATE FUSION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for (text_batch, img_batch) in zip(text_loader, image_loader):

        # Text inputs
        input_ids = text_batch["input_ids"].to(DEVICE)
        attention_mask = text_batch["attention_mask"].to(DEVICE)
        labels = text_batch["labels"].to(DEVICE)

        # Image inputs
        imgs, _ = img_batch
        imgs = imgs.to(DEVICE)

        # Forward passes
        text_logits = text_model(input_ids, attention_mask)

        # 🔹 KEY FIX: align text model labels with image model (NO RETRAINING)
        text_logits = torch.flip(text_logits, dims=[1])

        img_logits = image_model(imgs)

        # ---- LATE FUSION IN PROBABILITY SPACE (BEST PRACTICE) ----
        text_probs = F.softmax(text_logits, dim=1)
        img_probs = F.softmax(img_logits, dim=1)

        alpha = 0.85  # 85% text, 15% image
        fused_probs = alpha * text_probs + (1 - alpha) * img_probs
        preds = fused_probs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# Concatenate all results
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# ---------------- METRICS ----------------
accuracy = (all_preds == all_labels).float().mean().item()

from sklearn.metrics import f1_score
macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")

print("\nLate Fusion (Full 5309 dataset)")
print(f"Samples: {len(all_labels)}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")