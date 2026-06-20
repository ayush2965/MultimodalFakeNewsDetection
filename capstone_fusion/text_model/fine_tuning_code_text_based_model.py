# imports
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel,  get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

#dataset loading
DATA_CSV = "IFND.csv"
#directory to save model and tokenizer
MODEL_DIR = "bert_lstm_end2end"
#pretrained model BERT model
PRETRAINED = "bert-base-uncased"
#label mapping
LABEL_MAP = {"TRUE": 0, "Fake": 1}

MAX_LEN = 256            # 256 usually a good compromise; reduce to 128 to speed up
BATCH_SIZE = 16          # try 16 for MAX_LEN=256 on 5070Ti; reduce if OOM
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 100       # linear warmup steps
UNFREEZE_LAST_N = 4      # number of last encoder layers to unfreeze; set to 0 to freeze all BERT
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# Dataset
# -------------------------
class IFNDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 256):
        self.texts = df["Statement"].astype(str).tolist()
        self.labels = df["Label"].tolist()
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
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),       # shape: [max_len]
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# Model: BERT -> BiLSTM -> classifier
# -------------------------
class BertLSTMForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name=PRETRAINED, lstm_hidden=256, lstm_layers=1, bidirectional=True, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # typically 768

        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # BERT returns last_hidden_state: (batch, seq_len, hidden)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_out.last_hidden_state          # (B, seq_len, hidden)
        lstm_out, _ = self.lstm(token_embeddings)             # (B, seq_len, hidden*dir)
        lengths = attention_mask.sum(dim=1) - 1   
        last_timestep = lstm_out[torch.arange(lstm_out.size(0)), lengths, :]                # take last timestep
        logits = self.classifier(last_timestep)
        return logits


# -------------------------
# Metrics (accuracy + macro-F1)
# -------------------------
def compute_metrics(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    # macro F1
    num_classes = int(max(labels.max().item(), preds.max().item()) + 1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    return correct, total, macro_f1


# -------------------------
# Utility: unfreeze last N bert encoder layers
# -------------------------
def unfreeze_bert_layers(model: BertLSTMForSequenceClassification, last_n: int):
    # freeze all first
    for p in model.bert.parameters():
        p.requires_grad = False
    if last_n <= 0:
        return
    # unfreeze pooler and encoder last_n layers
    # encoder layers are under model.bert.encoder.layer (list)
    n_layers = len(model.bert.encoder.layer)
    start = max(0, n_layers - last_n)
    for i in range(start, n_layers):
        for p in model.bert.encoder.layer[i].parameters():
            p.requires_grad = True
    # optionally unfreeze pooler
    for p in model.bert.pooler.parameters():
        p.requires_grad = True


# -------------------------
# Train / Eval loops
# -------------------------
def train_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = running_loss / (len(dataloader.dataset))
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    correct, total, macro_f1 = compute_metrics(all_preds, all_labels)
    accuracy = correct / total
    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def eval_epoch(model, dataloader):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

        running_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = running_loss / (len(dataloader.dataset))
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    correct, total, macro_f1 = compute_metrics(all_preds, all_labels)
    accuracy = correct / total
    return avg_loss, accuracy, macro_f1


# -------------------------
# Main
# -------------------------
def main():
    print(f"Device: {DEVICE}; PyTorch: {torch.__version__}")
    df = pd.read_csv(DATA_CSV, encoding="MacRoman")
    df["Label"] = df["Label"].map(LABEL_MAP)

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    dataset = IFNDDataset(df, tokenizer, max_len=MAX_LEN)

    # train/val split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    model = BertLSTMForSequenceClassification(pretrained_model_name=PRETRAINED).to(DEVICE)

    # Freeze all BERT weights first, then unfreeze last N layers (and pooler)
    for p in model.bert.parameters():
        p.requires_grad = False
    unfreeze_bert_layers(model, UNFREEZE_LAST_N)
    print(f"Unfrozen last {UNFREEZE_LAST_N} BERT layers (and pooler).")

    # Optimizer only for parameters that require grad
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    scaler = GradScaler()
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader)
        t1 = time.time()

        print(f"\nEpoch {epoch+1}/{EPOCHS} ({(t1-t0)/60:.2f} min)")
        print(f"Train loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | F1_mac: {train_f1:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1_mac: {val_f1:.4f}")

        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New best model — saving...")
            # save full state dict
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model_state_dict.pt"))
            # also save tokenizer and (optionally) bert + classifier via huggingface format
            model.bert.save_pretrained(MODEL_DIR)
            # Save a small wrapper config (not strictly necessary)
            torch.save({
                "model_class": "BertLSTMForSequenceClassification",
                "max_len": MAX_LEN,
                "lstm_hidden": model.lstm.hidden_size,
                "bidirectional": model.lstm.bidirectional,
            }, os.path.join(MODEL_DIR, "model_meta.pt"))

    print("Training complete. Best val F1 (macro):", best_val_f1)


if __name__ == "__main__":
    main()