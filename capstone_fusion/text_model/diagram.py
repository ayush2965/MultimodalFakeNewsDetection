# -------------------------
# Heuristic Safe Samples for Late Fusion
# -------------------------

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
DATA_CSV = "IFND.csv"
MODEL_DIR = "bert_lstm_end2end"
SAFE_CSV = "IFND_safe_heuristic.csv"
PRETRAINED = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"TRUE": 0, "Fake": 1}
CONF_THRESHOLD = 0.99  # confidence threshold

# -------------------------
# Dataset
# -------------------------
class IFNDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
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
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "index": idx
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    indices = [b["index"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "indices": indices}

# -------------------------
# Model
# -------------------------
class BertLSTMForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name=PRETRAINED, lstm_hidden=256, lstm_layers=1, bidirectional=True, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(token_embeddings)
        lengths = attention_mask.sum(dim=1) - 1
        last_timestep = lstm_out[torch.arange(lstm_out.size(0)), lengths, :]
        logits = self.classifier(last_timestep)
        return logits

# -------------------------
# Main
# -------------------------
def main():
    # Load dataset
    df = pd.read_csv(DATA_CSV, encoding="MacRoman")
    df["Label"] = df["Label"].map(LABEL_MAP)

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    dataset = IFNDDataset(df, tokenizer, MAX_LEN)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load pretrained model
    model = BertLSTMForSequenceClassification().to(DEVICE)
    state_dict_path = os.path.join(MODEL_DIR, "best_model_state_dict.pt")
    model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
    model.eval()

    # -------------------------
    # Heuristic selection of "safe" samples
    # -------------------------
    safe_indices = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating dataset"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            indices = batch["indices"]

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            for i, idx in enumerate(indices):
                if preds[i] != labels[i] or conf[i] < CONF_THRESHOLD:
                    safe_indices.append(idx)

    # -------------------------
    # Save safe samples to CSV
    # -------------------------
    safe_df = df.iloc[safe_indices].reset_index(drop=True)
    os.makedirs(os.path.dirname(SAFE_CSV) or ".", exist_ok=True)
    safe_df.to_csv(SAFE_CSV, index=False)
    print(f"Saved {len(safe_df)} heuristic safe samples to {SAFE_CSV}")

if __name__ == "__main__":
    main()
