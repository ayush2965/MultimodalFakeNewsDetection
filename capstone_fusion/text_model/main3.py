import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
import time
import multiprocessing

# ============================================================
# 1. Dataset Class
# ============================================================
class IFND_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.texts)

# ============================================================
# 2. BERT + LSTM Model
# ============================================================
class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # BERT encoder
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT

        # LSTM on BERT token embeddings
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_embeddings = bert_out.last_hidden_state  # (B,128,768)

        lstm_out, _ = self.lstm(seq_embeddings)
        final_output = lstm_out[:, -1, :]  # last timestep

        return self.classifier(final_output)

# ============================================================
# 3. Main function
# ============================================================
def main():
    # Load dataset
    df = pd.read_csv("IFND.csv", encoding="MacRoman")
    df["Label"] = df["Label"].map({"TRUE": 0, "Fake": 1})

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = IFND_Dataset(
        df["Statement"].tolist(),
        df["Label"].tolist(),
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,  # multiprocessing
        pin_memory=True
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device.upper()}")

    model = BertLSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3

    # Training loop with tqdm
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        running_loss = 0.0

        progress_bar = tqdm(
            loader,
            total=len(loader),
            unit="batch",
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        for batch_idx, (input_ids, attn_mask, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(
                f"Loss: {loss.item():.4f} | Avg: {running_loss/(batch_idx+1):.4f}"
            )

        epoch_time = time.time() - start_time
        print(f"⏳ Epoch {epoch+1} finished in {epoch_time/60:.2f} minutes")

    # Save model
    torch.save(model.state_dict(), "bert_lstm_ifnd.pth")
    print("\n✅ Model saved as bert_lstm_ifnd.pth")

# ============================================================
# 4. Entrypoint for Windows
# ============================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # required on Windows
    main()