# test_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import pandas as pd

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
# 2. Model Class
# ============================================================
class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.eval()  # freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_embeddings = bert_out.last_hidden_state  # (B, seq_len, 768)

        lstm_out, _ = self.lstm(seq_embeddings)
        final_output = lstm_out[:, -1, :]  # last timestep
        return self.classifier(final_output)

# ============================================================
# 3. Load model and tokenizer
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

model = BertLSTMClassifier()
model.load_state_dict(torch.load("bert_lstm_ifnd.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ============================================================
# 4. Load dataset and validation split
# ============================================================
df = pd.read_csv("IFND.csv", encoding="MacRoman")
df["Label"] = df["Label"].map({"TRUE": 0, "Fake": 1})

dataset = IFND_Dataset(df["Statement"].tolist(), df["Label"].tolist(), tokenizer)
val_size = int(0.2 * len(dataset))
_, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ============================================================
# 5. Evaluate accuracy
# ============================================================
correct = 0
total = 0
with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")

# ============================================================
# 6. Interactive prediction
# ============================================================
def predict_fake_news(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    label_map = {1: "TRUE", 0: "Fake"}
    return label_map[pred_class], probs[0][pred_class].item()


print("\n🚀 Enter news text to test (type 'exit' to quit).")
while True:
    news_text = input("\nEnter news: ")
    if news_text.lower() == "exit":
        print("Exiting...")
        break

    label, confidence = predict_fake_news(news_text)
    print(f"Prediction: {label} | Confidence: {confidence:.2f}")