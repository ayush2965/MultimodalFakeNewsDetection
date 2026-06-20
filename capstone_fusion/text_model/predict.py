# predict.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# -------------------------
# Config
# -------------------------
MODEL_DIR = "bert_lstm_end2end"
PRETRAINED = "bert-base-uncased"
MAX_LEN = 256
LABEL_MAP = {0: "TRUE", 1: "Fake"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Model definition (EXACT match)
# -------------------------
class BertLSTMForSequenceClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED)
        hidden = self.bert.config.hidden_size  # 768

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_out.last_hidden_state)
        lengths = attention_mask.sum(dim=1) - 1
        last_hidden = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        return self.classifier(last_hidden)


# -------------------------
# Main prediction loop
# -------------------------
def main():
    print("Loading tokenizer and model...")

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)

    model = BertLSTMForSequenceClassification()
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/best_model_state_dict.pt", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Type text to classify (type 'exit' to quit)\n")

    while True:
        text = input("Input text: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("⚠️ Empty input, try again.\n")
            continue

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = LABEL_MAP[pred]
        confidence = probs[0][pred].item() * 100

        print(f"Prediction : {label}")
        print(f"Confidence : {confidence:.2f}%\n")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    main()