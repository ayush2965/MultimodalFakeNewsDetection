# predict.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from serper_factcheck import google_search, get_verdict, get_sve_status  # <- import functions only

MODEL_DIR = r"D:\capstone_fusion\capstone_fusion\text_model\bert_lstm_end2end"
PRETRAINED = "bert-base-uncased"
MAX_LEN = 256
LABEL_MAP = {0: "TRUE", 1: "Fake"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertLSTMForSequenceClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_DIR)
        hidden = self.bert.config.hidden_size

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
        lengths = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        last_hidden = lstm_out[torch.arange(lstm_out.size(0)), lengths, :]
        return self.classifier(last_hidden)


def text_predict_clone(text):
   

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    model = BertLSTMForSequenceClassification()
    model.load_state_dict(torch.load(r"D:\capstone_fusion\capstone_fusion\text_model\bert_lstm_end2end\best_model_state_dict.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

   
 # ---- BERT+LSTM Prediction ----
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = LABEL_MAP[pred]
    confidence = probs[0][pred].item() * 100

    # ---- SERPER FACT CHECK ----
    results = google_search(text)
    if results:
        verdict = get_verdict(results)
        sve_outcome, trusted_count, debunk_count = get_sve_status()
    else:
        
        sve_outcome, trusted_count, debunk_count = 3, 0, 0  # fallback

    if label == "Fake" and sve_outcome == 0:
        verdictlocal="Probably Fake"
        text_outcome=0
    elif label=="Fake" and sve_outcome==3:
        text_outcome=0
        verdictlocal="Most Likely Fake"
    elif label=="Fake" and sve_outcome==2:
        text_outcome=0
        verdictlocal="Uncertain but Probably Fake"
    elif label=="Fake" and sve_outcome==1:
        text_outcome=0
        verdictlocal="Partially Verified"
    elif label=="TRUE" and sve_outcome==1:
        text_outcome=2
        verdictlocal="Most Likely True"
    elif label=="TRUE" and sve_outcome==2:
        text_outcome=2
        verdictlocal="Partially Verified"
    elif label=="TRUE" and sve_outcome==0:
        text_outcome=0
        verdictlocal="Possibly Fake"
    elif label=="TRUE" and sve_outcome==3:
        text_outcome=0
        verdictlocal="Possibly Fake"
    return confidence,text_outcome, trusted_count, debunk_count, verdictlocal,results




