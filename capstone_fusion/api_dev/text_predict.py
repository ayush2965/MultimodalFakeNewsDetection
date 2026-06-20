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


def text_predict(text):
    finalresults={}
    print("Loading tokenizer and model...")

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    model = BertLSTMForSequenceClassification()
    model.load_state_dict(torch.load(r"D:\capstone_fusion\capstone_fusion\text_model\bert_lstm_end2end\best_model_state_dict.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Model loaded. Type text to classify (type 'exit' to quit)\n")
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
        print("No results found from Serper.\n")
        sve_outcome, trusted_count, debunk_count = 3, 0, 0  # fallback


    print("\n--- Model Prediction ---")
    if label == "Fake" and sve_outcome == 0:
        print(f"Verdict: Probably Fake")
        print(f"May be designed to spread through unverified sources, multiple trusted sources may have debunk this claim")
        print(f"Clickbait confidence : {confidence:.2f}%\n")
        verdictlocal="Probably Fake"
        text_outcome=0
        finalresults["final_outcome"] = "Probably Fake"
        finalresults["subtitle"] = "May be designed to spread through unverified sources, multiple trusted sources may have debunk this claim."
        finalresults["text_confidence"] = confidence
    elif label=="Fake" and sve_outcome==3:
        print(f"Verdict: Most Likely Fake")
        print(f"Found across no trusted sources, may be designed to spread through unverified sources")
        print(f"Clickbait confidence : {confidence:.2f}%\n")
        text_outcome=0
        verdictlocal="Most Likely Fake"
        finalresults["final_outcome"] = "Most Likely Fake"
        finalresults["subtitle"] = "Found across no trusted sources, may be designed to spread through unverified sources."
        finalresults["text_confidence"] = confidence
    elif label=="Fake" and sve_outcome==2:
        print(f"Verdict: Uncertain but Probably Fake")
        print(f"Found on one trusted source, may be fake due to clickbait nature or lack of coverage")
        print(f"Clickbait confidence : {confidence:.2f}%\n")
        text_outcome=0
        verdictlocal="Uncertain but Probably Fake"
        finalresults["final_outcome"] = "Uncertain but Probably Fake"
        finalresults["subtitle"] = "Found on one trusted source, may be fake due to clickbait nature or lack of coverage."
        finalresults["text_confidence"] = confidence    
    elif label=="Fake" and sve_outcome==1:
        print(f"Verdict: Partially Verified")
        print(f"Relevant to reality but may be misleading")
        print(f"Clickbait confidence : {confidence:.2f}%\n")
        text_outcome=0
        verdictlocal="Partially Verified"
        finalresults["final_outcome"] = "Partially Verified"
        finalresults["subtitle"] = "Relevant to reality but may be misleading."
        finalresults["text_confidence"] = confidence
    elif label=="TRUE" and sve_outcome==1:
        print(f"Verdict: Most Likely True")
        print(f"Verified by multiple trusted sources, no clickbait indicators")
        print(f"Clickbait Confidence : -{confidence:.2f}%\n")
        text_outcome=2
        verdictlocal="Most Likely True"
        finalresults["final_outcome"] = "Most Likely True"
        finalresults["subtitle"] = "Verified by multiple trusted sources, no clickbait indicators."
        finalresults["text_confidence"] = confidence
    elif label=="TRUE" and sve_outcome==2:
        print(f"Verdict: Partially Verified")
        print(f"Found on one trusted source, may be true but lacks coverage")
        print(f"Clickbait Confidence : -{confidence:.2f}%\n")
        text_outcome=2
        verdictlocal="Partially Verified"
        finalresults["final_outcome"] = "Partially Verified"
        finalresults["subtitle"] = "Found on one trusted source, may be true but lacks coverage."
        finalresults["text_confidence"] = confidence
    elif label=="TRUE" and sve_outcome==0:
        print(f"Verdict: Possibly Fake")
        print(f"Some sources have debunked this claim, but no clickbait indicators")
        print(f"Clickbait Confidence : -{confidence:.2f}%\n")
        text_outcome=0
        verdictlocal="Possibly Fake"
        finalresults["final_outcome"] = "Possibly Fake"
        finalresults["subtitle"] = "Some sources have debunked this claim, but no clickbait indicators."
        finalresults["text_confidence"] = confidence
    elif label=="TRUE" and sve_outcome==3:
        print(f"Verdict: Possibly Fake")
        print(f"Not found on trusted sources, but has no clickbait nature. ")
        print(f"Clickbait Confidence : -{confidence:.2f}%\n")
        text_outcome=0
        verdictlocal="Possibly Fake"
        finalresults["final_outcome"] = "Possibly Fake"
        finalresults["subtitle"] = "Not found on trusted sources, but has no clickbait nature."
        finalresults["text_confidence"] = confidence
    print()
    print("==================================================")
    print()
    print(f"Sources Found: {len(results)}")
    print(f"Possible Debunked Claims found: {debunk_count}")
    print(f"Possible Trusted Sources Found: {trusted_count}\n")
    print(f"SVE thinks : {verdict}")
    finalresults["sources_found"] = len(results)
    finalresults["debunk_count"] = debunk_count
    finalresults["trusted_count"] = trusted_count
    finalresults["sve_verdict"] = verdict
    

        # Top Sources
    if results:
        print("Top Sources:")
        for r in results:
            print("-", r.get("title", "No title"))
            print("  ", r.get("link", ""))
    print(finalresults)
    return confidence,text_outcome, trusted_count, debunk_count, verdictlocal, finalresults,results



if __name__ == "__main__":
    while True:
        text = input("Input text: ").strip()
        if text.lower() == "exit":
            break
        result = text_predict(text)
        print(result)
