import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------
df = pd.read_csv("coda_text.csv",encoding='ascii')

# Convert labels from string → int

texts = df["clean_title"].tolist()
labels = df["id"].tolist()

# -----------------------------------------
# 2. Load tokenizer & model (BERT-base)
# -----------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# -----------------------------------------
# 3. Function to create embeddings
# -----------------------------------------
def get_cls_embedding(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)

    cls_embedding = output.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0)

# -----------------------------------------
# 4. Generate embeddings
# -----------------------------------------
embeddings = []

for text in tqdm(texts, desc="Embedding"):
    emb = get_cls_embedding(text)
    embeddings.append(emb)

# -----------------------------------------
# 5. Convert to tensor + save
# -----------------------------------------
embeddings_tensor = torch.stack(embeddings)        # shape: (N, 768)
labels_tensor = torch.tensor(labels, dtype=torch.long)

torch.save(
    {"embeddings": embeddings_tensor, "labels": labels_tensor},
    "embedded_dataset.pt"
)

print("Saved embedded_dataset.pt!")