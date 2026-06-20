import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ================= CONFIG =================
MODEL_DIR = Path("unfiltered_model_visual_gnn_model")
MODEL_PATH = MODEL_DIR / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 380
LABEL_MAP = {0: "Fake", 1: "True"}

# ================= IMAGE PATH =================
# <-- CHANGE THIS TO YOUR IMAGE PATH -->
IMAGE_PATH = r"api_dev\final.jpg"

# ================= TRANSFORMS =================
val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= MODEL =================
class VisualPatchGNN(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 16 nodes

        self.node_proj = nn.Linear(1792, hidden_dim)

        self.gnn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        feat = self.cnn(x)
        feat = self.pool(feat)
        feat = feat.flatten(2).transpose(1, 2)
        nodes = self.node_proj(feat)
        for layer in self.gnn:
            agg = nodes.mean(dim=1, keepdim=True)
            nodes = self.dropout(self.relu(layer(nodes + agg)))
        fused = nodes.mean(dim=1)
        return self.classifier(fused)

# ================= PREDICTION FUNCTION =================
def predict_image(model, img_path):
    if not Path(img_path).exists():
        print(f"❌ Image not found: {img_path}")
        return None, None, None

    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = val_tfms(img).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(1).item()
        confidence = probs[0, pred_class].item() * 100  # confidence %

    return LABEL_MAP[pred_class], probs.squeeze().cpu().numpy(), confidence

# ================= MAIN =================
def main():
    model = VisualPatchGNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Model loaded successfully")

    label, probs, confidence = predict_image(model, IMAGE_PATH)
    if label is not None:
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities: Fake={probs[0]:.4f}, True={probs[1]:.4f}")

if __name__ == "__main__":
    main()
