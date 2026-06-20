import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ================= CONFIG =================
MODEL_PATH=r"D:\capstone_fusion\capstone_fusion\unfiltered_model_visual_gnn_model\best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 380
LABEL_MAP = {0: "Fake", 1: "True"}

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
        self.gnn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim),
                                  nn.Linear(hidden_dim, hidden_dim)])
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

# ================= HELPER FUNCTIONS =================
def load_model(model_path=MODEL_PATH, device=DEVICE):
    """Load the VisualPatchGNN model with pretrained weights."""
    model = VisualPatchGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, img_path):
    """Predict the label, probabilities, and confidence for an image."""
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img = val_tfms(img).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(1).item()
        confidence = probs[0, pred_class].item() * 100  # confidence %

    return LABEL_MAP[pred_class], probs.squeeze().cpu().numpy(), confidence
    print(f"Predicted: {LABEL_MAP[pred_class]} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    # Example usage
    model = load_model()
    img_path = r"D:\\rice.jpg"
    label, probabilities, confidence = predict_image(model, img_path)
    print(f"Predicted Label: {label}")
    print(f"Probabilities: {probabilities}")
    print(f"Confidence: {confidence:.2f}%")

