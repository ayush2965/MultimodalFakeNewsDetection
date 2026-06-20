import torch
import clip
from PIL import Image

# ---- Load model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def verify_claim(image_path, claim):
    # Load image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize claim
    text = clip.tokenize([claim]).to(device)

    # Compute embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity score
        similarity = (image_features @ text_features.T).item()

    final_score=round(similarity, 3)
    if final_score > 0.30:
        print("Image likely matches claim")
        outcome=0
        clipclass="Image likely matches claim"
    elif final_score > 0.20:
        print(" Uncertain match")
        outcome=1
        clipclass="Uncertain match"
    else:
        print("Image likely unrelated")
        outcome=2
        clipclass="Image likely unrelated"
    return final_score, outcome, clipclass


# ---- Example usage ----
if __name__ == "__main__":
    img_path = r"api_dev\test10.jpg"   # image file
    claim = "Ajit Dada Pawar passed away"  #claim
    verify_claim(img_path, claim)

    

    
