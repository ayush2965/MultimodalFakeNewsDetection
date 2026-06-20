# test_predict.py
from im_model_prediction import load_model, predict_image, DEVICE, LABEL_MAP
from pathlib import Path
from ocr_verification import extract_text
from serper_factcheck import google_search, get_verdict, get_sve_status  # import SVE functions

# Load the model once
model = load_model()

# Predict an image
IMAGE_PATH = r"C:\Users\Admin\Desktop\myapps\capstone_fusion\capstone_fusion\api_dev\test3.jpg"

sve_outcome = -1
trusted_count = 0
debunk_count = 0

def predict_image_and_verify(img_path):
    label, probs, confidence = predict_image(model, img_path)
    finalresults={}
    claim = extract_text(img_path)
    if not claim:
        if label == "True":
            print("Verdict: Most Likely Real")
            print("Image is authentic, but no claims found to verify.")
            finalresults["final_outcome"] = "Most Likely Real"
            finalresults["subtitle"] = "Image is authentic, but no claims found to verify."
            finalresults["image_confidence"] = confidence
        elif label == "Fake":
            print("Verdict: Most Likely Misleading")
            print("Image has no claims but may be visually tampered.")
            finalresults["final_outcome"] = "Most Likely Misleading"
            finalresults["subtitle"] = "Image has no claims but may be visually tampered."
            finalresults["image_confidence"] = confidence
    else:
        results = google_search(claim)
        if results:
            verdict_text = get_verdict(results)
            sve_outcome, trusted_count, debunk_count = get_sve_status()
        else:
            print("No results found.\n")
            sve_outcome, trusted_count, debunk_count = 4, 0, 0  # fallback
        if sve_outcome==0 and label=="Fake":
            print(f"Verdict: Most Likely Fake")
            print(f"Image claims may be partially debunked as well as misleading.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Most Likely Fake"
            finalresults["subtitle"] = "Image claims may be partially debunked as well as misleading."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==0 and label=="True":
            print(f"Verdict: Possibly Fake")
            print(f"Claims may have been debunked even if image is authentic.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Possibly Fake"
            finalresults["subtitle"] = "Claims may have been debunked even if image is authentic."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==1 and label=="True":
            print(f"Verdict: Most Likely Real")
            print(f"Claims found on trusted sources, visuals seem authentic.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Most Likely Real"
            finalresults["subtitle"] = "Claims found on trusted sources, visuals seem authentic."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==1 and label=="Fake":
            print(f"Verdict: True but Possibly Misleading")
            print(f"Claim found to be true, but content may be partially misleading")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "True but Possibly Misleading"
            finalresults["subtitle"] = "Claim found to be true, but content may be partially misleading."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==2 and label=="True":
            print(f"Verdict: Possibly Real")
            print(f"Single trusted source found, but content may be misleading")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Possibly Real"
            finalresults["subtitle"] = "Single trusted source found, but content may be misleading."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==2 and label=="Fake":
            print(f"Verdict: Uncertain, Possibly Misleading")
            print(f"Single trusted source found, but image may be visually tampered.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Uncertain, Possibly Misleading"
            finalresults["subtitle"] = "Single trusted source found, but image may be visually tampered."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==3 and label=="Fake":
            print(f"Verdict: Possibly Fake")
            print(f"No trusted sources found, image may be visually tampered.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Possibly Fake"
            finalresults["subtitle"] = "No trusted sources found, image may be visually tampered."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==3 and label=="True":
            print(f"Verdict: Probably Fake")
            print(f"No trusted sources found, but image seems authentic.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Probably Fake"
            finalresults["subtitle"] = "No trusted sources found, but image seems authentic."
            finalresults["image_confidence"] = confidence
        elif sve_outcome==4:
            print(f"Verdict: Uncertain, Authentic Image")
            print(f"No sources found to verify claims, but image seems authentic.")
            print(f"Confidence of clickbait nature: {confidence:.2f}%\n")
            finalresults["final_outcome"] = "Uncertain, Possibly Fake"
            finalresults["subtitle"] = "No sources found to verify claims, but image seems authentic."
            finalresults["image_confidence"] = confidence
        print(f"Sources Found: {len(results)}")
        print(f"Possible Debunked Claims found: {debunk_count}")
        print(f"Possible Trusted Sources Found: {trusted_count}\n")
        finalresults["sources_found"] = len(results)
        finalresults["debunk_count"] = debunk_count
        finalresults["trusted_count"] = trusted_count
       
    print(finalresults)
    return finalresults 
