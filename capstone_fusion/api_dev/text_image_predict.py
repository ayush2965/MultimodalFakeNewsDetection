from text_clone import text_predict_clone
from im_model_prediction import predict_image
from text_image_matcher import verify_claim
from ocr_verification import extract_text
from im_model_prediction import load_model



def predict_text_image(image_path, claim):
    finalresults={}
    # ---- Image model prediction ----
    model = load_model()
    img_label, img_probs, img_confidence = predict_image(model, image_path)

    # ---- Text model prediction ----
    clickbait_confidence,text_outcome, trusted_count, debunk_count, verdictobtained,results= text_predict_clone(claim)

    # ---- Image-text relation (CLIP) ----
    clip_score, clip_outcome, clip_class = verify_claim(image_path, claim)

    if img_label == "True" and text_outcome == 2 and clip_outcome == 0:
        print("Verdict: Most Likely Real")
        print("Image is authentic, claim is verified by multiple sources, and image closely matches claim.")
        finalresults["final_outcome"] = "Most Likely Real"
        finalresults["subtitle"] = "Image is authentic, claim is verified by multiple sources, and image closely matches claim."
    elif img_label=="True" and text_outcome==0 and clip_outcome==0:
        print("Verdict: Most Likely Fake")
        print("Claim may be using authentic yet irrelevant image to mislead individuals.")
        finalresults["final_outcome"] = "Most Likely Fake"
        finalresults["subtitle"] = "Claim may be using authentic yet irrelevant image to mislead individuals."
    elif img_label=="True" and text_outcome==2 and clip_outcome==1:
        print("Verdict: Possibly Real but Unrelated")
        print("Image is authentic and claim is verified, but image may be misdirecting.")
        finalresults["final_outcome"] = "Possibly Real but Unrelated"
        finalresults["subtitle"] = "Image is authentic and claim is verified, but image may be misdirecting."
    elif img_label=="True" and text_outcome==0 and clip_outcome==1:
        print("Verdict: Uncertain but Possibly Misleading")
        print("Image is authentic but claim may be misleading")
        finalresults["final_outcome"] = "Uncertain but Possibly Misleading"
        finalresults["subtitle"] = "Image is authentic but claim may be misleading."
    elif img_label=="True" and text_outcome==2 and clip_outcome==2: 
        print("Verdict: Possibly Real")
        print("Some sources verify claim and image may be authentic, yet may be out of context.")
        finalresults["final_outcome"] = "Possibly Real"
        finalresults["subtitle"] = "Some sources verify claim and image may be authentic, yet may be out of context."
    elif img_label=="True" and text_outcome==0 and clip_outcome==2:
        print("Verdict: Possibly Fake and Misleading")
        print("Claims may be using realistic misleading images to spread misinformation.")
        finalresults["final_outcome"] = "Possibly Fake and Misleading"
        finalresults["subtitle"] = "Claims may be using realistic misleading images to spread misinformation."
    elif img_label=="Fake" and text_outcome==2 and clip_outcome==0:
        print("Verdict: Uncertain")
        print("May be real as multiple sources verify claims, but image may be tampered.")
        finalresults["final_outcome"] = "Uncertain"
        finalresults["subtitle"] = "May be real as multiple sources verify claims, but image may be tampered."
    elif img_label=="Fake" and text_outcome==0 and clip_outcome==0:
        print("Verdict: Possibly Fake")
        print("May be fake as no sources verify claims")
        finalresults["final_outcome"] = "Possibly Fake"
        finalresults["subtitle"] = "May be fake as no sources verify claims"
    elif img_label=="Fake" and text_outcome==2 and clip_outcome==1:
        print("Verdict: Possibly Real")
        print("May be real as multiple sources verify claims, image may be misleading")
        finalresults["final_outcome"] = "Possibly Real"
        finalresults["subtitle"] = "May be real as multiple sources verify claims, image may be misleading"
    elif img_label=="Fake" and text_outcome==0 and clip_outcome==1:
        print("Verdict: Most Likely Fake")
        print("No sources verify claims, image may be tampered")
        finalresults["final_outcome"] = "Most Likely Fake"
        finalresults["subtitle"] = "No sources verify claims, image may be tampered"
    elif img_label=="Fake" and text_outcome==2 and clip_outcome==2:
        print("Verdict: Possibly Real")
        print("May be real as multiple sources verify claims, but image may be fake")
        finalresults["final_outcome"] = "Possibly Real"
        finalresults["subtitle"] = "May be real as multiple sources verify claims, but image may be fake"
    elif img_label=="Fake" and text_outcome==0 and clip_outcome==2:
        print("Verdict: Most Likely Fake")
        print("No sources verify claims, tampered text-image relation")
        finalresults["final_outcome"] = "Most Likely Fake"
        finalresults["subtitle"] = "No sources verify claims, tampered text-image relation"
    print(f"Image Integrity Maintained: {img_label} (Confidence: {img_confidence:.2f}%)")
    print("Text Clickbait Confidence:", clickbait_confidence)
    print("Textual Claims:", verdictobtained)
    print("Text-Image Match:",clip_class)
    print(f"Sources Found: {trusted_count + debunk_count} (Trusted: {trusted_count}, Debunked: {debunk_count})")
    finalresults["image_integrity"] = img_label
    finalresults["image_confidence"] = img_confidence
    finalresults["text_clickbait_confidence"] = clickbait_confidence
    finalresults["textual_claims"] = verdictobtained
    finalresults["text_image_match"] = clip_class
    finalresults["sources_found"] = trusted_count + debunk_count
    finalresults["trusted_count"] = trusted_count
    finalresults["debunk_count"] = debunk_count
    print(finalresults)
    return finalresults,results

if __name__ == "__main__":
    img_path = r"api_dev\india_u19.jpg"   # image file
    claim = "India lift the 2026 U19 World Cup."  #claim
    predict_text_image(img_path, claim)


