import requests
import os
API_KEY = os.getenv("google_api_key")

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# ---------------------------
# Convert rating to score
# ---------------------------
def rating_to_score(rating):
    rating = rating.lower()

    if "true" in rating and "mostly" not in rating:
        return 1
    if "mostly true" in rating:
        return 0.5
    if "half true" in rating:
        return 0
    if "misleading" in rating:
        return -0.5
    if "mostly false" in rating:
        return -0.75
    if "false" in rating:
        return -1

    return 0


# ---------------------------
# Final decision logic
# ---------------------------
def decision(avg_score):
    if avg_score >= 0.3:
        return "TRUE"
    elif avg_score <= -0.3:
        return "FAKE"
    else:
        return "UNCERTAIN"


# ---------------------------
# Fact-check pipeline
# ---------------------------
def predict_claim(query):
    params = {
        "query": query,
        "key": API_KEY,
        "languageCode": "en",
        "pageSize": 5
    }

    response = requests.get(ENDPOINT, params=params)
    data = response.json()

    claims = data.get("claims", [])
    if not claims:
        return "NO FACT CHECK FOUND", 0

    scores = []

    for claim in claims:
        for review in claim.get("claimReview", []):
            rating = review.get("textualRating", "")
            score = rating_to_score(rating)
            scores.append(score)

    if not scores:
        return "NO RATINGS FOUND", 0

    avg_score = sum(scores) / len(scores)
    result = decision(avg_score)

    confidence = round(abs(avg_score) * 100, 2)

    return result, confidence


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    query = input("Enter news claim: ")

    result, confidence = predict_claim(query)

    print("\nPrediction:", result)
    print("Confidence:", confidence, "%")
