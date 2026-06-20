import os

import requests
from urllib.parse import urlparse

# ===== INSERT YOUR SERPER API KEY HERE =====
SERPER_API_KEY = os.getenv("serperi_api_key")
URL = "https://google.serper.dev/search"

# Global vars
sve_outcome = -1
debunk_count = 0
trusted_count = 0


def google_search(query):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {"q": query, "num": 5}

    r = requests.post(URL, headers=headers, json=payload)

    if r.status_code != 200:
        print("Search API error:", r.status_code)
        return []

    data = r.json()
    return data.get("organic", [])


def get_verdict(results):
    global sve_outcome, trusted_count, debunk_count  # <-- important!

    # Reset counts each call
    trusted_count = 0
    debunk_count = 0

    trusted = [
        "bbc.com", "reuters.com", "apnews.com", "theguardian.com",
        "cnn.com", "nytimes.com", "snopes.com", "factcheck.org",
        "factly.in", "ndtv.com", "politifact.com", "indiatoday.in",
        "thehindu.com", "indianexpress.com", "timesofindia.com",
        "hindustantimes.com", "bloomberg.com", "forbes.com",
        "economictimes.com", "livemint.com", "business-standard.com",
        "espn.com", "skysports.com", "espncricinfo.com", "cricbuzz.com",
        "wikipedia.org", "nasa.gov", "icc-cricket.com", "who.int", "un.org", "cdc.gov", "nih.gov",
        "timesnow.com", "euronews.com"
    ]

    debunk_words = [
        "false", "fake", "myth", "debunked", "conspiracy", "not true",
        "incorrect", "misleading", "hoax", "untrue", "disproved", "refuted",
        "busted", "exposed", "fabricated", "falsely", "erroneous",
        "inaccurate", "unfounded", "baseless", "unsubstantiated", "spurious",
        "deceptive", "bogus", "phony", "counterfeit", "fraudulent",
        "misinformation", "disinformation", "propaganda", "pseudoscience",
        "quackery", "scam", "sham", "fake news", "false claim", "debunk",
        "debunks", "debunking"
    ]

    for r in results:
        link = r.get("link", "")
        snippet = r.get("snippet", "").lower()
        title = r.get("title", "").lower()

        domain = urlparse(link).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # count trusted domains
        if any(t in domain for t in trusted):
            trusted_count += 1

            # check if title or snippet debunks claim
            combined_text = snippet + " " + title
            if any(word in combined_text for word in debunk_words):
                debunk_count += 1

    # adjust verdict based on counts
    if debunk_count >= 1:
        sve_outcome = 0
        return "Likely Fake (trusted sources debunk claim)"
    elif trusted_count >= 2:
        sve_outcome = 1
        return "Likely True (found on multiple trusted sites)"
    elif trusted_count == 1:
        sve_outcome = 2
        return "Uncertain (found on one trusted source)"
    else:
        sve_outcome = 3
        return "Possibly Fake"


def get_sve_status():
    """Return the current global SVE values"""
    global sve_outcome, trusted_count, debunk_count
    return sve_outcome, trusted_count, debunk_count


def main():
    print("\n=== Fake News Detector ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter headline or claim: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter something.\n")
            continue

        print("Searching...\n")
        results = google_search(query)

        if not results:
            print("No results found.\n")
            continue

        verdict = get_verdict(results)
        sve_outcome_val, trusted, debunk = get_sve_status()

        print(f"Sources found: {len(results)}")
        print("Verdict:", verdict)
        print("\nTop Sources Found:")
        for r in results:
            print("-", r.get("title", "No title"))
            print("  ", r.get("link", ""))
        print("\nSVE Outcome:", sve_outcome_val, "| Trusted:", trusted, "| Debunked:", debunk)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
