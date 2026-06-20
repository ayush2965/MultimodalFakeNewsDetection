from serper_factcheck import google_search, get_verdict,get_sve_status
def return_top_sources(query):
    results = google_search(query)

    if not results:
        results="No relevant sources found."
        return results
        
    return results
       



        # print(f"Sources found: {len(results)}")
        # print("Verdict:", verdict)
        # print("\nTop Sources Found:")
        # for r in results:
        #     print("-", r.get("title", "No title"))
        #     print("  ", r.get("link", ""))
        # print("\nSVE Outcome:", sve_outcome_val, "| Trusted:", trusted, "| Debunked:", debunk)
        # print("\n" + "="*50 + "\n")