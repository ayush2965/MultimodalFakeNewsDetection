import React, { useState } from "react";

export default function TextVerify() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);

  const API_BASE = "http://localhost:8000";

  const handleVerify = async () => {
    if (!text.trim()) {
      alert("Please enter news text.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("text", text);

      const response = await fetch(`${API_BASE}/predict/text`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        alert(data.error);
      } else {
        setResult(data.result);
        setSources(data.sources || []);
        setShowPopup(true);
      }
    } catch (err) {
      console.error(err);
      alert("Verification failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="text-verify">
      <textarea
        className="news-input"
        placeholder="Paste news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        className="verify-button"
        onClick={handleVerify}
        disabled={loading}
      >
        {loading ? "VERIFYING..." : "VERIFY"}
      </button>

      {showPopup && result && (
        <div className="popup-overlay">
          <div className="popup-box newspaper-result">
            <button
              className="popup-close"
              onClick={() => setShowPopup(false)}
            >
              ✕
            </button>

            <h2 className="result-headline">
              {result.final_outcome.toUpperCase()}
            </h2>

            <p className="result-subtitle">{result.subtitle}</p>

            {/* Stats */}
            <div className="result-stats">
              <span>Confidence: {result.text_confidence.toFixed(2)}%</span>
              <span>Sources Found: {result.sources_found}</span>
              <span>Trusted: {result.trusted_count}</span>
              <span>Debunked: {result.debunk_count}</span>
            </div>

            <p className="result-verdict">
              {result.sve_verdict}
            </p>

            <h4 className="sources-title">Reported By</h4>

            <div className="sources-columns">
              {sources.map((src, i) => (
                <div key={i} className="source-line">
                  <a href={src.link} target="_blank" rel="noreferrer">
                    {src.title}
                  </a>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
