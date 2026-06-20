import React, { useState } from "react";

export default function MultimodalVerify() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);

  const handleSubmit = async () => {
    if (!text || !image) {
      alert("Provide both text and image.");
      return;
    }

    const formData = new FormData();
    formData.append("text", text);
    formData.append("image", image);

    try {
      setLoading(true);
      setResult(null);

      const response = await fetch("http://localhost:8000/predict/multimodal", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data.result);
      setShowPopup(true);
    } catch (err) {
      console.error(err);
      alert("Verification failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="multimodal-verify">

      <textarea
        placeholder="Enter news text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        className="news-input"
      />

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImage(e.target.files[0])}
        className="image-input"
      />

      <button
        className="verify-button"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? "VERIFYING..." : "VERIFY"}
      </button>

      {/* Popup Result */}
      {showPopup && result && Array.isArray(result) && (
  <div className="popup-overlay">
    <div className="popup-box newspaper-result">
      <button
        className="popup-close"
        onClick={() => setShowPopup(false)}
      >
        ✕
      </button>

      {/* Main Result */}
      <h2 className="result-headline">
        {result[0].final_outcome?.toUpperCase() || "VERIFICATION RESULT"}
      </h2>

      {result[0].subtitle && (
        <p className="result-subtitle">{result[0].subtitle}</p>
      )}

      {/* Stats / Metrics */}
      <div className="result-stats">
        {Object.entries(result[0]).map(([key, value]) => {
          // skip non-stat fields
          if (
            [
              "final_outcome",
              "subtitle",
              "sve_verdict",
            ].includes(key)
          )
            return null;

          if (typeof value === "number") value = value.toFixed(2);
          return (
            <span key={key}>
              {key.replace(/_/g, " ")}: {value}
            </span>
          );
        })}
      </div>

      {/* Optional Verdict */}
      {result[0].sve_verdict && (
        <p className="result-verdict">{result[0].sve_verdict}</p>
      )}

      

      {/* Sources Section */}
      {Array.isArray(result[1]) && result[1].length > 0 && (
        <>
          <h4 className="sources-title">Reported By</h4>
          <div className="sources-columns">
            {result[1].map((src, i) => (
              <div key={i} className="source-line">
                <a href={src.link} target="_blank" rel="noreferrer">
                  {src.title}
                </a>
                {src.date && <span className="source-date">({src.date})</span>}
                {src.snippet && <p className="source-snippet">{src.snippet}</p>}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  </div>
)}

    </div>
  );
}
