import React, { useState } from "react";
import "./ImageVerify.css";

export default function ImageVerify() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);

  const API_BASE = "http://localhost:8000";

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleVerify = async () => {
    if (!image) {
      alert("Upload an image first.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch(`${API_BASE}/predict/image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        alert(data.error);
      } else {
        setResult(data.result);
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
    <div className="image-verify">
      <label className="upload-button">
        CHOOSE IMAGE
        <input type="file" accept="image/*" hidden onChange={handleImageChange} />
      </label>

      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{
            width: "200px",
            margin: "10px auto",
            display: "block",
            border: "2px solid black",
          }}
        />
      )}

      <button
        className="verify-button"
        onClick={handleVerify}
        disabled={loading}
      >
        {loading ? "VERIFYING..." : "VERIFY IMAGE"}
      </button>

      {/* Newspaper Popup */}
      {/* Popup result */}
{showPopup && result && (
  <div className="popup-overlay">
    <div className="popup-box newspaper-result">
      <button
        className="popup-close"
        onClick={() => setShowPopup(false)}
      >
        ✕
      </button>

      {/* Headline */}
      <h2 className="result-headline">
        {(result.final_outcome || "RESULT").toUpperCase()}
      </h2>

      {/* Subtitle */}
      {result.subtitle && (
        <p className="result-subtitle">{result.subtitle}</p>
      )}

      {/* Stats / metrics */}
      <div className="result-stats">
        {result.text_confidence !== undefined && (
          <span>Text Confidence: {Number(result.text_confidence).toFixed(2)}%</span>
        )}
        {result.image_confidence !== undefined && (
          <span>Image Confidence: {Number(result.image_confidence).toFixed(2)}%</span>
        )}
        {result.sources_found !== undefined && (
          <span>Sources Found: {result.sources_found}</span>
        )}
        {result.trusted_count !== undefined && (
          <span>Trusted: {result.trusted_count}</span>
        )}
        {result.debunk_count !== undefined && (
          <span>Debunked: {result.debunk_count}</span>
        )}
      </div>

      {/* Optional Verdict */}
      {result.sve_verdict && (
        <p className="result-verdict">{result.sve_verdict}</p>
      )}
    </div>
  </div>
)}


    </div>
  );
}
