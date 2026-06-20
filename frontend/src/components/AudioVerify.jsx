import React, { useState } from "react";
import "./AudioVerify.css";
import "./MultimodalVerify.css"; // reuse styles for consistency

export default function AudioVerify() {
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);

  const handleSubmit = async () => {
    if (!audioFile) {
      alert("Please upload an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
      setLoading(true);
      setResult(null);
      setSources([]);

      const response = await fetch("http://localhost:8000/predict/audio", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.result) setResult(data.result);
      if (data.sources) setSources(data.sources);
      setShowPopup(true);
    } catch (err) {
      console.error(err);
      alert("Audio verification failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="audio-verify">
      

      <div className="audio-input-wrapper">
  <input
    type="file"
    id="audioFile"
    accept="audio/*"
    onChange={(e) => setAudioFile(e.target.files[0])}
    hidden
  />
  <label htmlFor="audioFile" className="audio-input">
    {audioFile ? audioFile.name : "CHOOSE AUDIO"}
  </label>
</div>


      <button
        className="verify-button"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? "VERIFYING..." : "VERIFY"}
      </button>

      {/* Popup Result */}
      {showPopup && result && (
        <div className="popup-overlay">
          <div className="popup-box newspaper-result">
            <button
              className="popup-close"
              onClick={() => setShowPopup(false)}
            >
              ✕
            </button>

            {/* Main Outcome */}
            <h2 className="result-headline">
              {result.final_outcome?.toUpperCase() || "VERIFICATION RESULT"}
            </h2>
            <p className="result-subtitle">{result.subtitle || ""}</p>

            {/* Detailed Stats */}
            <div className="result-stats">
              {result.text_confidence !== undefined && (
                <span>Text Confidence: {result.text_confidence.toFixed(2)}%</span>
              )}
              {result.trusted_count !== undefined && (
                <span>Trusted Sources Found: {result.trusted_count}</span>
              )}
              {result.debunk_count !== undefined && (
                <span>Debunked Sources Found: {result.debunk_count}</span>
              )}
              {result.sources_found !== undefined && (
                <span>Total Sources Found: {result.sources_found}</span>
              )}
            </div>

            {/* SVE Verdict */}
            <p className="result-verdict">{result.sve_verdict || ""}</p>

            {/* Audio Preview */}
            {audioFile && (
              <div className="audio-player">
                <h4>Uploaded Audio</h4>
                <audio controls src={URL.createObjectURL(audioFile)} />
              </div>
            )}

            {/* Sources */}
            {sources.length > 0 && (
              <>
                <h4 className="sources-title">Sources</h4>
                <div className="sources-columns">
                  {sources.map((src, i) => (
                    <div key={i} className="source-line">
                      <a href={src.link} target="_blank" rel="noreferrer">
                        {src.title}
                      </a>
                      {src.date && <span className="source-date">{src.date}</span>}
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
