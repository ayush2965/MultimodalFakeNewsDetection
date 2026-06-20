import React, { useState } from "react";


import TextVerify from "./TextVerify";
import ImageVerify from "./ImageVerify";
import MultimodalVerify from "./MultimodalVerify";
import AudioVerify from "./AudioVerify";

export default function Verify() {
  const [activeSection, setActiveSection] = useState("text");

  const renderSection = () => {
    switch (activeSection) {
      case "text":
        return <TextVerify />;
      case "image":
        return <ImageVerify />;
      case "multimodal":
        return <MultimodalVerify />;
      case "audio":
        return <AudioVerify />;
      default:
        return <TextVerify />;
    }
  };

  return (
    <div className="verify-container">
      <h3>VERIFY YOUR NEWS</h3>

      {/* Section Buttons */}
      <div className="verify-tabs">
        <button onClick={() => setActiveSection("text")}>TEXT</button>
        <button onClick={() => setActiveSection("image")}>IMAGE</button>
        <button onClick={() => setActiveSection("multimodal")}>
          MULTIMODAL
        </button>
        <button onClick={() => setActiveSection("audio")}>AUDIO</button>
      </div>

      {/* Render Selected Section */}
      {renderSection()}
    </div>
  );
}
