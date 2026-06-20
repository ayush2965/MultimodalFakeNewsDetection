import React from "react";
import "./index.css";
import logo from "./assets/newsp.png";
import committeeImage from "./assets/charcha3.png";
import dugongImage from "./assets/Dugong.png";
import economicImage from "./assets/finalsharks.png";
import kbcImage from "./assets/kbc5.png";
import indianIdolImage from "./assets/indianidolfinal.png";
import aletheiaImage from "./assets/aletheia.png";
import { useState } from "react";
import Verify from "./components/Verify";

export default function App() {
  const [newsText, setNewsText] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);

  const API_BASE = "http://127.0.0.1:8000";

  /* ---------------- Speech Recognition ---------------- */
  const startSpeechRecognition = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech recognition not supported.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      setNewsText(event.results[0][0].transcript);
    };

    recognition.start();
  };

  /* ---------------- Image Upload ---------------- */
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
  };

  /* ---------------- Verify Logic ---------------- */
  const verifyNews = async () => {
    try {
      let response;

      if (selectedImage && newsText.trim() !== "") {
        const formData = new FormData();
        formData.append("image", selectedImage);
        formData.append("text", newsText);

        response = await fetch(`${API_BASE}/predict/multimodal`, {
          method: "POST",
          body: formData,
        });
      } else if (selectedImage) {
        const formData = new FormData();
        formData.append("image", selectedImage);

        response = await fetch(`${API_BASE}/predict/image`, {
          method: "POST",
          body: formData,
        });
      } else {
        const formData = new FormData();
        formData.append("text", newsText);

        response = await fetch(`${API_BASE}/predict/text`, {
          method: "POST",
          body: formData,
        });
      }

      const data = await response.json();
      alert(JSON.stringify(data, null, 2));
    } catch (err) {
      console.error(err);
      alert("Verification failed");
    }
  };
  return (
    <div className="newspaper">

      {/* ===== Header ===== */}
      <header className="masthead">
        <div className="mast-top">
          RESTRICTING MISINFORMATION, EMPOWERING TRUTH
          <span>VERIFYING AND DETECTING DIGITALLY SPREAD MULTIMODAL</span>
        </div>

        <div className="title-row">
          <h1>ALETHEIA</h1>

          <img src={logo} alt="Newspaper Logo" className="paper-logo" />

          <h1>FAKE NEWS</h1>
        </div>

        <div className="meta-row">
          <span>Verifying Since 2026</span>
          <span>Sunday Morning</span>
          <span>20 Aug, 1922</span>
          <span>Price 0.00$</span>
        </div>
      </header>

      {/* ===== Main Page Grid ===== */}
      <main className="grid">

        {/* Left Article */}
        <section className="panel tall">
          <h2>NATION TO SOLVE ALL PROBLEMS BY CREATING A COMMITTEE TO STUDY THEM FOR 25 YEARS</h2>
          <img src={committeeImage} alt="Committee Meeting" className="article-image" />
          <p>
In a landmark moment of administrative efficiency, the nation today proudly announced the formation of the Supreme Long-Term Committee for the Comprehensive Evaluation of All Existing and Future Problems, a body expected to deliver solutions sometime around the year 2051.

Officials confirmed the committee was formed after intense discussions on how to immediately address pressing issues affecting citizens. After several rounds of meetings, members unanimously agreed that the best immediate action was to study the matter very carefully for a very long time.

"The people deserve well-researched solutions," said the committee's interim spokesperson. "Rushing into solving problems could create new problems, which would require another committee. We are preventing future committees."

The committee will first spend three years deciding meeting schedules, followed by five years determining terminology definitions, and another seven years gathering opinions on whether the problems are truly problems.
          </p>
        </section>

        {/* NEWS VERIFICATION DESK - Divided into two parts */}
        <section className="panel feature verification-desk">
          {/* Upper Part - Article Style */}
          <div className="verification-upper">
            <h2>NEWS VERIFICATION DESK</h2>
            <div className="horizontal-layout">
              
  <img src={aletheiaImage} className="logo-culture-image-float" alt="Aletheia verification desk" />


              <div className="text-content">
                <p>
                  Aletheia, a new multimodal fake news verification desk, reported a growing challenge in combating misinformation as viral stories increasingly rely on unverified forwards and outdated content. Following a recent internal review, the agency strengthened its verification process and urged citizens to confirm sources and check publication details before sharing news online, emphasizing that responsible sharing plays a key role in maintaining public trust in information. The agency also announced plans to collaborate with media platforms and educational institutions to reduce the spread of misleading content.

                </p>
              
              </div>
            </div>
          </div>

         {/* Lower Part - Input Section */}
<div className="verification-lower">
 <Verify />
  {/* Social Media Buttons */}
          <div className="social-buttons">
            <a 
              href="https://instagram.com/yourpage" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-btn instagram"
              aria-label="Instagram"
            >
              <svg viewBox="0 0 24 24" fill="black">
                <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/>
              </svg>
            </a>
            
            <a 
              href="https://discord.gg/yourserver" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-btn discord"
              aria-label="Discord"
            >
              <svg viewBox="0 0 24 24" fill="black">
                <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
              </svg>
            </a>
            
            <a 
              href="https://t.me/yourchannel" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-btn telegram"
              aria-label="Telegram"
            >
              <svg viewBox="0 0 24 24" fill="black">
                <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12a12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472c-.18 1.898-.962 6.502-1.36 8.627c-.168.9-.499 1.201-.82 1.23c-.696.065-1.225-.46-1.9-.902c-1.056-.693-1.653-1.124-2.678-1.8c-1.185-.78-.417-1.21.258-1.91c.177-.184 3.247-2.977 3.307-3.23c.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345c-.48.33-.913.49-1.302.48c-.428-.008-1.252-.241-1.865-.44c-.752-.245-1.349-.374-1.297-.789c.027-.216.325-.437.893-.663c3.498-1.524 5.83-2.529 6.998-3.014c3.332-1.386 4.025-1.627 4.476-1.635z"/>
              </svg>
            </a>
            
            <a 
              href="https://wa.me/1234567890" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-btn whatsapp"
              aria-label="WhatsApp"
            >
              <svg viewBox="0 0 24 24" fill="black">
                <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967c-.273-.099-.471-.148-.67.15c-.197.297-.767.966-.94 1.164c-.173.199-.347.223-.644.075c-.297-.15-1.255-.463-2.39-1.475c-.883-.788-1.48-1.761-1.653-2.059c-.173-.297-.018-.458.13-.606c.134-.133.298-.347.446-.52c.149-.174.198-.298.298-.497c.099-.198.05-.371-.025-.52c-.075-.149-.669-1.612-.916-2.207c-.242-.579-.487-.5-.669-.51c-.173-.008-.371-.01-.57-.01c-.198 0-.52.074-.792.372c-.272.297-1.04 1.016-1.04 2.479c0 1.462 1.065 2.875 1.213 3.074c.149.198 2.096 3.2 5.077 4.487c.709.306 1.262.489 1.694.625c.712.227 1.36.195 1.871.118c.571-.085 1.758-.719 2.006-1.413c.248-.694.248-1.289.173-1.413c-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214l-3.741.982l.998-3.648l-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884c2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/>
              </svg>
            </a>
          </div>
</div>

        </section>

        {/* Right Article */}
        <section className="panel tall">
          <h2>SHARK TANK INDIA ANNOUNCES ACTUAL SHARKS FOR AUTHENTIC FEEDBACK IN NEW SEASON</h2>
          <img src={economicImage} alt="Economic Analysis" className="article-image" />
          <p>
            Producers of Shark Tank have officially announced a bold reboot aimed at restoring “raw honesty” to the startup ecosystem by replacing celebrity investors with actual, live sharks. According to insiders, founders will now pitch their ideas from a reinforced glass platform suspended over a saltwater tank, while the sharks circle silently, offering what the network calls “instinct-driven market validation.” Early trials suggest feedback is immediate and brutally clear: ideas with weak revenue models are ignored entirely, while overly ambitious valuations result in sudden splashing and emergency commercial breaks. The channel insists safety protocols are in place, noting that life jackets, first-aid teams, and a strong swim stroke are now considered “essential entrepreneurial skills. Several venture capitalists have reportedly volunteered as guest judges, saying the new format finally reflects how ruthless fundraising can feel in real life, with several news sponsors already lining up.
          </p>
        </section>

        {/* Bottom Panels */}
        <section className="panel">
          <h3>KILLER DUGONGS SPOTTED IN AUSTRALIA</h3>
          <img src={dugongImage} alt="Dugong illustration" className="culture-image-float" />
          <p>
            Killer Dugongs have allegedly been spotted off the coast of Australia, sending shockwaves through absolutely no marine biologist anywhere. Yes, the same gentle, slow-moving sea cows famous for grazing sea grass like underwater lawnmowers have apparently decided to rebrand as apex predators overnight. Eyewitnesses claim these fearsome creatures were “lurking suspiciously” and “floating menacingly,” which experts confirm is just… existing. Authorities assure the public that dugongs remain dangerously aggressive only toward sea grass and are far more likely to nap than hunt humans.
          </p>
        </section>

     {/* MAJOR DISCOVERY - Now in bottom section */}
<section className="panel horizontal-panel">
  <h3>
    INDIAN IDOL INTRODUCES "AUTO-TUNE HONESTY ROUND," CONTESTANTS ASKED TO SING WITHOUT BACKSTORY
  </h3>
  <div style={{ overflow: "hidden" }}>
    <img src={indianIdolImage} alt="Indian Idol Segment" className="idol-culture-image-float" />
    <p style={{ margin: 0, textAlign: "justify", lineHeight: "1.35", fontSize: "15px" }}>
      Contestants are now reportedly given only a microphone and ten seconds of silence before singing, a moment producers claim reveals more nerves than any dramatic intro package ever could. Judges, long accustomed to emotional cues from background scores, were briefly seen looking around the studio in confusion when no piano music arrived to guide their reactions.

Sources say several participants attempted to sneak emotional details into song introductions, beginning lines with, “This next song is for…” before producers swiftly cut them off and reminded them of the new rules. One crew member admitted that the editing team initially struggled, as episodes suddenly became shorter without extended slow-motion reaction shots and tearful family interviews.

Social media reactions have been divided, with some viewers praising the return to talent-focused competition while others confess they miss guessing contestants’ life stories before hearing the first note. Meme pages quickly joked that the toughest challenge for singers is now surviving sixty seconds without mentioning their hometown’s support.

Meanwhile, backstage vocal coaches say the new round has unintentionally improved rehearsals, as contestants now spend more time practicing pitch and less time perfecting emotional camera angles. Industry analysts believe the segment could inspire other reality shows to experiment with stripped-down formats, though insiders doubt producers will abandon drama entirely.

As one longtime crew member reportedly summed it up, “Reality TV without extra drama works surprisingly well — but don’t worry, the violins will probably return next week.”
    </p>
  </div>
</section>


        <section className="panel">
          <h3>AMITABH BACHCHAN ‘BANKRUPT’ AFTER GIVING AWAY CRORES ON KBC? TRUTH BEHIND THE VIRAL CLAIM
</h3>
          <img src={kbcImage} alt="KBC illustration" className="culture-image-float" />
          <p>
            Rumors circulating online claim that veteran actor Amitabh Bachchan has gone bankrupt after distributing large prize amounts to contestants on Kaun Banega Crorepati. However, these claims are misleading. Prize money awarded on the show is funded by the production house and sponsoring partners, not from Bachchan’s personal wealth. Industry sources confirm that the actor continues to remain financially stable, and the viral claim appears to be exaggerated social media speculation rather than fact.
          </p>
        </section>

      </main>
    </div>
  );
}
