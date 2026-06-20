from faster_whisper import WhisperModel

# Load Whisper model once (reused for all calls)
model = WhisperModel(
    "small",
    device="cpu",       # change to "cuda" if GPU available
    compute_type="int8" # change if needed
)

def transcribe_audio(audio_path: str) -> str:
    segments, _ = model.transcribe(audio_path, language="en")
    text = " ".join(segment.text for segment in segments).strip()
    return text


# Example test run
if __name__ == "__main__":
    audio_file = r"C:\Users\Admin\Desktop\myapps\capstone_fusion\capstone_fusion\api_dev\speech.mp3"  # replace with your file
    result = transcribe_audio(audio_file)
    print(result)
