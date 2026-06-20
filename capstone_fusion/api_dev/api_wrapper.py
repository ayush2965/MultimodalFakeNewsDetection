# api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from fastapi.middleware.cors import CORSMiddleware
from stt import transcribe_audio


import text_predict
from image_predict import predict_image_and_verify
from text_image_predict import predict_text_image


app = FastAPI(title="Fake News Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



# Utility: save uploaded image

def save_upload(file: UploadFile) -> str:
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(file_path)



# Health route

@app.get("/")
def home():
    return {"status": "API running"}



# TEXT ONLY

@app.post("/predict/text")
async def text_prediction(text: str = Form(...)):
    try:
        confidence, text_outcome, trusted_count, debunk_count, verdictlocal, finalresults, results = text_predict.text_predict(text)

        return JSONResponse({
            "type": "text",
            "result": finalresults,
            "sources": results
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# IMAGE ONLY

@app.post("/predict/image")
async def image_prediction(image: UploadFile = File(...)):
    try:
        image_path = save_upload(image)
        result = predict_image_and_verify(image_path)

        return JSONResponse({
            "type": "image",
            "result": result
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# TEXT + IMAGE (Multimodal)

@app.post("/predict/multimodal")
async def multimodal_prediction(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        image_path = save_upload(image)
        result = predict_text_image(image_path, text)

        return JSONResponse({
            "type": "multimodal",
            "result": result
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# AUDIO

@app.post("/predict/audio")
async def audio_prediction(audio: UploadFile = File(...)):
    try:
        audio_path = save_upload(audio)

        transcription = transcribe_audio(audio_path)
        auconfidence, autext_outcome, autrusted_count, audebunk_count, auverdictlocal, aufinalresults, ausearchresults = text_predict.text_predict(transcription)

        return JSONResponse({
            "type": "audio",
            "result": aufinalresults,
            "sources": ausearchresults
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
