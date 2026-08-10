import os
import io
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key fallback setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JO6Uarqpvuh-s5fh84xDf2d9qyhGxCF2k9ItM9Dmk1vg")
genai.configure(api_key=GEMINI_API_KEY)

# Use fast vision model
model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/")
def read_root():
    return {"status": "Gemini Vision AI backend is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Center crop frame to focus on object held in hand
        width, height = image.size
        cropped_image = image.crop((width * 0.25, height * 0.25, width * 0.75, height * 0.75))

        # Direct Gemini prompt for rapid voice readout
        prompt = "Identify the primary object in this image. Respond with only 1 to 3 words naming the object. No punctuation or full sentences."
        
        response = model.generate_content([prompt, cropped_image])
        clean_text = response.text.strip().capitalize()

        return {"label": clean_text, "confidence": 0.98}

    except Exception as e:
        return {"label": f"Server processing error: {str(e)}"}
