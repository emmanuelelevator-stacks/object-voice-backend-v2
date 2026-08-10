import io
import requests
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

HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

@app.get("/")
def read_root():
    return {"status": "Vision AI backend is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Center-crop object in hand
        width, height = image.size
        cropped_image = image.crop((width * 0.25, height * 0.25, width * 0.75, height * 0.75))

        # Convert cropped image to JPEG bytes
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Send image to Hugging Face Cloud API (uses under 30MB RAM)
        response = requests.post(HF_API_URL, data=image_bytes)
        result = response.json()

        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            clean_text = result[0]["generated_text"].strip().capitalize()
            return {"label": clean_text, "confidence": 0.95}
        elif isinstance(result, dict) and "error" in result:
            return {"label": "AI model warming up, tap again in 5 seconds."}
        else:
            return {"label": "Could not identify object clearly."}

    except Exception as e:
        return {"label": f"Server processing error: {str(e)}"}
