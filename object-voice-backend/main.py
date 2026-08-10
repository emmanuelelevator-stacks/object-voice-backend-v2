import os
import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Checks Render environment variable first; uses direct key as fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

@app.get("/")
def read_root():
    return {"status": "Groq Vision AI backend is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
            return {"label": "Server error: Please set GROQ_API_KEY in main.py or Render."}

        client = Groq(api_key=GROQ_API_KEY)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Center crop frame
        width, height = image.size
        cropped_image = image.crop((width * 0.25, height * 0.25, width * 0.75, height * 0.75))

        # Convert image to Base64
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Send request to Groq Vision
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the primary object in this image. Respond with only 1 to 3 words naming the object. No punctuation or extra words."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=20
        )

        clean_text = response.choices[0].message.content.strip().capitalize()
        return {"label": clean_text}

    except Exception as e:
        return {"label": f"Server processing error: {str(e)}"}
