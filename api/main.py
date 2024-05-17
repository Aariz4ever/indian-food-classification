from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model('../models/1')
CLASS_NAMES = ['chapati', 'chicken_tikka', 'jalebi']

@app.get("/ping")
async def ping():
    return 'Hello, I am alive'  # to check the server availability or live

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        print(f"Original image mode: {image.mode}, size: {image.size}")
        image = image.resize((256, 256))
        print(f"Resized image size: {image.size}")
        image = image.convert("RGB")  # Ensure image is in RGB mode
        image = np.array(image)
        print(f"Image shape after conversion to array: {image.shape}, dtype: {image.dtype}")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # file as input using postman POST method
    image = read_file_as_image(await file.read())
    if image is None:
        return {"error": "Failed to process image"}
    image_batch = np.expand_dims(image, 0)
    try:
        predictions = MODEL.predict(image_batch)
        index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[index]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
