import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Charger les noms de classes depuis signnames.csv
signnames_df = pd.read_csv("signnames.csv")
class_names = list(signnames_df.sort_values("ClassId")['SignName'])

MODEL_PATH = "gtsrb_cnn_model_best.keras"
model = load_model(MODEL_PATH)
IMG_SIZE = (48, 48)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    preds = model.predict(img)
    class_id_index = int(np.argmax(preds))
    class_id = class_id_index
    confidence = float(np.max(preds))
    class_name = class_names[class_id_index]
    return JSONResponse({
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": confidence
    })