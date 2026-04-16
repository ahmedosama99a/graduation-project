from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import io

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

APP_TITLE = "Lung MobileNet API"

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mobilenet_lung_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"

app = FastAPI(title=APP_TITLE)

model = None
class_names = None
config = None


def enhance_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    image = cv2.filter2D(image, -1, kernel)

    hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return image


def load_artifacts() -> None:
    global model, class_names, config

    missing = [str(p.name) for p in [MODEL_PATH, CLASS_NAMES_PATH, CONFIG_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names_data = json.load(f)
    class_names = class_names_data["class_names"]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    img_size = tuple(config.get("img_size", [256, 256]))
    img = img.resize(img_size)
    arr = np.array(img).astype(np.float32)

    if config.get("preprocessing", {}).get("enhance_image", False):
        arr = enhance_image(arr)

    rescale = config.get("preprocessing", {}).get("rescale", None)
    if rescale is not None:
        arr = arr * float(rescale)

    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/")
def home():
    index_path = BASE_DIR / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names,
        "errors": []
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    x = preprocess_image_bytes(image_bytes)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return JSONResponse(
        {
            "filename": file.filename,
            "predicted_class": class_names[pred_idx],
            "confidence": round(confidence, 6),
            "class_probabilities": {
                class_names[i]: round(float(probs[i]), 6) for i in range(len(class_names))
            }
        }
    )


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
