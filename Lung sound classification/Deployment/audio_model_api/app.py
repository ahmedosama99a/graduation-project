import os
import tempfile
import numpy as np
import joblib
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Respiratory Sound Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "saved_models/MLP_model.joblib"
SCALER_PATH = "saved_models/scaler.pkl"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def extract_features_file(wav_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(wav_path, sr=sr)

    if y is None or len(y) == 0:
        raise ValueError("Empty or unreadable audio file")

    duration = librosa.get_duration(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    sc_mean = np.mean(spec_contrast, axis=1)
    sc_std = np.std(spec_contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    feats = np.hstack([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        sc_mean, sc_std,
        np.mean(zcr), np.std(zcr),
        np.mean(centroid), np.std(centroid),
        np.mean(rolloff), np.std(rolloff),
        np.mean(rms), np.std(rms),
        duration
    ])

    return feats


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        features = extract_features_file(temp_path).reshape(1, -1)
        features_scaled = scaler.transform(features)

        pred_idx = model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probs))

        return {
            "filename": file.filename,
            "predicted_class": str(pred_label),
            "confidence": confidence
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)