import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
STATIC_DIR = BASE_DIR / 'static'

MODEL_PATH = ARTIFACTS_DIR / 'model.pkl'
SCALER_PATH = ARTIFACTS_DIR / 'scaler.pkl'
LABEL_ENCODER_PATH = ARTIFACTS_DIR / 'label_encoder.pkl'
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / 'feature_columns.json'
CONFIG_PATH = ARTIFACTS_DIR / 'config.json'

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac'}
MAX_FILE_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title='COVID Cough Clinical Support API',
    description='Prototype physician decision-support API for cough audio analysis.',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

_state: Dict[str, Any] = {
    'ready': False,
    'errors': [],
    'model': None,
    'scaler': None,
    'label_encoder': None,
    'feature_columns': [],
    'config': {},
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _artifact_paths() -> List[Path]:
    return [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, FEATURE_COLUMNS_PATH, CONFIG_PATH]


def load_artifacts() -> None:
    missing = [str(p.name) for p in _artifact_paths() if not p.exists()]
    if missing:
        _state['ready'] = False
        _state['errors'] = [
            'Missing required artifacts in ./artifacts/: ' + ', '.join(missing),
            'Copy model.pkl, scaler.pkl, label_encoder.pkl, feature_columns.json, and config.json into the artifacts folder.',
        ]
        return

    try:
        _state['model'] = joblib.load(MODEL_PATH)
        _state['scaler'] = joblib.load(SCALER_PATH)
        _state['label_encoder'] = joblib.load(LABEL_ENCODER_PATH)
        _state['feature_columns'] = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding='utf-8'))
        _state['config'] = _load_json(CONFIG_PATH)
        _state['ready'] = True
        _state['errors'] = []
    except Exception as exc:
        _state['ready'] = False
        _state['errors'] = [f'Failed to load artifacts: {exc}']


load_artifacts()


def extract_features(file_path: str, sample_rate: int, duration: float) -> Dict[str, float]:
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError('Empty or unreadable audio file.')

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features: Dict[str, float] = {
        'chroma_stft': float(np.mean(chroma_stft)),
        'rmse': float(np.mean(rmse)),
        'spectral_centroid': float(np.mean(spectral_centroid)),
        'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
        'rolloff': float(np.mean(rolloff)),
        'zero_crossing_rate': float(np.mean(zcr)),
    }

    for i, coef in enumerate(mfcc, start=1):
        features[f'mfcc{i}'] = float(np.mean(coef))

    return features


def build_response(predicted_class: str, probabilities: Dict[str, float], sample_rate: int, duration: float, filename: str) -> Dict[str, Any]:
    covid_prob = probabilities.get('covid')
    if covid_prob is None:
        covid_prob = 1.0 if predicted_class.lower() == 'covid' else 0.0

    support_label = 'higher_covid_likelihood' if predicted_class.lower() == 'covid' else 'lower_covid_likelihood'

    return {
        'filename': filename,
        'predicted_class': predicted_class,
        'support_label': support_label,
        'risk_score': float(covid_prob),
        'class_probabilities': probabilities,
        'clinical_use': 'physician_decision_support',
        'disclaimer': 'This result is for physician decision support only and must not be used as a standalone diagnosis.',
        'sample_rate': sample_rate,
        'analyzed_duration_seconds': duration,
    }


def predict_from_audio(file_path: str, filename: str) -> Dict[str, Any]:
    if not _state['ready']:
        raise RuntimeError('Artifacts are not loaded. ' + ' | '.join(_state['errors']))

    config = _state['config']
    sample_rate = int(config.get('sample_rate', 22050))
    duration = float(config.get('sample_duration', 5))

    feats = extract_features(file_path, sample_rate=sample_rate, duration=duration)
    feature_columns = _state['feature_columns']
    missing = [col for col in feature_columns if col not in feats]
    if missing:
        raise ValueError('Missing expected features: ' + ', '.join(missing))

    x = np.array([[feats[col] for col in feature_columns]], dtype=np.float32)
    x_scaled = _state['scaler'].transform(x)

    model = _state['model']
    label_encoder = _state['label_encoder']

    pred_encoded = model.predict(x_scaled)[0]
    predicted_class = str(label_encoder.inverse_transform([pred_encoded])[0])

    probabilities: Dict[str, float] = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(x_scaled)[0]
        for label, prob in zip(label_encoder.classes_, proba):
            probabilities[str(label)] = float(prob)

    return build_response(predicted_class, probabilities, sample_rate, duration, filename)


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return (STATIC_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/health')
def health() -> Dict[str, Any]:
    return {
        'status': 'ok' if _state['ready'] else 'missing_artifacts',
        'model_loaded': bool(_state['ready']),
        'classes': [] if _state['label_encoder'] is None else [str(x) for x in _state['label_encoder'].classes_],
        'errors': _state['errors'],
    }


@app.post('/reload-artifacts')
def reload_artifacts() -> Dict[str, Any]:
    load_artifacts()
    return health()


@app.post('/predict')
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or 'upload.wav').suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {suffix}')

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        total_size = 0
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=400, detail='Uploaded file is too large.')
                tmp.write(chunk)
        finally:
            await file.close()

    try:
        return predict_from_audio(tmp_path, file.filename or 'upload.wav')
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {exc}')
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('PORT', '7860'))
    uvicorn.run('app:app', host='0.0.0.0', port=port, reload=False)
