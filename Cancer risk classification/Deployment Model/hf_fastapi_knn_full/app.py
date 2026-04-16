from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "api_artifacts"
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = ARTIFACTS_DIR / "lung_cancer_risk_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values keyed by feature name")


class PredictResponse(BaseModel):
    predicted_class_index: int
    predicted_class_name: str
    probabilities: Dict[str, float] | None = None
    model_name: str


app = FastAPI(
    title="Lung Cancer Risk Classification API",
    description="FastAPI service for KNN-based lung cancer risk prediction with a built-in test page.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = joblib.load(MODEL_PATH)
    feature_names: List[str] = _load_json(FEATURES_PATH)
    class_names: List[str] = _load_json(CLASS_NAMES_PATH)
    config: Dict[str, Any] = _load_json(CONFIG_PATH)
except Exception as exc:  # pragma: no cover
    model = None
    feature_names = []
    class_names = []
    config = {"load_error": str(exc)}


def validate_and_build_frame(features: Dict[str, float]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in features]
    extra = [name for name in features if name not in feature_names]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features.",
                "missing_features": missing,
            },
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected extra features found.",
                "extra_features": extra,
            },
        )

    ordered_values: Dict[str, float] = {}
    for name in feature_names:
        value = features[name]
        try:
            ordered_values[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Feature '{name}' must be numeric.",
                    "received_value": value,
                },
            )

    return pd.DataFrame([ordered_values], columns=feature_names)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "error",
        "artifacts_loaded": model is not None,
        "model_path": MODEL_PATH.name,
        "feature_names_path": FEATURES_PATH.name,
        "class_names_path": CLASS_NAMES_PATH.name,
        "config": config,
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail=config)
    return {
        "model_name": config.get("model_name", "unknown"),
        "task_type": config.get("task_type", "multiclass_classification"),
        "class_names": class_names,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "supports_predict_proba": hasattr(model, "predict_proba"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=500, detail=config)

    frame = validate_and_build_frame(request.features)
    pred_idx = int(model.predict(frame)[0])

    if pred_idx < 0 or pred_idx >= len(class_names):
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Predicted class index is out of range.",
                "predicted_index": pred_idx,
                "class_names": class_names,
            },
        )

    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        probabilities = {class_names[i]: float(prob) for i, prob in enumerate(probs)}

    return PredictResponse(
        predicted_class_index=pred_idx,
        predicted_class_name=class_names[pred_idx],
        probabilities=probabilities,
        model_name=config.get("model_name", "unknown"),
    )
