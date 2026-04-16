---
title: Lung Cancer Risk FastAPI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Lung Cancer Risk FastAPI

This Space serves a trained KNN classifier through FastAPI and includes a built-in browser test page.

## Required files

Put these files inside the `api_artifacts` folder before pushing:

- `lung_cancer_risk_model.joblib`
- `feature_names.json`
- `class_names.json`
- `config.json`

## Endpoints

- `GET /` → test page
- `GET /health` → health check
- `GET /metadata` → model metadata and feature names
- `POST /predict` → prediction endpoint

## Predict request body

```json
{
  "features": {
    "FEATURE_1": 1,
    "FEATURE_2": 2
  }
}
```

Use the exact feature names from `feature_names.json`.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
