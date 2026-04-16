---
title: COVID Cough Clinical Support
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# COVID Cough Clinical Support

Prototype FastAPI application for cough-audio analysis as **physician decision support**.

## Before you run or deploy
Copy these training artifacts into the `artifacts/` folder:

- `model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `feature_columns.json`
- `config.json`

## Local run
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Then open:
- `http://127.0.0.1:7860/`
- `http://127.0.0.1:7860/health`

## Hugging Face Spaces
1. Create a new **Docker Space**.
2. Upload all files from this folder.
3. Put your five artifacts inside `artifacts/`.
4. Wait for the build to finish.
5. Open the Space URL and test recording/upload.

## Endpoints
- `GET /` : recording and upload page
- `GET /health` : service health and artifact status
- `POST /predict` : upload audio file for inference
- `POST /reload-artifacts` : reload artifacts after copying them

## Important note
This app is a **prototype clinical support tool** and **must not be used as a standalone diagnosis**.
