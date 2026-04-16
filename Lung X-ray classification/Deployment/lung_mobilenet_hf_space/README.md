---
title: Lung MobileNet API
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Lung MobileNet API

FastAPI app for chest X-ray classification.

## Endpoints
- `/` simple upload page
- `/health`
- `/docs`
- `/predict`

## Required files in `artifacts/`
- `mobilenet_lung_model.keras`
- `class_names.json`
- `config.json`
