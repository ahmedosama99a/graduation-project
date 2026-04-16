# AI-Based Graduation Project for Respiratory Disease Analysis

An integrated graduation project that applies **Artificial Intelligence** in the medical field to support **respiratory disease analysis**, **early risk prediction**, and **smart healthcare solutions** using medical images, respiratory sounds, cough recordings, structured health data, and hardware integration.

---

## Overview

This repository contains the full implementation of our graduation project, which combines multiple AI-based healthcare modules into one unified system.

The project is designed to address different respiratory and health-related tasks through:

- **Chest X-ray image classification**
- **Respiratory sound analysis**
- **Cough-based classification**
- **Cancer risk prediction**
- **API deployment for inference**
- **Hardware-assisted monitoring**

Each module was developed independently and can be studied, tested, and deployed separately.

---

## Project Modules

### 1. COVID Cough Classification
This module focuses on analyzing **cough audio recordings** to identify patterns related to COVID and similar respiratory conditions.

**Main tasks:**
- Audio preprocessing
- Data cleaning
- Feature extraction
- Model training and evaluation
- Deployment preparation

### 2. Cancer Risk Classification
This module predicts **cancer risk level** using structured/tabular patient-related and environmental features.

**Main tasks:**
- Data preprocessing
- Feature selection and handling
- Machine learning model training
- Risk prediction pipeline
- Deployment-ready implementation

### 3. Lung X-ray Classification
This module analyzes **chest X-ray images** to classify lung diseases using deep learning and feature extraction approaches.

**Main tasks:**
- End-to-end deep learning classification
- Feature extraction using multiple backbones
- Explainable AI with **Grad-CAM**
- MobileNet-based experiments
- Model deployment

### 4. Lung Sound Classification
This module classifies **respiratory/lung sounds** using audio signal processing and machine learning techniques.

**Main tasks:**
- Respiratory sound preprocessing
- Audio feature extraction
- Model training and evaluation
- Inference API preparation

### 5. Hardware Integration
This section contains the hardware-related part of the project for respiratory monitoring and data acquisition support.

**Included components:**
- Arduino code
- Hardware notebook
- Design illustration/image

---

## Repository Structure

```bash
graduation-project/
│
├── COVID cough classification/
│   ├── Deployment/
│   └── NoteBook/
│
├── Cancer risk classification/
│   ├── Deployment Model/
│   └── NoteBook/
│
├── Lung X-ray classification/
│   ├── Deployment/
│   └── NoteBook/
│
├── Lung sound classification/
│   ├── Deployment/
│   └── NoteBook/
│
└── hard/
    ├── design.png
    ├── hard Notebook.ipynb
    └── sketch_mar19a.ino
```
Datasets

This project uses multiple public datasets for different healthcare-related tasks:

Lung X-ray / Lung Disease Dataset

Used for chest X-ray image classification.
Source: Kaggle - Lung Disease Dataset

COVID Cough Dataset

Used for cough-based classification and COVID-related audio analysis.
Source: Kaggle - CoughClassifier Trial

Respiratory Sound Dataset

Used for lung sound and respiratory audio classification.
Source: Kaggle - Respiratory Sound Database

Cancer Risk Dataset

Used for cancer risk prediction based on patient and environmental factors.
Source: Kaggle - Cancer Patients and Air Pollution

Technologies Used

This project was built using the following technologies and tools:

Python
Jupyter Notebook
Machine Learning
Deep Learning
Computer Vision
Audio Signal Processing
FastAPI
Hugging Face Spaces
Arduino
Workflow Summary

The general workflow across the project modules includes:

Data collection and loading
Data preprocessing and cleaning
Feature extraction or deep feature learning
Model training and validation
Performance evaluation
Explainability or visualization where needed
Deployment through API or demo applications
Objectives

The main goals of this project are to:

Support respiratory disease analysis using AI
Classify chest X-ray images automatically
Analyze cough and respiratory sounds
Predict cancer-related risk using structured data
Provide deployable AI services for inference
Integrate software intelligence with embedded hardware solutions
Possible Use Cases

This project can be useful in scenarios such as:

Preliminary respiratory disease screening
AI-assisted healthcare research
Educational medical AI projects
Smart diagnostic support systems
Intelligent monitoring applications
Deployment

Some project modules include deployment-ready folders for building APIs or interactive demos.

Examples include:

FastAPI-based inference services
Hugging Face Spaces deployment
Lightweight model-serving pipelines

Each deployment folder contains the files required for running the related module independently.

Notes
Each module is designed as a separate pipeline.
Some parts are focused on experimentation through notebooks.
Other parts are prepared for deployment and practical inference.
This repository acts as a central place for all graduation project components.
Future Improvements

Possible future enhancements include:

Adding a unified frontend for all modules
Improving deployment automation
Adding more detailed documentation for each subproject
Uploading trained model files separately
Adding experiment tracking and reproducibility steps
Expanding evaluation on larger datasets
Integrating all modules into one complete healthcare platform
Contributors

Graduation Project Team
