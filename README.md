# Disease Symptom Predictor

A Machine Learning web app that predicts the most likely diseases based on selected symptoms using a Random Forest classifier trained on 4,920 patient records.

## Live Demo
https://huggingface.co/spaces/Abhisyanth-M/disease-symptom-predictor

## Problem Statement
70% of India's population lives in rural areas where access to qualified doctors and specialists is extremely limited. Delayed diagnosis leads to delayed treatment and worse health outcomes.

## Solution
A symptom-based disease prediction app that gives patients data-driven insights about their condition instantly — helping them make informed decisions about seeking medical help.

## Features
- Search and select from 133 symptoms
- Predicts Top 3 most likely diseases with probability scores
- Confidence percentage shown for each prediction
- Sidebar showing model accuracy and dataset info
- Trained on real patient data across 41 diseases

## Tech Stack
- Python
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Pandas
- NumPy

## Dataset
- Source: Kaggle — Disease Prediction Using Machine Learning
- Size: 4,920 patient records
- Symptoms: 133
- Diseases: 41

## ML Model
- Algorithm: Random Forest Classifier
- Trees: 100
- Accuracy: 87%+
- Output: Top 3 diseases with probability scores

## How to Run Locally
```bash
git clone https://github.com/Abhisyanth-M/disease-symptom-predictor
cd disease-symptom-predictor
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

## Disclaimer
This app is for educational purposes only. Always consult a qualified doctor for medical advice.

## Limitations
- Predictions are based on statistical patterns in training data
- Not a replacement for professional medical diagnosis
- Dataset is limited to 41 diseases

## GitHub
https://github.com/Abhisyanth-M/disease-symptom-predictor
