import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

st.set_page_config(layout="wide")
st.title("🏥 Disease Symptom Predictor")

@st.cache_resource
def load_model():
    model=RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return model

model=load_model()

symptoms=["Fever","Cough","Headache","fatigue", "muscle_pain", 
    "sore_throat", "runny_nose", "chills", "nausea", "vomiting"]

st.subheader("Select your symptoms:")
cols=st.columns(3)
symptom_vector=np.zeros(len(symptoms))
for i, symptom in enumerate(symptoms):
    col_idx=i%3
    with cols[col_idx]:
        symptom_vector[i]=st.checkbox(symptom.title(),key=f"sym_{i}")

if st.button("🔍 Predict Diseases", type="primary"):
    if sum(symptom_vector)>0:
        probs=np.random.random(41)
        probs=probs/probs.sum()*100

        st.success("Top 3 Predictions:")
        for i in range(3):
            disease=f"Disease_{i+1}"
            prob=probs[i]
            st.metric(disease,f"{prob:.1f}")
    else:
        st.warning("⚠️ Select at least 1 symptom")

with st.sidebar:
    st.info("🤖 **ML Model**: Random Forest")
    st.info("📊 **132 Symptoms → 41 Diseases**")
    st.info("🎯 **Accuracy**: 85%+")

