import streamlit as st
import numpy as np
import pickle

st.set_page_config(layout="wide", page_title="Disease Symptom Predictor")
st.title("Disease Symptom Predictor")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("symptoms.pkl", "rb") as f:
        symptoms = pickle.load(f)
    return model, symptoms

model, symptoms = load_model()

# Display symptoms as clean labels
def fmt(s):
    return s.strip().replace("_", " ").title()

symptom_labels = [fmt(s) for s in symptoms]

st.subheader("Select your symptoms:")

# Searchable multiselect for 132 symptoms
selected = st.multiselect(
    "Search and select symptoms (type to filter):",
    options=symptom_labels,
    placeholder="Start typing a symptom...",
)

# Also show checkbox grid for quick browsing
with st.expander("Or browse all symptoms"):
    cols = st.columns(4)
    checked = {}
    for i, label in enumerate(symptom_labels):
        with cols[i % 4]:
            checked[label] = st.checkbox(label, key=f"chk_{i}")

# Combine multiselect + checkboxes
all_selected = set(selected) | {lbl for lbl, val in checked.items() if val}

st.markdown(f"**{len(all_selected)} symptom(s) selected**")

if st.button("Predict Disease", type="primary"):
    if not all_selected:
        st.warning("Select at least one symptom.")
    else:
        # Build input vector
        vec = np.array([1 if fmt(s) in all_selected else 0 for s in symptoms]).reshape(1, -1)

        proba = model.predict_proba(vec)[0]
        top3_idx = np.argsort(proba)[::-1][:3]

        st.success("Top 3 Predictions")
        for rank, idx in enumerate(top3_idx, 1):
            disease = model.classes_[idx]
            prob = proba[idx] * 100
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(int(prob), text=f"**#{rank} {disease}**")
            with col2:
                st.metric("Probability", f"{prob:.1f}%")

with st.sidebar:
    st.info("**ML Model**: Random Forest (200 trees)")
    st.info(f"**Symptoms**: {len(symptoms)}")
    st.info(f"**Diseases**: {len(model.classes_)}")
    st.info("**Training Accuracy**: 100%")
