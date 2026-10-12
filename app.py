
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Cutoff Predictor", layout="centered")

st.title("🎓 Cutoff Prediction App")
st.write("Enter details below to predict the admission cutoff.")

# Load the saved regression model
@st.cache_resource
def load_model():
    return joblib.load('best_regression_model.joblib')

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure 'best_regression_model.joblib' is in the same directory. Details: {e}")
    st.stop()

# Create input form for user features
with st.form("prediction_form"):
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2024, step=1)
    university_id = st.text_input("University ID", value="banaras-hindu-university")
    programme_id = st.text_input("Programme ID", value="b-sc-hons-physics")
    seat_type = st.selectbox("Seat Type", options=["general", "Paid Fee Course", "Special Fee Course"])
    category = st.text_input("Category", value="General")
    round_val = st.text_input("Round", value="Round 1")
    
    submit = st.form_submit_button("Predict Cutoff")

if submit:
    # Create a DataFrame from the inputs with correct feature names
    input_data = pd.DataFrame({
        'university_id': [university_id],
        'programme_id': [programme_id],
        'seat_type': [seat_type],
        'category': [category],
        'round': [round_val],
        'year': [year]
    })
    
    # The other columns expected by the model during training can be empty/NaN
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = np.nan
            
    # Reordering columns to match the training layout
    input_data = input_data[model.feature_names_in_]
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        st.balloons()
        st.metric(label="Predicted Cutoff Figure", value=f"{prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
