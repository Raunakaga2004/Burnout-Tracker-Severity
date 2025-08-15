import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

# Paths
DATA_PATH = "data/synthetic_burnout_data.csv"
MODEL_PATH = "models/burnout_model.pkl"

# Load model if exists, else create new
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Load dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # Empty dataset with correct columns
    df = pd.DataFrame(columns=[
        "screen_time_hours", "sleep_duration_hours", "study_time_hours",
        "workout_time_hours", "stress_level", "anxiety_level",
        "coffee_intake_mg", "socializing_time_hours", "water_intake_litres",
        "junk_food_intake", "burnout_score"
    ])

# Streamlit UI
st.title("Burnout Prediction App")
st.write("Fill in your daily habits to see your burnout score.")

# User inputs
screen_time_hours = st.number_input("Screen Time (hours)", 0.0, 16.0, 6.0)
sleep_duration_hours = st.number_input("Sleep Duration (hours)", 0.0, 12.0, 7.0)
study_time_hours = st.number_input("Study Time (hours)", 0.0, 12.0, 4.0)
workout_time_hours = st.number_input("Workout Time (hours)", 0.0, 3.0, 1.0)
stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
anxiety_level = st.slider("Anxiety Level (1-5)", 1, 5, 3)
coffee_intake_mg = st.number_input("Coffee Intake (mg)", 0.0, 500.0, 150.0)
socializing_time_hours = st.number_input("Socializing Time (hours)", 0.0, 8.0, 2.0)
water_intake_litres = st.number_input("Water Intake (litres)", 0.5, 5.0, 2.0)
junk_food_intake = st.slider("Junk Food Intake (1-5)", 1, 5, 3)

if st.button("Predict Burnout Score"):
    # Prepare user input row
    new_row = {
        "screen_time_hours": screen_time_hours,
        "sleep_duration_hours": sleep_duration_hours,
        "study_time_hours": study_time_hours,
        "workout_time_hours": workout_time_hours,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
        "coffee_intake_mg": coffee_intake_mg,
        "socializing_time_hours": socializing_time_hours,
        "water_intake_litres": water_intake_litres,
        "junk_food_intake": junk_food_intake
    }

    # Predict burnout score
    if len(df) > 0:
        X = df.drop(columns=["burnout_score"])
        y = df["burnout_score"]
        model.fit(X, y)  # Ensure latest training on current dataset
        burnout_score = model.predict(pd.DataFrame([new_row]))[0]
    else:
        burnout_score = 0  # No data yet

    st.success(f"Predicted Burnout Score: {burnout_score:.2f}")

    # Append to dataset
    new_row["burnout_score"] = burnout_score
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    # Retrain model on full dataset
    X = df.drop(columns=["burnout_score"])
    y = df["burnout_score"]
    model.fit(X, y)

    # Save updated model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    st.info("Model retrained with your data!")
