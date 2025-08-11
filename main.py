import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.title("DelFloods")
st.write("🌧️ Welcome to our flood prediction model")

# ✅ Cached model training to avoid re-training on each rerun
#@st.cache_resource
def train_model():
    df = pd.read_csv("delhi_flood_data_2023.csv")
    X = df[['precip', 'river_level', 'temp', 'humidity', 'windspeed']]
    y = df['Flood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Train model
model, accuracy = train_model()
## st.write("✅ Model accuracy on test data:", accuracy)

# Input for river level
river_level = st.number_input("🌊 Enter current river level (in meters):", min_value=0.0, step=0.1)

# ✅ Function to safely fetch weather data
def fetch_weather_data():
    location = "Delhi,IN"
    today = datetime.now().strftime("%Y-%m-%d")
    api_key = "HC8QD5Y25CNY89PCZB3643W4X"

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{today}/{today}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": api_key,
        "contentType": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "days" in data and len(data["days"]) > 0:
            today_data = data["days"][0]
            return {
                "precip": today_data.get("precip") or 0.0,
                "temp": today_data.get("temp") or 0.0,
                "humidity": today_data.get("humidity") or 0.0,
                "windspeed": today_data.get("windspeed") or 0.0
            }
    return None

# Get weather
weather = fetch_weather_data()

# Make prediction if both inputs are ready
if weather and river_level:
    st.subheader("📊 Today's Weather Data:")
    st.json(weather)

    # Format data for prediction
    input_data = pd.DataFrame([{
        'precip': weather['precip'],
        'river_level': river_level,
        'temp': weather['temp'],
        'humidity': weather['humidity'],
        'windspeed': weather['windspeed']
    }])

      # Show river level rule-based prediction
    st.subheader("📢 River Level:")
    if river_level > 205.55:
        st.error("🔴 ALERT: River level is above 205.55m. Flood WILL occur.")
    elif river_level > 202:
        st.warning("🟠 WARNING: River level is above 202m. Flood MAY occur.")
    else:
        st.success("🟢 River level is below 202m. Flood is NOT expected from river level alone.")

     # Also run the model prediction
    prediction = model.predict(input_data)[0]
    st.subheader("📊 Model-Based Prediction:")
    if prediction == 1:
        st.error("⚠️ Model says: FLOOD LIKELY – Stay safe!")
    else:
        st.success("✅ Model says: NO FLOOD expected today.")

#REMOVE
#st.write("🧠 Model feature names:", model.feature_names_in_)
'''importances = model.feature_importances_
features = ['precip', 'river_level', 'temp', 'humidity', 'windspeed']
for f, imp in zip(features, importances):
    st.write(f"{f}: {imp:.3f}")'''
import heapq

priority_queue = []

# Example inputs (use your actual inputs here)
river_level = st.number_input("🌊 Enter current river level (in meters):", min_value=0.0, step=0.1)
prediction = model.predict(input_data)[0]
model_proba = model.predict_proba(input_data)[0][1]

# Add alerts with priority (lower number = higher priority)
if river_level > 205.55:
    heapq.heappush(priority_queue, (1, f"🔴 CRITICAL ALERT: River level is dangerously high at {river_level}m! Flood imminent."))
elif river_level > 202:
    heapq.heappush(priority_queue, (2, f"🟠 WARNING: River level is above 202m. Flood possible."))

# Model-based alerts get lower priority (higher number)
if prediction == 1:
    heapq.heappush(priority_queue, (3, f"⚠️ Model predicts flood with probability {model_proba:.2f}"))
else:
    heapq.heappush(priority_queue, (4, "✅ Model says no flood expected."))

# Display alerts in priority order
st.subheader("🚨 Flood Risk Alerts (Priority-based):")
while priority_queue:
    p, msg = heapq.heappop(priority_queue)
    if p == 1:
        st.error(msg)
    elif p == 2:
        st.warning(msg)
    elif p == 3:
        st.warning(msg)
    else:
        st.success(msg)








