import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("DelFloods")
st.write("Welcome to our flood prediction model")

# 1. Load the data
df = pd.read_csv("delhi_flood_data_2023.csv")

# 2. Features and target
X = df[['precip', 'River_Level', 'temp', 'humidity', 'windspeed']]
y = df['Flood']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate the model (optional)
y_pred = model.predict(X_test)
st.write("✅ Model Accuracy on Test Set:", accuracy_score(y_test, y_pred))

# 6. Take river level input
river_level = st.number_input("🌊 River level (in meters):", min_value=0.0, max_value=50.0, step=0.1)

# 7. Fetch weather data
def fetch_weather_data():
    location = "Delhi,IN"
    today = datetime.now().strftime("%Y-%m-%d")
    api_key = "HC8QD5Y25CNY89PCZB3643W4X"  # Replace with your real key

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
                "precip": today_data.get("precip", 0),
                "temp": today_data.get("temp", 0),
                "humidity": today_data.get("humidity", 0),
                "windspeed": today_data.get("windspeed", 0)
            }
        else:
            st.warning("⚠️ No weather data found for today.")
            return None
    else:
        st.error(f"❌ Failed to fetch weather data. Code: {response.status_code}")
        return None

# ✅ Save the result of fetch_weather_data()
weather = fetch_weather_data()

# 8. Make prediction only if both inputs are ready
if weather and river_level:
    st.subheader("📊 Today's Weather:")
    st.write(weather)

    new_data = pd.DataFrame([{
        'precip': weather['precip'],
        'River_Level': river_level,
        'temp': weather['temp'],
        'humidity': weather['humidity'],
        'windspeed': weather['windspeed']
    }])

    prediction = model.predict(new_data)

    st.subheader("📢 Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ FLOOD LIKELY – Stay alert!")
    else:
        st.success("✅ NO FLOOD expected today.")
else:
    st.info("Please enter a valid river level and make sure weather data is available.")
