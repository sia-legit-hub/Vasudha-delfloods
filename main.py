import streamlit as st
st.title("DelFloods")
st.write("Welcome to our flood prediction model")

import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n🔎 Enter today's weather details to predict flood risk:")

river_level = st.number_input("🌊 River level (in meters): ")


def fetch_weather_data():
    # Define location and date (Delhi, India)
    location = "Delhi,IN"
    today = datetime.now().strftime("%Y-%m-%d")

    # Visual Crossing API Key (replace with your actual key)
    api_key = "HC8QD5Y25CNY89PCZB3643W4X"  # <- Replace this with your API key

    # Build the API endpoint
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{today}/{today}"
    
    # Query parameters
    params = {
        "unitGroup": "metric",       # Use "us" for Fahrenheit and mph
        "include": "days",
        "key": api_key,
        "contentType": "json"
    }

    # Make the request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Print the weather data for today
        today_data = data.get("days", [{}])[0]
        print(f"Weather in {location} on {today}:")
        print(today_data)
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print(response.text)

# Call the function
fetch_weather_data()

# Format the input as a DataFrame
new_data = pd.DataFrame([{
    'precip': precip,
    'River_Level': river_level,
    'temp': temp,
    'humidity': humidity,
    'windspeed': wind
}])

# Predict
prediction = model.predict(new_data)

# Show result
print("\n📢 Prediction based on your input:")
print("➡️ FLOOD ⚠️" if prediction[0] == 1 else "➡️ NO FLOOD ✅")





