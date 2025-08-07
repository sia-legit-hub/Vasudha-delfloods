import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import os


def fetch_weather_data():
    api_key = 'HC8QD5Y25CNY89PCZB3643W4X'
    location = 'Delhi,IN'
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/today?unitGroup=metric&key={api_key}&include=current'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data['currentConditions']
        weather_data = {
            'datetime': datetime.now(),
            'temp': current.get('temp'),
            'humidity': current.get('humidity'),
            'precip': current.get('precip'),
            'windspeed': current.get('windspeed')
        }

        df = pd.DataFrame([weather_data])
        file_exists = os.path.isfile('weather_data.csv')
        header_needed = not file_exists or os.stat('weather_data.csv').st_size == 0
        df.to_csv('weather_data.csv', mode='a', header=header_needed, index=False)
        return weather_data
    else:
        st.error("Failed to fetch weather data.")
        return None

# Streamlit App
st.title("🌊 Delhi Flood Prediction")

river_level = st.number_input("Enter current river level (in meters):", min_value=0.0, max_value=15.0)

if st.button("Fetch weather and predict flood"):
    weather = fetch_weather_data()
    if weather:
        st.write("Live weather data:", weather)

        # Create model input
        new_data = pd.DataFrame([{
            'precip': weather['precip'],
            'River_Level': river_level,
            'temp': weather['temp'],
            'humidity': weather['humidity'],
            'windspeed': weather['windspeed']
        }])

        # Make prediction
        prediction = model.predict(new_data)

        # Show result
        if prediction[0] == 1:
            st.error("⚠️ FLOOD RISK PREDICTED!")
        else:
            st.success("✅ No flood expected today.")





