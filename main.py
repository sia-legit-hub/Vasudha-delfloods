import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the data
df = pd.read_csv("delhi_flood_data_2023.csv")

# 2. Features and target
X = df[['Rainfall', 'River_Level', 'Temp', 'Humidity', 'Wind']]
y = df['Flood']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
### print("Accuracy:", accuracy_score(y_test, y_pred)) 
### print(classification_report(y_test, y_pred))

print("\n🔎 Enter today's weather details to predict flood risk:")

rainfall = float(input("🌧️ Rainfall (in mm): "))
river_level = float(input("🌊 River level (in meters): "))
temp = float(input("🌡️ Temperature (°C): "))
humidity = float(input("💧 Humidity (%): "))
wind = float(input("🍃 Wind Speed (km/h): "))

# Format the input as a DataFrame
new_data = pd.DataFrame([{
    'Rainfall': rainfall,
    'River_Level': river_level,
    'Temp': temp,
    'Humidity': humidity,
    'Wind': wind
}])

# Predict
prediction = model.predict(new_data)

# Show result
print("\n📢 Prediction based on your input:")
print("➡️ FLOOD ⚠️" if prediction[0] == 1 else "➡️ NO FLOOD ✅")
