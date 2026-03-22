import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv("influenza_clean_weekly.csv")

data = data.sort_values(["country", "year", "week"])

#Lags (last 3 weeks per country)
data["lag1"] = data.groupby("country")["cases"].shift(1)
data["lag2"] = data.groupby("country")["cases"].shift(2)
data["lag3"] = data.groupby("country")["cases"].shift(3)
data["lag4"] = data.groupby("country")["cases"].shift(4)
data["lag5"] = data.groupby("country")["cases"].shift(5)

#Seasonality variables (annual cycle)
data["week_sin"] = np.sin(2 * np.pi * data["week"] / 52)
data["week_cos"] = np.cos(2 * np.pi * data["week"] / 52)

#Eliminate initial rows without lags
data = data.dropna()

#Encode country as number (random forest only understand numbers)
data["country_code"] = data["country"].astype("category").cat.codes

#Features and target
feature_cols = ["country_code", "year", "week_sin", "week_cos", "lag1", "lag2", "lag3", "lag4", "lag5"]
X = data[feature_cols]
y = data["cases"]

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

#Random Forest
rf = RandomForestRegressor(n_estimators = 100, random_state = 30)
rf.fit(X_train, y_train)

#Prediction and metrics
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Prediction (example): Spain, first week of 2023
spain_lags = data[data["country"] == "Spain"].sort_values(["year","week"]).iloc[-5:]
new_data = pd.DataFrame({
    "country_code": [data[data["country"] == "Spain"]["country_code"].iloc[0]],
    "year": [2023],
    "week_sin": [np.sin(2 * np.pi * 1 / 52)],
    "week_cos": [np.cos(2 * np.pi * 1 / 52)],
    "lag1": [spain_lags["cases"].iloc[-1]],
    "lag2": [spain_lags["cases"].iloc[-2]],
    "lag3": [spain_lags["cases"].iloc[-3]],
    "lag4": [spain_lags["cases"].iloc[-4]],
    "lag5": [spain_lags["cases"].iloc[-5]],})

predicted_cases = rf.predict(new_data)
print(f"Predicted cases for Spain 2023-W01: {predicted_cases[0]:.0f}")