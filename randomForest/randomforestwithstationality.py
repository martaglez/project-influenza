import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(__file__)  

csv_path = os.path.join(base_dir, "..", "data", "influenza_clean_weekly.csv")

data = pd.read_csv(csv_path)

data = data.sort_values(["country", "year", "week"])

# Lags
for lag in range(1, 5):
    data[f"lag{lag}"] = data.groupby("country")["cases"].shift(lag)

# Rolling means
data["rolling_mean_3"] = data.groupby("country")["cases"].transform(lambda x: x.rolling(3).mean())
data["rolling_mean_5"] = data.groupby("country")["cases"].transform(lambda x: x.rolling(5).mean())

# Seasonality
data["week_sin"] = np.sin(2 * np.pi * data["week"] / 52)
data["week_cos"] = np.cos(2 * np.pi * data["week"] / 52)

# Eliminar NaNs
data = data.dropna()

countries_series = data["country"].copy()

data = pd.get_dummies(data, columns=["country"], drop_first=True)

feature_cols = [
    "year",
    "week_sin",
    "week_cos",
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "rolling_mean_3",
    "rolling_mean_5"
] + [col for col in data.columns if col.startswith("country_")]

X = data[feature_cols]
y = data["cases"]


train_list = []
val_list = []
test_list = []
for country in countries_series.unique():
    df_country = data[countries_series == country].sort_values(["year", "week"])
    
    n = len(df_country)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_list.append(df_country.iloc[:train_end])
    val_list.append(df_country.iloc[train_end:val_end])
    test_list.append(df_country.iloc[val_end:])

train = pd.concat(train_list).reset_index(drop = True)
val = pd.concat(val_list).reset_index(drop = True)
test = pd.concat(test_list).reset_index(drop = True)

output_dir = os.path.join(base_dir, "..", "data")

train.to_csv(os.path.join(output_dir, "train.csv"), index = False)
val.to_csv(os.path.join(output_dir, "val.csv"), index = False)
test.to_csv(os.path.join(output_dir, "test.csv"), index = False)

print("Datasets saved: train.csv, val.csv, test.csv")

X_train = train[feature_cols]
y_train = train["cases"]

X_val = val[feature_cols]
y_val = val["cases"]

X_test = test[feature_cols]
y_test = test["cases"]


# Model + hyperparameters
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestRegressor(random_state = 30, n_jobs = -1)

grid_search = GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 3,
    scoring = "neg_mean_squared_error",
    verbose = 1
)

# Train
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\n--- Best Parameters ---")
print(grid_search.best_params_)


# Validation
y_val_pred = best_rf.predict(X_val)

rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print("\n--- Validation (RF) ---")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAE: {mae_val:.2f}")
print(f"R2: {r2_val:.2f}")


# Final model
final_rf = best_rf
final_rf.fit(X_train, y_train)


# Test
y_test_pred = final_rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n--- Test (RF) ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.2f}")


# Feature importance
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\n--- Feature Importances ---")
print(importances)


# BASELINE
y_val_baseline = X_val["lag1"]

rmse_val_base = np.sqrt(mean_squared_error(y_val, y_val_baseline))
mae_val_base = mean_absolute_error(y_val, y_val_baseline)
r2_val_base = r2_score(y_val, y_val_baseline)

print("\n--- Baseline (Validation) ---")
print(f"RMSE: {rmse_val_base:.2f}")
print(f"MAE: {mae_val_base:.2f}")
print(f"R2: {r2_val_base:.2f}")

y_test_baseline = X_test["lag1"]

rmse_test_base = np.sqrt(mean_squared_error(y_test, y_test_baseline))
mae_test_base = mean_absolute_error(y_test, y_test_baseline)
r2_test_base = r2_score(y_test, y_test_baseline)

print("\n--- Baseline (Test) ---")
print(f"RMSE: {rmse_test_base:.2f}")
print(f"MAE: {mae_test_base:.2f}")
print(f"R2: {r2_test_base:.2f}")

#PLOTS
# Obtener columnas de países dummy
country_cols = [c for c in data.columns if c.startswith("country_")]

# REAL
plt.figure(figsize=(14, 6))

for col in country_cols:
    country_name = col.replace("country_", "")

    df_country = data[data[col] == 1].sort_values(["year", "week"])

    time = df_country["year"] + df_country["week"] / 52

    plt.plot(time, df_country["cases"], label=country_name, alpha=0.7)

plt.title("Influenza cases over time by country")
plt.xlabel("Year")
plt.ylabel("Cases")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# PREDICTIONS
plt.figure(figsize=(14, 6))

for col in country_cols:
    country_name = col.replace("country_", "")

    df_country = data[data[col] == 1].sort_values(["year", "week"])

    X_country = df_country[feature_cols]
    y_pred = final_rf.predict(X_country)

    time = df_country["year"] + df_country["week"] / 52

    plt.plot(time, y_pred, label=country_name, alpha=0.4)

plt.title("Model predictions across countries")
plt.xlabel("Year")
plt.ylabel("Predicted cases")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# PARITY PLOT
y_pred = final_rf.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.title("Parity plot (Real vs predicted)")
plt.xlabel("Real cases")
plt.ylabel("Predicted cases")

plt.tight_layout()
plt.show()

# SPAIN 
country_col = "country_Spain"

df_country = data[data[country_col] == 1].sort_values(["year", "week"])

X_country = df_country[feature_cols]
y_real = df_country["cases"]
y_pred = final_rf.predict(X_country)

time = df_country["year"] + df_country["week"] / 52

plt.figure(figsize=(14, 5))
plt.plot(time, y_real, label="Real cases", alpha=0.7)
plt.plot(time, y_pred, label="Predicted cases", alpha=0.7)

plt.title("Real vs predicted influenza cases - Spain")
plt.xlabel("Year")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.show()


""" spain_data = data[data["country"] == "Spain"].sort_values(["year", "week"])
last_rows = spain_data.iloc[-5:]

new_data = pd.DataFrame({
    "lat": [spain_data["lat"].iloc[0]],
    "lon": [spain_data["lon"].iloc[0]],
    "year": [2024],
    "week_sin": [np.sin(2 * np.pi * 1 / 52)],
    "week_cos": [np.cos(2 * np.pi * 1 / 52)],
    "lag1": [last_rows["cases"].iloc[-1]],
    "lag2": [last_rows["cases"].iloc[-2]],
    "lag3": [last_rows["cases"].iloc[-3]],
    "lag4": [last_rows["cases"].iloc[-4]],
    "rolling_mean_3": [last_rows["cases"].iloc[-3:].mean()],
    "rolling_mean_5": [last_rows["cases"].mean()]
})

predicted_cases = rf.predict(new_data)

print(f"\nPredicted cases for Spain 2024-W01: {predicted_cases[0]:.0f}")
 """

""" 
#TODO: LINEAL
# LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline: escalado + modelo
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train
lr_pipeline.fit(X_train, y_train)

# Prediction
y_pred_lr = lr_pipeline.predict(X_test)

# Metrics
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n--- Linear Regression ---")
print(f"Root Mean Squared Error: {rmse_lr:.2f}")
print(f"Mean Absolute Error: {mae_lr:.2f}")
print(f"R2 Score: {r2_lr:.2f}")

# COEFFICIENTS

lr_model = lr_pipeline.named_steps["model"]

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_
})

print("\n--- Feature Importance (Linear Model) ---")
print(coef_df.sort_values(by="coefficient", ascending=False))
 """