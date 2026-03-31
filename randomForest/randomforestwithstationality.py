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

# Country encoding
data["country_code"] = data["country"].astype("category").cat.codes

feature_cols = ["country_code", "year", "week_sin", "week_cos", "lag1", "lag2", "lag3", "lag4", "rolling_mean_3", "rolling_mean_5"]

X = data[feature_cols]
y = data["cases"]

train_list = []
test_list = []

# TODO: a nivel de serie temporal
for country in data["country"].unique():
    df_country = data[data["country"] == country].sort_values(["year", "week"])
    split_index = int(len(df_country) * 0.8)
    train_list.append(df_country.iloc[:split_index])
    test_list.append(df_country.iloc[split_index:])

train = pd.concat(train_list)
test = pd.concat(test_list)

X_train = train[feature_cols]
y_train = train["cases"]

X_test = test[feature_cols]
y_test = test["cases"]

#HYPERPARAMETERS 
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

grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

print("\n--- Best Parameters ---")
print(grid_search.best_params_)

rf = best_rf
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.2f}")

# BASELINE
y_pred_baseline = X_test["lag1"]

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_base = mean_absolute_error(y_test, y_pred_baseline)
r2_base = r2_score(y_test, y_pred_baseline)

print("\n--- Baseline (lag1) ---")
print(f"RMSE: {rmse_base:.2f}")
print(f"MAE: {mae_base:.2f}")
print(f"R2: {r2_base:.2f}")

#PLOTS
test_sorted = test.sort_values(["year", "week"]).reset_index(drop = True)
X_test_sorted = test_sorted[feature_cols]
y_pred_sorted = rf.predict(X_test_sorted)

test_sorted = test_sorted[(test_sorted["year"] >= 2022) & (test_sorted["year"] <= 2026)]
X_test_sorted = test_sorted[feature_cols]
y_pred_sorted = rf.predict(X_test_sorted)

years = sorted(test_sorted["year"].unique())
palette = sns.color_palette("tab10", len(years))

# 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)

# PLOT 1: REAL
for i, year in enumerate(years):
    df_year = test_sorted[test_sorted["year"] == year]
    axes[0].scatter(df_year["week"], df_year["cases"], color = palette[i], label = str(year), s = 25, alpha = 0.6)

axes[0].set_title("Real Cases (2022-2026)")
axes[0].set_xlabel("Week")
axes[0].set_ylabel("Cases")
axes[0].set_xticks(range(1,53,2))
axes[0].set_ylim(0, 400)
axes[0].legend(title = "Year", bbox_to_anchor = (1.05,1), loc = 'upper left')

# PLOT 2: PREDICTED
for i, year in enumerate(years):
    df_year = test_sorted[test_sorted["year"] == year]
    y_pred_year = rf.predict(df_year[feature_cols])
    axes[1].scatter(df_year["week"], y_pred_year, color = palette[i], label = str(year), s = 25, alpha = 0.6)

axes[1].set_title("Predicted Cases (Random Forest, 2022-2026)")
axes[1].set_xlabel("Week")
axes[1].set_xticks(range(1,53,2))
axes[1].set_ylim(0, 400)
axes[1].legend(title = "Year", bbox_to_anchor = (1.05,1), loc = 'upper left')

plt.tight_layout()
plt.show()


spain_data = data[data["country"] == "Spain"].sort_values(["year", "week"])
last_rows = spain_data.iloc[-5:]

new_data = pd.DataFrame({
    "country_code": [spain_data["country_code"].iloc[0]],
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