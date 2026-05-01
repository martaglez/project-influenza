import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# LOAD DATA
#base_dir = os.path.dirname(__file__)
#data_dir = os.path.join(base_dir, "..", "data")
data_dir = "data"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
val = pd.read_csv(os.path.join(data_dir, "val.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

# FILTRAR PAÍS
country_col = "country_Spain"

train = train[train[country_col] == 1]
val = val[val[country_col] == 1]
test = test[test[country_col] == 1]

# FEATURE TIEMPO
def create_time(df):
    return (df["year"] - 2022) + (df["week"] / 52)

train["t"] = create_time(train)
val["t"] = create_time(val)
test["t"] = create_time(test)

# ordenar
train = train.sort_values("t")
val = val.sort_values("t")
test = test.sort_values("t")

# FEATURES (con lags)
#features = ["t", "lag1"] 
features = ["t"] 

X_train = train[features].values
y_train = train["cases"].values

X_val = val[features].values
y_val = val["cases"].values

X_test = test[features].values
y_test = test["cases"].values

# ESCALADO
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# KERNEL
kernel = (
    C(5.0, (0.1, 10)) * RBF(length_scale = 1, length_scale_bounds = (0.1, 5)) #si pongo (0.1, 5) empeora mucho
    + C(5.0, (0.1, 10)) * ExpSineSquared( #modela ciclos
         length_scale = 1.0,
         periodicity = 1.0,
         periodicity_bounds = "fixed"
     ) #sin esto btt peor
    + WhiteKernel(noise_level = 1.0, noise_level_bounds = (0.1, 5))
    #con WhiteKernel(noise_level = 0.5, noise_level_bounds = (1e-2, 2)) muchísimo peor
)

# MODELO
gpr = GaussianProcessRegressor(
    kernel = kernel,
    alpha = 1e-4, 
    n_restarts_optimizer = 3,
    random_state = 42
)

# TRAIN
gpr.fit(X_train_s, y_train_s)

print("\nKernel aprendido:")
print(gpr.kernel_)



ypred, ystd = gpr.predict(X_train_s, return_std=True)
plt.figure(figsize=(16,9))
plt.scatter(X_train_s, y_train_s)
plt.plot(X_train_s, ypred)
plt.fill_between(X_train_s.flatten(), ypred - 2*ystd, ypred + 2*ystd, alpha=0.3)
plt.show()





# VALIDATION
y_val_pred_s, y_val_std = gpr.predict(X_val_s, return_std = True)
y_val_pred = scaler_y.inverse_transform(y_val_pred_s.reshape(-1, 1)).ravel()

rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print("\n--- Validation (GPR) ---")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAE: {mae_val:.2f}")
print(f"R2: {r2_val:.2f}")

# TEST
y_test_pred_s, y_test_std = gpr.predict(X_test_s, return_std = True)
y_test_pred = scaler_y.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n--- Test (GPR) ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.2f}")

# PLOT
plt.figure(figsize = (12,5))

plt.plot(test["t"], y_test, 'k.', label = "Real")
plt.plot(test["t"], y_test_pred, 'b-', label = "Predicción")

plt.fill_between(
    test["t"].values,
    y_test_pred - 2*y_test_std,
    y_test_pred + 2*y_test_std,
    alpha = 0.2,
    label = "Incertidumbre (+-2σ)"
)

plt.legend()
plt.title("GPR estable - Spain")
plt.xlabel("t")
plt.ylabel("Cases")
plt.show()


# UNIR TODO
full = pd.concat([train, val, test]).sort_values(["year", "week"])

time = full["year"] + full["week"] / 52
X_full = full[features].values
X_full_s = scaler_X.transform(X_full)

y_full_pred_s, y_full_std = gpr.predict(X_full_s, return_std = True)
y_full_pred = scaler_y.inverse_transform(
    y_full_pred_s.reshape(-1, 1)
).ravel()

# PLOT GLOBAL
plt.figure(figsize = (14,6))

plt.plot(time, full["cases"].values, "k.", alpha = 0.4, label = "Real")
plt.plot(time, y_full_pred, "b-", label = "Predicción")

plt.fill_between(
    time.values,
    y_full_pred - 2 * y_full_std,
    y_full_pred + 2 * y_full_std,
    alpha = 0.2,
    label = "Incertidumbre (+-2σ)"
)

plt.title("Gaussian Process Regression - Influenza (todos los años)")
plt.xlabel("Year")
plt.ylabel("Cases")

years = np.arange(int(time.min()), int(time.max()) + 1, 1)
plt.xticks(years)

plt.grid(alpha = 0.3)
plt.legend()
plt.tight_layout()
plt.show()