import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, DotProduct, RationalQuadratic
from sklearn.preprocessing import StandardScaler
import os

# Load and prepare data
data_dir = "data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
val = pd.read_csv(os.path.join(data_dir, "val.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Filter for Spain
country_col = "country_Spain"
train = train[train[country_col] == 1]
val = val[val[country_col] == 1]
test = test[test[country_col] == 1]
test = pd.concat([val, test], axis=0).reset_index(drop=True)

# Create time feature
def create_time(df):
    return (df["year"] - 2022) + (df["week"] / 52)
    

train["t"] = create_time(train)
#val["t"] = create_time(val)
test["t"] = create_time(test)

# Sort by time
train = train.sort_values("t")
#val = val.sort_values("t")
test = test.sort_values("t")

# Features
features = ["t"]
X_train = train[features].values
y_train = train["cases"].values
#X_val = val[features].values
#y_val = val["cases"].values
X_test = test[features].values
y_test = test["cases"].values

# Scale features
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
#X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

# Scale target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#y_val_s = scaler_y.transform(val["cases"].values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(test["cases"].values.reshape(-1, 1)).ravel()

## 
# Define the kernel
period = 1.0  # Annual periodicity

kernel = (
    ConstantKernel(1.0)* ExpSineSquared(length_scale=0.3, periodicity=period)
    #+ ConstantKernel(1.0) * RBF(length_scale=10.0, length_scale_bounds=(10.0, 20.0))
    + WhiteKernel(noise_level=1.0)
)
# Create and fit the GP model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(X_train_s, y_train_s)

# Predict on all datasets
y_train_pred_s, y_train_std_s = gp.predict(X_train_s, return_std=True)
#y_val_pred_s, y_val_std_s = gp.predict(X_val_s, return_std=True)
y_test_pred_s, y_test_std_s = gp.predict(X_test_s, return_std=True)

# Inverse transform predictions to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_s.reshape(-1, 1)).ravel()
#y_val_pred = scaler_y.inverse_transform(y_val_pred_s.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()

# Inverse transform standard deviations (approximate)
y_train_std = y_train_std_s * np.std(y_train) / np.std(y_train_s)
#y_val_std = y_val_std_s * np.std(y_train) / np.std(y_train_s)
y_test_std = y_test_std_s * np.std(y_train) / np.std(y_train_s)

def plot_dataset(ax, df, y_actual, y_pred, y_std, color, label):
    """Plot a single dataset (actual, predicted, and confidence interval)."""
    ax.scatter(df["t"], y_actual, color=color, alpha=0.5, s=10, label=f"{label} (Actual)")
    ax.plot(df["t"], y_pred, color=color, linestyle="-", linewidth=1, label=f"{label} (Predicted)")
    ax.fill_between(
        df["t"],
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color=color, alpha=0.1, label=f"{label} (95% CI)"
    )

def plot_results(train, test, y_train_pred, y_test_pred,
                 y_train_std, y_test_std):
    """Plot actual vs. predicted cases with uncertainty intervals for all datasets."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each dataset
    plot_dataset(ax, train, train["cases"].values, y_train_pred, y_train_std, "blue", "Train")
    #plot_dataset(ax, val, val["cases"].values, y_val_pred, y_val_std, "green", "Val")
    plot_dataset(ax, test, test["cases"].values, y_test_pred, y_test_std, "red", "Test")

    # Labels and legend
    ax.set_xlabel("Time (Years since 2022)")
    ax.set_ylabel("Influenza Cases")
    ax.set_title("Gaussian Process Model: Actual vs. Predicted Influenza Cases (Spain)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# Plot the results
print(gp.kernel_)
plot_results(
    train, test,
    y_train_pred, y_test_pred,
    y_train_std, y_test_std
)
