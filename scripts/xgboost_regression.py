import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
data = pd.read_csv(f"../data/egfr_rdkit_features.csv")

# --- Define target and features ---
target_col = "pIC50"
exclude_cols = ['Unnamed: 0', 'molecule_chembl_id', 'smiles', 'class']
#exclude_cols = ['Unnamed: 0', 'molecule_chembl_id', 'smiles', 'class', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]

X = data[feature_cols]
y = data[target_col]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Regressor ---
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# --- Predictions ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --- Evaluation ---
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("✅ XGBoost Regressor Performance:")
print(f"R² Train = {r2_train:.3f}")
print(f"R² Test  = {r2_test:.3f}")
print(f"RMSE     = {rmse_test:.3f}")
print(f"MAE      = {mae_test:.3f}")

# --- Save plot: Actual vs Predicted (Test) ---
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')

plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
plt.title("XGBoost: Actual vs Predicted pIC50 (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/xgb_actual_vs_predicted.png", dpi=300)
print("📁 Plot saved to: xgb_actual_vs_predicted.png")

