import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load data ---
data = pd.read_csv("../data/egfr_rdkit_features.csv")

# --- Prepare features and target ---
target_col = "pIC50"
exclude_cols = ['Unnamed: 0', 'molecule_chembl_id', 'smiles', 'class']
feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]

X = data[feature_cols]
y = data[target_col]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest Regressor ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# --- Evaluation ---
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

print("✅ Random Forest Regressor Performance:")
print(f"R² (Train): {r2_train:.3f}")
print(f"R² (Test) : {r2_test:.3f}")
print(f"RMSE      : {rmse:.3f}")
print(f"MAE       : {mae:.3f}")

# --- Save plot: Actual vs Predicted ---
os.makedirs("../results", exist_ok=True)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
plt.title("Random Forest: Actual vs Predicted pIC50")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/rf_actual_vs_predicted.png", dpi=300)
print("📁 Plot saved to: ../outputs/rf_actual_vs_predicted.png")
