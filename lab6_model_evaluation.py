import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("carbon_model.pkl")

# Generate test data (same format as training)
np.random.seed(42)

hours = np.random.randint(0,24,200)
days = np.random.randint(0,7,200)
renewable = np.random.uniform(30,90,200)
load = np.random.uniform(0.3,0.9,200)

X_test = np.column_stack([hours, days, renewable, load])

# Simulated true carbon values (ground truth)
y_true = 900 - renewable*4 + load*220 + hours*2

# Model prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("Model Evaluation Metrics")
print("------------------------")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,3))

# Plot
plt.scatter(y_true, y_pred)
plt.xlabel("True Carbon")
plt.ylabel("Predicted Carbon")
plt.title("Model Prediction vs True Values")
plt.show()
