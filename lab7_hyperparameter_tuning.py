import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# ---------------- DATA GENERATION ----------------

np.random.seed(42)

hours = np.random.randint(0,24,500)
days = np.random.randint(0,7,500)
renewable = np.random.uniform(30,90,500)
load = np.random.uniform(0.3,0.9,500)

X = np.column_stack([hours, days, renewable, load])

y = 900 - renewable*4 + load*220 + hours*2

# ---------------- MODEL ----------------

model = RandomForestRegressor()

# ---------------- PARAM GRID ----------------

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# ---------------- GRID SEARCH ----------------

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X, y)

best_model = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)

# ---------------- PERFORMANCE ----------------

pred = best_model.predict(X)
print("R2 Score:", round(r2_score(y, pred),3))

# ---------------- SAVE MODEL ----------------

joblib.dump(best_model, "carbon_model.pkl")

print("Optimized model saved as carbon_model.pkl")
