import joblib
import pandas as pd
import random

# Load ML model
model = joblib.load("carbon_model.pkl")

# ---------------- USER INPUT ----------------

task_name = input("Enter task: ")
duration = int(input("Enter duration (hours): "))
task_type = input("Task type (light/medium/heavy): ").lower()

# Task weights
weights = {
    "light": 1,
    "medium": 1.5,
    "heavy": 2
}

task_weight = weights.get(task_type, 1.5)

# ---------------- PREDICT 24H CARBON ----------------

future = []

for hour in range(24):
    renewable = random.randint(30, 90)
    load = round(random.uniform(0.3, 0.9), 2)
    future.append([hour, 2, renewable, load])

df = pd.DataFrame(
    future,
    columns=["hour","day","renewable_pct","grid_load"]
)

carbon_pred = model.predict(df)

# ---------------- FIND BEST WINDOW ----------------

best_hour = 0
best_score = float("inf")

for start in range(24 - duration + 1):
    window = carbon_pred[start:start+duration]
    score = sum(window) * task_weight
    
    if score < best_score:
        best_score = score
        best_hour = start

# ---------------- RESULT ----------------

print("\nTask:", task_name)
print("Best start hour:", best_hour)
print("Estimated carbon score:", int(best_score))
