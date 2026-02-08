import joblib
import pandas as pd
import random

model = joblib.load("carbon_model.pkl")

future_hours = []

for hour in range(24):
    day = 2
    renewable = random.randint(30, 90)
    load = round(random.uniform(0.3, 0.9), 2)

    future_hours.append([hour, day, renewable, load])

df = pd.DataFrame(
    future_hours,
    columns=["hour","day","renewable_pct","grid_load"]
)

predictions = model.predict(df)

for row, val in zip(df.values, predictions):
    print(row, "→", int(val))
