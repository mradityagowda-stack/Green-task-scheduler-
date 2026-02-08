import random
import pandas as pd

data = []

for _ in range(300):
    hour = random.randint(0, 23)
    day = random.randint(0, 6)
    renewable = random.randint(20, 90)
    load = round(random.uniform(0.3, 0.9), 2)

    carbon = 700 - (renewable * 4) + (load * 200)
    carbon = max(100, int(carbon))

    data.append([hour, day, renewable, load, carbon])

df = pd.DataFrame(data, columns=[
    "hour", "day", "renewable_pct", "grid_load", "carbon_intensity"
])

df.to_csv("carbon_data.csv", index=False)

print("Dataset created successfully!")
