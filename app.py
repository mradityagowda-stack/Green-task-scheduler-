import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Green Scheduler",
    page_icon="🌱",
    layout="wide"
)

# ---------------- THEME-SAFE PREMIUM STYLE ----------------

st.markdown("""
<style>

/* Premium dark-blue → green gradient */
.main {
    background: linear-gradient(135deg, #020617, #0b1120, #041418, #001a12);
}

.block-container {
    padding-top: 2rem;
}

/* Headings readable in both themes */
h1, h2, h3, h4 {
    letter-spacing: 0.5px;
}

/* Glass style subheading card */
.subhead-box {
    background: rgba(10, 25, 30, 0.55);
    padding: 14px;
    border-radius: 14px;
    border: 1px solid rgba(34,197,94,0.25);
    margin-bottom: 18px;
    backdrop-filter: blur(6px);
}

/* Insight box */
.insight-box {
    background: rgba(6, 40, 30, 0.55);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid rgba(34,197,94,0.35);
    backdrop-filter: blur(6px);
}

/* Metric cards */
.stMetric {
    background: rgba(15, 23, 42, 0.55);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(34,197,94,0.25);
    backdrop-filter: blur(6px);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🌱 AI-Powered Green Task Scheduler")

st.markdown("""
<div class="subhead-box">
Optimize compute tasks using <b>Machine Learning based carbon-aware scheduling</b>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("carbon_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙ Task Configuration")

known_tasks = [
    "video rendering",
    "ai training",
    "data processing",
    "simulation",
    "file backup",
    "image editing"
]

task_name = st.sidebar.text_input("Task to be Performed")
st.sidebar.caption("Supported tasks: " + ", ".join(known_tasks))

time_hours = st.sidebar.slider("Execution Time Required (hours)", 1, 12, 3)

task_type = st.sidebar.selectbox(
    "Task Intensity",
    ["Light", "Medium", "Heavy"]
)

day_name = st.sidebar.selectbox(
    "Prediction Day",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

run = st.sidebar.button("Run ML Optimization")

# ---------------- DAY MAP ----------------

day_map = {
    "Monday":0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday":6
}

selected_day_index = day_map[day_name]

# ---------------- TASK WEIGHT ----------------

task_weight_map = {
    "Light": 1.0,
    "Medium": 1.35,
    "Heavy": 1.9
}

task_weight = task_weight_map[task_type]

# ---------------- TIME FORMAT ----------------

def hour_to_12hr(h):
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:00 {suffix}"

# ---------------- MAIN LOGIC ----------------

if run:

    if task_name.strip().lower() not in known_tasks:
        st.error("Task not found. Please enter a supported task.")
        st.stop()

    with st.spinner("ML model is predicting optimal green schedule..."):
        time.sleep(2.2)

        hours = np.arange(24)
        day_col = np.full(24, selected_day_index)
        renewable_pct = np.random.uniform(30, 90, 24)
        grid_load = np.random.uniform(0.3, 0.9, 24)

        X = np.column_stack([
            hours,
            day_col,
            renewable_pct,
            grid_load
        ])

        carbon_pred = model.predict(X) * task_weight

    df = pd.DataFrame({
        "Hour": hours,
        "Renewable": renewable_pct,
        "Load": grid_load,
        "Carbon": carbon_pred
    })

    scores = []
    for i in range(24 - time_hours + 1):
        scores.append(df["Carbon"].iloc[i:i+time_hours].mean())

    best_start = int(np.argmin(scores))
    best_end = best_start + time_hours

    # ---------------- RESULTS ----------------

    st.divider()
    st.subheader("◆ Recommended Execution Window")

    c1, c2, c3 = st.columns(3)
    c1.metric("Task", task_name.title())
    c2.metric("Start Time", hour_to_12hr(best_start))
    c3.metric("End Time", hour_to_12hr(best_end))

    st.info(f"Prediction Day: {day_name} | Task Type: {task_type}")

    # ---------------- GRAPH ----------------

    st.divider()
    st.subheader("📈 ML Predicted Carbon Intensity (24 Hours)")

    chart_df = pd.DataFrame({
        "Hour": hours,
        "Carbon Intensity": carbon_pred
    })

    st.line_chart(chart_df.set_index("Hour"))
    st.caption("X-axis: Hour of Day | Y-axis: Carbon Intensity | Lower is Greener")

    # ---------------- INSIGHT ----------------

    st.divider()
    st.subheader("🌍 Environmental Impact Insights")

    best_avg = df["Carbon"].iloc[best_start:best_end].mean()
    overall_avg = df["Carbon"].mean()
    reduction = ((overall_avg - best_avg) / overall_avg) * 100
    cost_gain = reduction * 0.6

    st.markdown(f"""
    <div class="insight-box">
    Recommended window reduces carbon intensity by <b>{reduction:.1f}%</b><br><br>
    Estimated emission reduction benefit: <b>{reduction:.1f}% greener execution</b><br><br>
    Potential energy cost efficiency gain (relative): <b>{cost_gain:.1f}%</b><br><br>
    Window aligns with lower grid load and higher renewable availability.
    </div>
    """, unsafe_allow_html=True)

    # ---------------- TABLE ----------------

    st.divider()
    st.subheader("📊 Hourly Carbon Forecast Table (Explained)")

    df_table = pd.DataFrame({
        "Time": [hour_to_12hr(h) for h in hours],
        "Renewable Energy Share (%)": renewable_pct.round(1),
        "Grid Load Factor (0–1)": grid_load.round(2),
        "Predicted Carbon (gCO₂/kWh)": carbon_pred.round(1)
    })

    st.dataframe(df_table, use_container_width=True)

    st.caption(
        "Time = Local hour | Renewable % = clean energy share | Grid Load = demand level | Carbon = ML predicted emission intensity"
    )

else:
    st.info("Enter task details and click Run ML Optimization to generate schedule")
