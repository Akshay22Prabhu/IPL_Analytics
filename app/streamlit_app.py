import streamlit as st
import pandas as pd
import joblib

from src.time_series import forecast_player
from src.rag_chatbot import generate_player_summary

st.set_page_config(page_title="IPL Analytics", layout="wide")

# Load
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed/final_dataset.csv")

players = sorted(df['batter'].unique())

st.title("🏏 IPL Analytics Platform")

tab1, tab2, tab3 = st.tabs(["🏏 Prediction", "📈 Forecast", "🤖 AI Insights"])

# ---------------- Prediction ----------------
with tab1:
    st.header("Run Prediction")

    col1, col2 = st.columns(2)

    with col1:
        batter = st.selectbox("Player", players)
        avg_last_5 = st.number_input("Avg Last 5")

    with col2:
        avg_last_10 = st.number_input("Avg Last 10")
        strike_rate = st.number_input("Strike Rate")

    if st.button("Predict Runs"):
        input_df = pd.DataFrame([{
            "batter": batter,
            "avg_last_5": avg_last_5,
            "avg_last_10": avg_last_10,
            "strike_rate": strike_rate,
            "match_number": 50,
            "consistency": 10,
            "form_trend": 0
        }])

        pred = model.predict(input_df)[0]
        st.metric("Predicted Runs", f"{pred:.2f}")

    # Graph
    player_data = df[df['batter'] == batter]

    st.subheader("📈 Performance Trend")
    st.line_chart(player_data.set_index('match_number')['runs'])

# ---------------- Forecast ----------------
with tab2:
    st.header("Next 10 Matches Forecast")

    player_ts = st.selectbox("Player", players, key="ts")

    if st.button("Forecast"):
        forecast = forecast_player(player_ts)
        st.line_chart(forecast.set_index('ds')['yhat'])

# ---------------- AI ----------------
with tab3:
    st.header("AI Cricket Analyst")

    player_ai = st.selectbox("Player", players, key="ai")

    if st.button("Analyze"):
        summary = generate_player_summary(player_ai)
        st.write(summary)