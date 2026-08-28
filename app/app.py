import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.fetch_valet import get_fx_series
from features.build_features import build_features
from models.isolation_forest import detect_anomalies
import plotly.graph_objects as go

st.set_page_config(page_title="FX Anomaly Detector", layout="wide")

@st.cache_data
def load_data():
    raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
    featured_df = build_features(raw_df)
    result_df = detect_anomalies(featured_df)
    return result_df

df = load_data()

anomalies = df[df["anomaly"] == True]
normal = df[df["anomaly"] == False]

st.title("🇨🇦 FX Anomaly Detector")
st.markdown("Detecting statistically unusual USD/CAD exchange rate movements using Isolation Forest.")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=normal["date"], y=normal["rate"],
    mode="lines", name="Normal", line=dict(color="steelblue", width=1)
))

fig.add_trace(go.Scatter(
    x=anomalies["date"], y=anomalies["rate"],
    mode="markers", name="Anomaly",
    marker=dict(color="red", size=6)
))

fig.update_layout(
    title="USD/CAD Exchange Rate (2017–2024)",
    xaxis_title="Date",
    yaxis_title="Rate",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Flagged Anomalies")
st.dataframe(anomalies[["date", "rate", "z_score"]].reset_index(drop=True))