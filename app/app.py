import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.fetch_valet import get_fx_series
from features.build_features import build_features
from models.isolation_forest import detect_anomalies
import plotly.graph_objects as go

st.set_page_config(
    page_title="FX Anomaly Detector",
    page_icon="🇨🇦",
    layout="wide"
)

@st.cache_data
def load_data():
    raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
    featured_df = build_features(raw_df)
    print("build_features columns:", featured_df.columns.tolist())
    result_df = detect_anomalies(featured_df)
    print("detect_anomalies columns:", result_df.columns.tolist())
    return result_df

st.cache_data.clear()
df = load_data()
required_columns = {"date", "rate", "anomaly", "z_score", "aas_score"}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    st.error(f"Data pipeline is missing required columns: {missing_columns}.")
    st.stop()

anomalies = df[df["anomaly"] == True]
normal = df[df["anomaly"] == False]


#Sidebar navigation
st.sidebar.title("FX Anomaly Detector")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate",
    ["App", "Background", "Play"]
)

if section == "App":
    page = st.sidebar.selectbox(
        "Select Tab",
        ["Dashboard", "Anomaly Explorer", "Currency Comparison", "Model Controls", "Anomaly Finder", "Anomaly Report"]
    )
elif section == "Background":
    page = st.sidebar.selectbox(
        "Select Tab",
        ["EDA", "Methodology", "Data Source", "Limitations"]
    )
else:
    page = "Quiz"

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Akash | Bank of Canada Valet API")

if page == "Dashboard": 
    st.title("Dashboard")
    st.markdown("This dashboard provides an overview of the FX anomaly detection results for the USD/CAD currency pair from 2017 to 2024. The anomalies are detected using a combination of statistical features and machine learning techniques.")

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trading Days", len(df))
    col2.metric("Anomalies Detected", int(df["anomaly"].sum()))
    col3.metric("Highest AAS Score", f"{df['aas_score'].max():.1f}")
    col4.metric("Most Anomalous Date", str(df.loc[df["aas_score"].idxmax(), "date"].date()))

    st.markdown("---")

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["rate"],
        mode="lines", name="Normal",
        line=dict(color="steelblue", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["date"], y=anomalies["rate"],
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=6),
        hovertemplate="<b>%{x}</b><br>Rate: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title="USD/CAD Exchange Rate (2017–2024)",
        xaxis_title="Date",
        yaxis_title="Rate",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Flagged Anomalies Table
    st.subheader("Flagged Anomalies")
    display_df = anomalies[["date", "rate", "z_score", "aas_score"]].copy()
    display_df["date"] = display_df["date"].dt.date
    display_df = display_df.sort_values("aas_score", ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)
