import pandas as pd
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
    ["Home", "App", "Background", "Play"]
)

if section == "Home":
    page = "Home"
elif section == "App":
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

elif page == "Anomaly Explorer":
    st.title("Anomaly Explorer")
    st.markdown("Deep dive into individual flagged anomalies and explore the market context around each event. You can filter by date range and view detailed information about each anomaly.")

    if anomalies.empty:
        st.warning("No anomalies were detected for the current dataset.")
    else:
        anomaly_dates = anomalies["date"].dt.date.tolist()
        selected_date = st.selectbox("Select an anomalous date", anomaly_dates)

        window = st.slider("Context window (days before and after)", min_value=7, max_value=100, value=30)

        selected_dt = pd.Timestamp(selected_date)
        mask = (df["date"] >= selected_dt - pd.Timedelta(days=window)) & (df["date"] <= selected_dt + pd.Timedelta(days=window))
        window_df = df[mask]

        selected_row = df[df["date"] == selected_dt].iloc[0]
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rate", f"{selected_row['rate']:.4f}")
        col2.metric("Daily Return", f"{selected_row['daily_return']:.4f}")
        col3.metric("Z-Score", f"{selected_row['z_score']:.2f}")
        col4.metric("AAS Score", f"{selected_row['aas_score']:.1f}")

        st.markdown("---")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=window_df["date"], y=window_df["rate"],
            mode="lines", name="Rate",
            line=dict(color="steelblue", width=1.5)
        ))
        fig2.add_trace(go.Scatter(
            x=[selected_dt], y=[selected_row["rate"]],
            mode="markers", name="Selected Anomaly",
            marker=dict(color="red", size=10)
        ))
        fig2.update_layout(
            title=f"USD/CAD Rate Around {selected_date}",
            xaxis_title="Date",
            yaxis_title="Rate",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        st.subheader("Anomaly Context")

        aas = selected_row['aas_score']
        z = selected_row['z_score']

        if aas >= 80:
            severity = "Extremely anomalous. This day represents one of the most statistically unusual movements in the dataset."
        elif aas >= 50:
            severity = "Highly anomalous. Significant deviation from expected market behaviour detected."
        elif aas >= 30:
            severity = "Moderately anomalous. Notable deviation from expected market behaviour worth monitoring."
        else:
            severity = "Mildly anomalous. Flagged by the model but relatively subtle compared to other detected events."

        if z > 0:
            direction = "The USD strengthened against the CAD on this day, with the rate moving above what the model expected."
        else:
            direction = "The USD weakened against the CAD on this day, with the rate moving below what the model expected."

        st.info(f"{severity} {direction} Z-Score: {z:.2f} | AAS Score: {aas:.1f}")

elif page == "Currency Comparison":
    st.title("Currency Comparison")
    st.markdown("Compare USD/CAD against other major currency pairs to determine whether anomalies are USD/CAD specific or reflect broader CAD weakness.")

    pair_options = {
        "USD/CAD": "FXUSDCAD",
        "EUR/CAD": "FXEURCAD",
        "GBP/CAD": "FXGBPCAD",
        "AUD/CAD": "FXAUDCAD",
        "JPY/CAD": "FXJPYCAD"
    }

    selected_pairs = st.multiselect(
        "Select currency pairs to compare",
        list(pair_options.keys()),
        default=["USD/CAD", "EUR/CAD"]
    )

    if len(selected_pairs) < 2:
        st.warning("Please select at least two currency pairs for comparison.")
    else:
        comparison_dfs = []
        for pair in selected_pairs:
            symbol = pair_options[pair]
            raw_df = get_fx_series(symbol, "2017-01-01", "2024-12-31")
            featured_df = build_features(raw_df)
            result_df = detect_anomalies(featured_df)
            result_df["pair"] = pair
            comparison_dfs.append(result_df)

        combined_df = pd.concat(comparison_dfs, ignore_index=True)
        comparison_df = combined_df.rename(columns={"pair": "currency"}).copy()

        fig3 = go.Figure()
        for pair in selected_pairs:
            pair_data = combined_df[combined_df["pair"] == pair]
            fig3.add_trace(go.Scatter(
                x=pair_data["date"], y=pair_data["rate"],
                mode="lines", name=pair,
                line=dict(width=2)
            ))
        fig3.update_layout(
            title="Currency Pair Comparison (2017–2024)",
            xaxis_title="Date",
            yaxis_title="Rate",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Currency Statistics Comparison")

        stats_df = comparison_df.groupby("currency").agg(
            latest_rate=("rate", "last"),
            average_rate=("rate", "mean"),
            min_rate=("rate", "min"),
            max_rate=("rate", "max"),
            volatility=("daily_return", "std"),
            average_daily_return=("daily_return", "mean"),
            anomaly_count=("anomaly", "sum"),
            average_aas_score=("aas_score", "mean"),
            max_aas_score=("aas_score", "max"),
        ).reset_index()

        stats_df = stats_df.round(4)

        higher_is_better = [
            "latest_rate",
            "average_rate",
            "max_rate",
            "average_daily_return",
        ]

        lower_is_better = [
            "min_rate",
            "volatility",
            "anomaly_count",
            "average_aas_score",
            "max_aas_score",
        ]

        def color_rank_column(column):
            if column.name == "currency":
                return [""] * len(column)

            if column.name in higher_is_better:
                ranks = column.rank(method="min", ascending=False)
            elif column.name in lower_is_better:
                ranks = column.rank(method="min", ascending=True)
            else:
                return [""] * len(column)

            colors = {
                1: "background-color: #2ecc71; color: white",
                2: "background-color: #f39c12; color: white",
                3: "background-color: #f1c40f; color: black",
                4: "background-color: #9b59b6; color: white",
                5: "background-color: #e74c3c; color: white",
            }

            return [colors.get(int(rank), "") for rank in ranks]

        styled_stats_df = stats_df.style.apply(color_rank_column, axis=0)
        st.dataframe(styled_stats_df, use_container_width=True)

elif page == "Model Controls":
    st.title("Model Controls")
    st.markdown("Adjust the contamination parameter in the isolation forest model to see how anamolies change with updated charts and tables. The contamination parameter represents the expected proportion of anomalies in the dataset. A higher value results in more anomalies being flagged, while a lower value is more conservative.")
    st.markdown("...")

    contamination = st.slider(
        "Contamination Parameter (Expected Proportion of Anomalies)",
        min_value=0.01,
        max_value=0.1,
        value=0.02,
        step=0.001,
        format="%.3f",
    )

    #Re-run the anomaly detection with the new contamination parameter
    @st.cache_data(ttl=0)
    def load_data_custom(contamination):
        raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
        featured_df = build_features(raw_df)
        result_df = detect_anomalies(featured_df, contamination=contamination)
        return result_df

    custom_df = load_data_custom(contamination)
    custom_anomalies = custom_df[custom_df["anomaly"] == True]
    custom_normal = custom_df[custom_df["anomaly"] == False]

    st.markdown("---")

    col1, col2 = st.columns(2)
    col1.metric("Anomalies Detected", int(custom_anomalies.shape[0]))
    col2.metric("% of Trading Days Flagged", f"{(custom_anomalies.shape[0] / len(custom_df) * 100):.1f}%")

    # Main Chart with updated contamination parameter
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=custom_normal["date"], y=custom_normal["rate"],
        mode="lines", name="Normal",
        line=dict(color="steelblue", width=1)
    ))
    fig4.add_trace(go.Scatter(
        x=custom_anomalies["date"], y=custom_anomalies["rate"],
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=6)
    ))
    fig4.update_layout(
        title=f"USD/CAD Anomalies at Contamination = {contamination:.2f}",
        xaxis_title="Date",
        yaxis_title="Rate",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.info(f"At {contamination:.0%} contamination the model flags approximately {custom_anomalies.shape[0]} trading days as anomalous. Increase the threshold to catch more subtle anomalies or decrease it to focus only on the most extreme events.")