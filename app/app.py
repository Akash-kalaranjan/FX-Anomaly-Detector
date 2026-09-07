import pandas as pd
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.fetch_valet import get_fx_series
from features.build_features import build_features
from models.isolation_forest import detect_anomalies
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import base64
from datetime import date
from io import BytesIO
import random


def get_video_base64(video_path):
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache_data
def get_animated_banner(rates, anomaly_indices):
    """Build a small animated GIF so the hero always renders as moving media."""
    width, height = 1280, 520
    frames = []
    minimum = min(rates)
    maximum = max(rates)
    span = maximum - minimum or 1
    anomaly_indices = set(anomaly_indices)

    for frame_number in range(12):
        frame = Image.new("RGB", (width, height), (13, 24, 38))
        pixels = frame.load()
        for x in range(width):
            shade = int(18 + (x / width) * 22)
            for y in range(height):
                pixels[x, y] = (10, shade, 38 + int(y / height * 18))

        draw = ImageDraw.Draw(frame)

        points = []
        for index, rate in enumerate(rates):
            x = 48 + index * (width - 96) / max(len(rates) - 1, 1)
            y = 410 - ((rate - minimum) / span) * 250
            points.append((x, y))
        draw.line(points, fill=(88, 211, 190), width=5)
        draw.line([(48, 420), (width - 48, 420)], fill=(66, 91, 112), width=2)

        visible_points = points[: max(2, int(len(points) * (frame_number + 2) / 13))]
        draw.line(visible_points, fill=(255, 187, 92), width=7)
        for index, point in enumerate(points):
            if index in anomaly_indices and frame_number % 4 < 2:
                draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9), fill=(247, 102, 93))
        frames.append(frame)

    output = BytesIO()
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=130, loop=0)
    return base64.b64encode(output.getvalue()).decode()

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

def build_mini_zoom_chart(df, selected_date, window_days=15):
    selected_index = df.index[df["date"] == selected_date][0]
    start_index = max(0, selected_index - window_days)
    end_index = min(len(df), selected_index + window_days + 1)

    zoom_df = df.iloc[start_index:end_index]
    zoom_anomalies = zoom_df[zoom_df["anomaly"]]
    selected_row = df.iloc[selected_index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zoom_df["date"],
        y=zoom_df["rate"],
        mode="lines+markers",
        name="USD/CAD",
        line=dict(color="steelblue", width=2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=zoom_anomalies["date"],
        y=zoom_anomalies["rate"],
        mode="markers",
        name="Flagged anomaly",
        marker=dict(color="red", size=9, symbol="diamond")
    ))
    fig.add_vline(
        x=selected_row["date"],
        line_color="black",
        line_dash="dash",
        line_width=2
    )
    fig.update_layout(
        title=f"Mini Zoom: {selected_row['date'].date()}",
        xaxis_title="Date",
        yaxis_title="Rate",
        hovermode="x unified",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig

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
        ["Dashboard", "Anomaly Explorer", "Currency Comparison", "Model Controls", "Anomaly Finder", "Normal Finder", "Anomaly Report"]
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

if page == "Home":
    st.title("FX Anomaly Home")
    st.markdown("### Detecting statistically unusual USD/CAD trading days using OLS Regression and Isolation Forest.")
    st.markdown("...")

    banner_df = df.tail(90).reset_index(drop=True)
    banner_gif = get_animated_banner(
        tuple(banner_df["rate"].round(5)),
        tuple(banner_df.index[banner_df["anomaly"]]),
    )

    st.markdown(
        f"""
        <style>
        .brand-shell {{
            position: relative;
            width: 100%;
            height: 520px;
            overflow: hidden;
            border-radius: 12px;
        }}
        .hero-gif {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
        }}
        .hero-overlay {{
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
        }}
        .brand-content {{
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2rem;
        }}
        .brand-name {{
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
        }}
        .brand-copy {{
            font-size: 1.1rem;
            color: rgba(255,255,255,0.85);
            max-width: 600px;
            margin-bottom: 1.5rem;
        }}
        .link-row {{
            display: flex;
            gap: 1rem;
        }}
        .link-pill {{
            padding: 0.5rem 1.5rem;
            border-radius: 999px;
            background: white;
            color: #1a1a2e;
            font-weight: 600;
            text-decoration: none;
            font-size: 0.95rem;
        }}
        .link-pill:hover {{
            background: #e0e0e0;
        }}
        </style>

        <section class="brand-shell">
            <img class="hero-gif" src="data:image/gif;base64,{banner_gif}" alt="Animated USD/CAD anomaly chart">
            <div class="hero-overlay"></div>
            <div class="brand-content">
                <h1 class="brand-name">FX Anomaly Detector</h1>
                <p class="brand-copy">
                    Detecting statistically unusual USD/CAD trading days using OLS Regression and Isolation Forest.
                    Built on public Bank of Canada data.
                </p>
                <div class="link-row">
                    <a class="link-pill" href="https://www.linkedin.com/in/akash-kalaranjan-9aa895255/" target="_blank">LinkedIn</a>
                    <a class="link-pill" href="https://github.com/Akash-kalaranjan/FX-Anomaly-Detector" target="_blank">GitHub</a>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trading Days Analyzed", "1,974")
    col2.metric("Anomalies Detected", "40")
    col3.metric("Date Range", "2017-2024")
    col4.metric("Highest AAS Score", f"{df['aas_score'].max():.1f}")

    st.markdown("---")

    st.subheader("Daily Anomaly Comparison")
    st.caption(f"A new deterministic comparison is selected each day: {date.today().isoformat()}")

    daily_rng = random.Random(date.today().isoformat())
    selected_anomaly = anomalies.iloc[daily_rng.randrange(len(anomalies))]
    selected_normal = normal.iloc[daily_rng.randrange(len(normal))]
    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.markdown("#### Flagged anomaly")
        st.metric("Trading day", str(selected_anomaly["date"].date()))
        st.metric("USD/CAD rate", f"{selected_anomaly['rate']:.4f}")
        st.write(f"AAS score: **{selected_anomaly['aas_score']:.1f}**  ")
        st.write(f"Z-score: **{selected_anomaly['z_score']:.2f}**")
        st.plotly_chart(build_mini_zoom_chart(df, selected_anomaly["date"]), use_container_width=True)

    with comparison_col2:
        st.markdown("#### Typical trading day")
        st.metric("Trading day", str(selected_normal["date"].date()))
        st.metric("USD/CAD rate", f"{selected_normal['rate']:.4f}")
        st.write(f"AAS score: **{selected_normal['aas_score']:.1f}**  ")
        st.write(f"Z-score: **{selected_normal['z_score']:.2f}**")
        st.plotly_chart(build_mini_zoom_chart(df, selected_normal["date"]), use_container_width=True)

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

elif page == "Anomaly Finder":
    st.title("Anomaly Finder")
    st.markdown(
        "Search for anomalies based on criteria such as year, direction, and AAS score. "
        "This tool helps you quickly identify and analyze the most anomalous trading days."
    )

    @st.cache_data
    def load_data():
        raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
        featured_df = build_features(raw_df)
        result_df = detect_anomalies(featured_df)
        return result_df

    def add_anomaly_finder_fields(df):
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["direction"] = df["daily_return"].apply(
            lambda value: "USD strengthened" if value > 0 else "USD weakened"
        )
        return df

    df = load_data()
    df = add_anomaly_finder_fields(df)

    anomalies = df[df["anomaly"]].copy()
    ranked_anomalies = anomalies.sort_values("aas_score", ascending=False).reset_index(drop=True)

    year_col, direction_col, threshold_col, zoom_col = st.columns([1, 1.2, 1.2, 1])

    with year_col:
        year_options = ["All years"] + [
            str(year) for year in sorted(ranked_anomalies["year"].unique())
        ]
        selected_year = st.selectbox("Year", year_options)

    with direction_col:
        selected_direction = st.selectbox(
            "Direction",
            ["All directions", "USD strengthened", "USD weakened"]
        )

    with threshold_col:
        selected_threshold = st.slider(
            "Minimum AAS",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5
        )

    with zoom_col:
        zoom_window = st.slider(
            "Zoom window",
            min_value=5,
            max_value=45,
            value=15,
            step=5
        )

    filtered = ranked_anomalies[ranked_anomalies["aas_score"] >= selected_threshold]

    if selected_year != "All years":
        filtered = filtered[filtered["year"] == int(selected_year)]

    if selected_direction != "All directions":
        filtered = filtered[filtered["direction"] == selected_direction]

    display_df = filtered.copy()
    display_df["rank"] = range(1, len(display_df) + 1)
    display_df["date_display"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["daily_return_pct"] = (display_df["daily_return"] * 100).round(2)

    st.caption(
        f"Showing {len(display_df)} of {len(ranked_anomalies)} flagged days, ranked by AAS score."
    )

    if display_df.empty:
        st.info("No flagged days match the current filters.")
    else:
        table_df = display_df[[
            "rank",
            "date_display",
            "rate",
            "daily_return_pct",
            "z_score",
            "aas_score",
            "direction"
        ]].rename(columns={
            "rank": "Rank",
            "date_display": "Date",
            "rate": "Rate",
            "daily_return_pct": "Daily return %",
            "z_score": "Z-score",
            "aas_score": "AAS",
            "direction": "Direction"
        })

        selected = st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )

        selected_rows = selected.selection.rows

        if selected_rows:
            selected_row = display_df.iloc[selected_rows[0]]

            metric_col, return_col, z_col = st.columns(3)
            metric_col.metric("AAS score", f"{selected_row['aas_score']:.2f}")
            return_col.metric("Daily return", f"{selected_row['daily_return'] * 100:.2f}%")
            z_col.metric("Z-score", f"{selected_row['z_score']:.2f}")

            st.plotly_chart(
                build_mini_zoom_chart(df, selected_row["date"], zoom_window),
                use_container_width=True
            )
        else:
            st.info("Click a row to open its mini zoom chart.")

elif page == "Normal Finder":
    st.title("Normal Finder")
    st.markdown(
        "Search for normal (non-anomalous) trading days based on criteria such as year, direction, and AAS score. "
        "This tool helps you quickly identify and analyze the most typical trading days."
    )

    @st.cache_data
    def load_data():
        raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
        featured_df = build_features(raw_df)
        result_df = detect_anomalies(featured_df)
        return result_df

    def add_normal_finder_fields(df):
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["direction"] = df["daily_return"].apply(
            lambda value: "USD strengthened" if value > 0 else "USD weakened"
        )
        return df

    df = load_data()
    df = add_normal_finder_fields(df)

    normal_days = df[~df["anomaly"]].copy()
    ranked_normals = normal_days.sort_values("aas_score", ascending=True).reset_index(drop=True)

    year_col, direction_col, threshold_col, zoom_col = st.columns([1, 1.2, 1.2, 1])

    with year_col:
        year_options = ["All years"] + [
            str(year) for year in sorted(ranked_normals["year"].unique())
        ]
        selected_year = st.selectbox("Year", year_options)

    with direction_col:
        selected_direction = st.selectbox(
            "Direction",
            ["All directions", "USD strengthened", "USD weakened"]
        )

    with threshold_col:
        selected_threshold = st.slider(
            "Maximum AAS",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5
        )

    with zoom_col:
        zoom_window = st.slider(
            "Zoom window",
            min_value=5,
            max_value=45,
            value=15,
            step=5
        )

    filtered_normals = ranked_normals[ranked_normals["aas_score"] <= selected_threshold]

    if selected_year != "All years":
        filtered_normals = filtered_normals[
            filtered_normals["year"] == int(selected_year)
        ]

    if selected_direction != "All directions":
        filtered_normals = filtered_normals[
            filtered_normals["direction"] == selected_direction
        ]

    display_df = filtered_normals.copy()
    display_df["rank"] = range(1, len(display_df) + 1)
    display_df["date_display"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["daily_return_pct"] = (display_df["daily_return"] * 100).round(2)

    st.caption(
        f"Showing {len(display_df)} of {len(ranked_normals)} normal days, ranked from lowest to highest AAS score."
    )

    if display_df.empty:
        st.info("No normal days match the current filters.")
    else:
        table_df = display_df[[
            "rank",
            "date_display",
            "rate",
            "daily_return_pct",
            "z_score",
            "aas_score",
            "direction"
        ]].rename(columns={
            "rank": "Rank",
            "date_display": "Date",
            "rate": "Rate",
            "daily_return_pct": "Daily return %",
            "z_score": "Z-score",
            "aas_score": "AAS",
            "direction": "Direction"
        })

        selected_label = st.selectbox(
            "Select a normal day",
            display_df["date_display"].tolist(),
            format_func=lambda value: (
                f"{value} - AAS "
                f"{display_df.loc[display_df['date_display'] == value, 'aas_score'].iloc[0]:.2f}"
            )
        )

        selected_row = display_df[display_df["date_display"] == selected_label].iloc[0]

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

        metric_col, return_col, z_col = st.columns(3)
        metric_col.metric("AAS score", f"{selected_row['aas_score']:.2f}")
        return_col.metric("Daily return", f"{selected_row['daily_return'] * 100:.2f}%")
        z_col.metric("Z-score", f"{selected_row['z_score']:.2f}")

        st.plotly_chart(
            build_mini_zoom_chart(df, selected_row["date"], zoom_window),
            use_container_width=True
        )

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("The same exploratory analysis shown in `notebooks/eda.ipynb`, rendered directly in the app.")

    eda_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31").copy()
    raw_preview = eda_df.head(10)
    raw_missing_values = int(eda_df.isnull().sum().sum())
    eda_df["daily_return"] = eda_df["rate"].pct_change()
    eda_df["rolling_std"] = eda_df["daily_return"].rolling(window=21).std()

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Trading days", f"{len(eda_df):,}")
    summary_col2.metric("Raw missing values", f"{raw_missing_values:,}")
    summary_col3.metric("Minimum rate", f"{eda_df['rate'].min():.4f}")
    summary_col4.metric("Maximum rate", f"{eda_df['rate'].max():.4f}")

    st.subheader("Raw Data")
    st.dataframe(raw_preview, use_container_width=True, hide_index=True)

    rate_fig = go.Figure(go.Scatter(
        x=eda_df["date"], y=eda_df["rate"], mode="lines",
        line=dict(color="steelblue", width=1), name="USD/CAD"
    ))
    rate_fig.update_layout(
        title="USD/CAD Exchange Rate (2017-2024)",
        xaxis_title="Date", yaxis_title="Rate", height=400
    )
    st.plotly_chart(rate_fig, use_container_width=True)

    return_fig = go.Figure(go.Scatter(
        x=eda_df["date"], y=eda_df["daily_return"], mode="lines",
        line=dict(color="steelblue", width=0.8), name="Daily return"
    ))
    return_fig.add_hline(y=0, line_color="red", line_dash="dash", line_width=1)
    return_fig.update_layout(
        title="USD/CAD Daily Returns (2017-2024)",
        xaxis_title="Date", yaxis_title="Daily Return", height=400
    )
    st.plotly_chart(return_fig, use_container_width=True)

    distribution_fig = go.Figure(go.Histogram(
        x=eda_df["daily_return"].dropna(), nbinsx=100,
        marker_color="steelblue", name="Daily returns"
    ))
    distribution_fig.add_vline(
        x=eda_df["daily_return"].mean(), line_color="red", line_dash="dash",
        annotation_text="Mean"
    )
    distribution_fig.add_vline(
        x=eda_df["daily_return"].std(), line_color="orange", line_dash="dash",
        annotation_text="+1 Std"
    )
    distribution_fig.add_vline(
        x=-eda_df["daily_return"].std(), line_color="orange", line_dash="dash",
        annotation_text="-1 Std"
    )
    distribution_fig.update_layout(
        title="Distribution of USD/CAD Daily Returns (2017-2024)",
        xaxis_title="Daily Return", yaxis_title="Frequency", height=400
    )
    st.plotly_chart(distribution_fig, use_container_width=True)

    volatility_fig = go.Figure(go.Scatter(
        x=eda_df["date"], y=eda_df["rolling_std"], mode="lines",
        line=dict(color="steelblue", width=1), name="21-day rolling volatility"
    ))
    volatility_fig.update_layout(
        title="21-Day Rolling Volatility (2017-2024)",
        xaxis_title="Date", yaxis_title="Volatility", height=400
    )
    st.plotly_chart(volatility_fig, use_container_width=True)

    st.subheader("EDA Summary")
    st.markdown("""
    **Dataset:** USD/CAD daily exchange rates, 2017–2024 (1,995 trading days, 0 missing values)

    **1. Raw Rate Over Time:**
    The USD/CAD exchange rate ranged between 1.20 and 1.45 over the 2017–2024 period. For the most part the rate moves gradually with minimal day-to-day swings, reflecting normal market conditions. The most prominent anomaly occurs in early 2020, where the rate spiked sharply to 1.45 during the COVID-19 market panic as investors fled to the US dollar as a safe haven. This is because COVID-19 created widespread economic uncertainty, such as businesses shutting down overnight, supply chains collapsing, and unemployment rates spiking globally. In these situations investors tend to flee to safe-haven assets, and the USD is consistently ranked as the world's primary safe-haven currency. Simultaneously, oil prices collapsed which severely weakened the CAD, as Canada's economy is heavily tied to commodity exports. The combination of surging USD demand and a weakening CAD drove the spike visible in March 2020. Approximately one year later the rate hit its lowest point of around 1.20, as markets recovered and commodity prices rebounded, strengthening the Canadian dollar.

    **2. Daily Returns:**
    The day-to-day change in the USD/CAD exchange rate is very marginal, with returns tightly clustered around 0 as expected. Most trading days see minimal movement, reflecting normal market conditions. This as expected by nature as in most days, trading days are uneventful, with only minor fluctuations in the exchange rate. Large moves only occur when significant events take place such as economic shocks, surprise interest rate decisions, or global crises. However, a small number of days stand out with extreme spikes in both directions, most notably in March 2020 during COVID-19, where the largest single-day moves in the entire dataset are clearly visible as expected given that Covid-19 was a global pandemic. A handful of other anomalous days are scattered across 2017–2024, suggesting isolated periods of unusual market activity worth flagging for further review.

    **3. Distribution of USD/CAD Daily Returns:**
    The distribution of daily returns ranges approximately from -0.012 to 0.012, with the vast majority of trading days showing minimal rate changes. The mode sits at approximately 0, with a slight negative skew, suggesting marginally more days with small negative returns than positive ones. The slight negative skew can be concluded as just statistical noise given how marginal they are over an 8 year period. The tails of the distribution are thin, emphasizing how rarely large moves occur over the 8 year period. The extreme value on the far right is consistent with the COVID-19 shock of March 2020, and this chart makes clear just how anomalous that event was relative to normal market behaviour. Overall the distribution averages close to 0, consistent with the random walk hypothesis which suggests currency returns have no consistent directional bias over time.

    ***4. Rolling Volatility:**
    Volatility remained relatively low and stable for the majority of the 2017–2024 period, reflecting normal calm market conditions. The most striking feature is the explosion in volatility during early 2020, peaking at nearly 3x the baseline level as COVID-19 triggered sustained panic across global markets. The biggest piece of information this plot offers is that this was not just a single anomalous day but a prolonged period of turbulence lasting several weeks. Notably, volatility was at its lowest point immediately before the COVID spike in late 2019, highlighting how abruptly conditions changed. Post-2022 shows a period of persistently elevated volatility compared to pre-COVID norms, consistent with the Bank of Canada aggressively hiking interest rates to combat inflation. This was because to in 2022, the Bank of Canada aggressively raised interest rates to fight inflation, going from 0.25% all the way to 5% in about 18 months. Every rate decision created uncertainty and market movement.

    **Conclusion:**
    Overall the USD/CAD exchange rate exhibits stable, predictable behaviour across the 2017–2024 period, with daily changes remaining marginal under normal market conditions. However all four plots consistently highlight how dramatically this changes during periods of genuine economic shock. The COVID-19 pandemic stands out as the most significant anomaly in the dataset, visible across every chart as a clear statistical outlier relative to the 8 year baseline. This confirms that the data is well suited for unsupervised anomaly detection as normal behaviour is stable and well defined, making true anomalies statistically distinct and identifiable.
    """)