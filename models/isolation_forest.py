import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import IsolationForest
from data_ingestion.fetch_valet import get_fx_series
from features.build_features import build_features

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Run Isolation Forest on engineered features to detect anomalous trading days.

    Args:
        df: Featured DataFrame from build_features()
        contamination: Expected proportion of anomalies in the dataset (default 2%)

    Returns:
        DataFrame with an additional 'anomaly' column (True = anomalous day)
    """
    df = df.copy()

    # Select features for the model
    features = ["daily_return", "rolling_mean", "rolling_std", "z_score"]
    X = df[features]

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X)

    # Predict — Isolation Forest returns 1 (normal) or -1 (anomaly)
    df["anomaly"] = model.predict(X)
    df["anomaly"] = df["anomaly"] == -1  # Convert to True/False

    # Continuous IF score
    if_scores = model.decision_function(X)

    # Flip and normalize IF score to 0-100 (more anomalous = higher score)
    flipped = -1 * if_scores
    if_normalized = (flipped - flipped.min()) / (flipped.max() - flipped.min()) * 100

    # Normalize OLS residual to 0-100 (larger absolute residual = higher score)
    abs_residual = df["ols_residual"].abs()
    residual_normalized = (abs_residual - abs_residual.min()) / (abs_residual.max() - abs_residual.min()) * 100

    # Final AAS — 65% OLS residual + 35% IF score
    df["aas_score"] = 0.65 * residual_normalized + 0.35 * if_normalized
    df["aas_score"] = df["aas_score"].round(2)

    return df


if __name__ == "__main__":
    raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
    featured_df = build_features(raw_df)
    result_df = detect_anomalies(featured_df)

    print(f"Total anomalies detected: {result_df['anomaly'].sum()}")
    print(f"\nTop 10 highest AAS scores:")
    print(result_df.nlargest(10, "aas_score")[["date", "rate", "z_score", "aas_score"]])