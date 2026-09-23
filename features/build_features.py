import os
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.fetch_valet import get_fx_series

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw FX rate data for anomaly detection.

    Args:
        df: Raw DataFrame with columns: date, rate

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Daily return — how much the rate changed from the previous day
    df["daily_return"] = df["rate"].pct_change()

    # Rolling 21-day mean — average rate over the past 21 trading days
    df["rolling_mean"] = df["daily_return"].shift(1).rolling(window=21).mean()

    # Rolling 21-day std — volatility over the past 21 trading days
    df["rolling_std"] = df["daily_return"].shift(1).rolling(window=21).std()

    # Z-score — how unusual today's return is relative to recent behaviour
    df["z_score"] = (df["daily_return"] - df["rolling_mean"]) / df["rolling_std"]

    # Drop rows with NaN values (first 21 rows won't have rolling stats)
    df = df.dropna().reset_index(drop=True)

    # OLS Regression — expected return based on recent market conditions
    features = ["rolling_mean", "rolling_std", "z_score"]
    X = df[features]
    y = df["daily_return"]

    ols = LinearRegression()
    ols.fit(X, y)

    df["expected_return"] = ols.predict(X)

    # OLS Residual — how far actual return deviated from expected
    df["ols_residual"] = df["daily_return"] - df["expected_return"]

    return df


if __name__ == "__main__":
    raw_df = get_fx_series("FXUSDCAD", "2017-01-01", "2024-12-31")
    featured_df = build_features(raw_df)
    print(featured_df[["date", "daily_return", "expected_return", "ols_residual"]].head(10))
    print(f"\nShape: {featured_df.shape}")