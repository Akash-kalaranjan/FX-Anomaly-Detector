import requests
import pandas as pd

BASE_URL = "https://www.bankofcanada.ca/valet"

def get_fx_series(currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily exchange rate data from the Bank of Canada Valet API.
    
    Args:
        currency_pair: e.g. "FXUSDCAD" (USD to CAD)
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
    
    Returns:
        DataFrame with columns: date, rate
    """
    url = f"{BASE_URL}/observations/{currency_pair}/json"
    params = {
        "start_date": start_date,
        "end_date": end_date
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    observations = data["observations"]

    records = []
    for obs in observations:
        records.append({
            "date": obs["d"],
            "rate": float(obs[currency_pair]["v"])
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = get_fx_series("FXUSDCAD", "2015-01-01", "2024-12-31")
    print(df.head())
    print(f"Total observations: {len(df)}")