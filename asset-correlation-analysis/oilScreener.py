import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Polygon.io free tier: 5 requests/minute — delay between calls (seconds)
REQUEST_DELAY = 2

API_KEY = os.environ.get("POLYGON_API_KEY")

# ---------------------------------------
# TICKERS
# ---------------------------------------

oil_ticker = "USO"  # benchmark
transport_tickers = {
    "AAL": "American Airlines",
    "UAL": "United Airlines",
    "DAL": "Delta Air Lines",
    "LUV": "Southwest Airlines",
    "JBLU": "JetBlue Airways",
    "FDX": "FedEx",
    "UPS": "UPS",
    "UBER": "Uber",
    "CSX": "CSX (Rail)",
    "UNP": "Union Pacific (Rail)",
}

# ---------------------------------------
# DOWNLOAD FUNCTION
# ---------------------------------------


def get_polygon_data(ticker, days=500):
    """Fetch 500 days of daily OHLC data from Polygon.io"""
    end = datetime.now()
    start = end - timedelta(days=days * 1.4)  # buffer for weekends

    from_date = start.strftime("%Y-%m-%d")
    to_date = end.strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"

    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    r = requests.get(url, params=params)

    # Debug safety: show error if JSON decode fails
    try:
        data = r.json()
    except Exception as e:
        print(f"\nError decoding JSON for {ticker}: {e}")
        print("Status code:", r.status_code)
        print("Response preview:", r.text[:300])
        return None

    if "results" not in data:
        print(f"✗ Failed: {ticker} - response: {data}")
        return None

    df = pd.DataFrame(data["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("date", inplace=True)

    print(f"✓ Downloaded {ticker}")
    return df[["c"]]  # use close price only


# ---------------------------------------
# DOWNLOAD ALL DATA
# ---------------------------------------

print("\nDownloading market data...\n")

oil_df = get_polygon_data(oil_ticker)
time.sleep(REQUEST_DELAY)

transport_df = {}
for t in transport_tickers:
    transport_df[t] = get_polygon_data(t)
    time.sleep(REQUEST_DELAY)  # stay under Polygon.io rate limit

# Retry failed tickers after a minute (in case we hit rate limit)
failed = [t for t, df in transport_df.items() if df is None]
if failed:
    print(f"\nWaiting 60s before retrying {len(failed)} failed ticker(s)...\n")
    time.sleep(60)
    for t in failed:
        transport_df[t] = get_polygon_data(t)
        time.sleep(REQUEST_DELAY)

# Remove tickers that still failed
transport_df = {t: df for t, df in transport_df.items() if df is not None}

# ---------------------------------------
# MERGE INTO ONE TABLE
# ---------------------------------------

combined = oil_df.rename(columns={"c": oil_ticker})

for t, df in transport_df.items():
    combined[t] = df["c"]

combined = combined.dropna()

# ---------------------------------------
# DAILY RETURNS
# ---------------------------------------

returns = combined.pct_change().dropna()

oil_ret = returns[oil_ticker]
stock_returns = returns.drop(columns=[oil_ticker])

# ---------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------

correlations = stock_returns.corrwith(oil_ret)

corr_df = pd.DataFrame(
    {
        "Ticker": correlations.index,
        "Company": [transport_tickers[t] for t in correlations.index],
        "Correlation": correlations.values,
    }
).sort_values("Correlation", ascending=False)

print("\n======= CORRELATIONS VS OIL =======")
print(corr_df.to_string(index=False))

# ---------------------------------------
# BETA (LINEAR REGRESSION)
# ---------------------------------------

betas = {}
X = oil_ret.values.reshape(-1, 1)

for t in stock_returns.columns:
    y = stock_returns[t].values
    model = LinearRegression().fit(X, y)
    betas[t] = model.coef_[0]

beta_df = pd.DataFrame(
    {
        "Ticker": betas.keys(),
        "Company": [transport_tickers[t] for t in betas.keys()],
        "Beta_to_Oil": betas.values(),
    }
).sort_values("Beta_to_Oil", ascending=False)

print("\n======= BETA (SENSITIVITY TO OIL PRICE) =======")
print(beta_df.to_string(index=False))

# ---------------------------------------
# VISUALIZATION
# ---------------------------------------

plt.figure(figsize=(12, 6))
plt.barh(corr_df["Ticker"], corr_df["Correlation"], color="green")
plt.axvline(0, color="black")
plt.title(f"Correlation of Stocks vs {oil_ticker} (Oil Benchmark)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
