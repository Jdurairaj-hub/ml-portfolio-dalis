import os
import pandas as pd
import requests
from datetime import datetime, timedelta


def fetch_polygon_data(ticker, api_key, days_back=365):
    """Fetch historical daily OHLCV aggregates from Polygon API.
    Returns DataFrame indexed by date with columns: open, high, low, close, volume.
    """
    if not api_key:
        raise ValueError("Polygon API key is missing. Set environment variable POLYGON_API_KEY.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'apiKey': api_key
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if 'results' not in data:
        raise ValueError(f"Error fetching data from Polygon: {data}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.set_index('date')

    return df
