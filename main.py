from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BINANCE_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines"
]

ALLOWED_INTERVALS = ["5m", "15m", "1h", "1d"]

# ------------------ DATA FETCH ------------------

def get_klines(symbol: str, interval: str, limit: int = 200):

    if interval not in ALLOWED_INTERVALS:
        interval = "15m"

    last_error = None

    for url in BINANCE_URLS:
        try:
            res = requests.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )

            if res.status_code != 200:
                last_error = f"{url} → {res.status_code}"
                continue

            data = res.json()

            if not isinstance(data, list):
                last_error = f"{url} → invalid data"
                continue

            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "_", "_", "_", "_", "_", "_"
            ])

            df["close"] = df["close"].astype(float)
            return df

        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(
        status_code=500,
        detail=f"Binance API error (all endpoints failed)"
    )

# ------------------ INDICATORS ------------------

def calculate_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)

def generate_explanation(trend, rsi, ema20, ema50, interval):
    tf = "Daily timeframe" if interval == "1d" else f"{interval} timeframe"

    if trend == "Bullish":
        return f"{tf}: EMA20 ({ema20}) EMA50 ({ema50}) ke upar hai aur RSI {rsi}. Buyers strong."
    elif trend == "Bearish":
        return f"{tf}: EMA20 ({ema20}) EMA50 ({ema50}) ke niche hai aur RSI {rsi}. Sellers dominate."
    else:
        return f"{tf}: EMA20 aur EMA50 ke beech price aur RSI {rsi} neutral."

# ------------------ API ------------------

@app.get("/analyze")
def analyze(symbol: str, interval: str = "15m"):

    symbol = symbol.upper()
    df = get_klines(symbol, interval)

    price = round(df["close"].iloc[-1], 2)

    rsi_series = calculate_rsi(df["close"])
    rsi = round(rsi_series.iloc[-1], 2)

    ema20_series = df["close"].ewm(span=20, adjust=False).mean()
    ema50_series = df["close"].ewm(span=50, adjust=False).mean()

    ema20 = round(ema20_series.iloc[-1], 2)
    ema50 = round(ema50_series.iloc[-1], 2)

    if ema20 > ema50 and rsi > 55:
        trend = "Bullish"
    elif ema20 < ema50 and rsi < 45:
        trend = "Bearish"
    else:
        trend = "Sideways"

    return {
        "symbol": symbol,
        "interval": interval,
        "trend": trend,
        "entry": price,
        "stoploss": round(price * (0.97 if interval == "1d" else 0.98), 2),
        "target": round(price * (1.06 if interval == "1d" else 1.03), 2),
        "confidence": f"RSI {rsi}",
        "explanation": generate_explanation(trend, rsi, ema20, ema50, interval),
        "prices": df["close"].tail(30).tolist(),
        "ema20_list": ema20_series.tail(30).tolist(),
        "ema50_list": ema50_series.tail(30).tolist(),
        "rsi_list": rsi_series.tail(30).tolist()
    }
