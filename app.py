"""
TRINETRA · Oracle — the forward-looking eye.

A small forecast microservice wrapping Kronos (shiyu-coder/Kronos, MIT),
an open-source foundation model for financial candlesticks. Given the
last ~200 daily candles of an NSE stock, it predicts the next N days of
OHLCV and reports the expected close-to-close return.

The Node backend consumes this as one number per symbol (fcstReturn),
which powers the "AI Forecast" criterion in the dashboard.

Modes (env MODE):
  kronos  — real Kronos-mini inference (needs torch; CPU is fine)
  naive   — statistical fallback (drift + momentum), no torch needed
  auto    — try kronos, fall back to naive (default)

Endpoints:
  GET /health
  GET /forecasts?symbols=POLYCAB,BEL&horizon=3
"""
import os
import io
import csv
import math
import time
import asyncio
import urllib.request
from datetime import date

from fastapi import FastAPI, Query

MODE = os.environ.get("MODE", "auto")
HORIZON_DEFAULT = int(os.environ.get("HORIZON_DAYS", "3"))
LOOKBACK = int(os.environ.get("LOOKBACK", "220"))
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "1"))

app = FastAPI(title="Trinetra Oracle")

# ---------------- Kronos (lazy, optional) ----------------------------
_predictor = None
_kronos_err = None

def get_predictor():
    """Load Kronos once, lazily. Falls back gracefully if unavailable."""
    global _predictor, _kronos_err
    if _predictor is not None or _kronos_err is not None:
        return _predictor
    if MODE == "naive":
        _kronos_err = "MODE=naive"
        return None
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # from the Kronos repo
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=2048)
        print("[oracle] Kronos-mini loaded (CPU)")
    except Exception as e:  # torch missing, no weights, etc.
        _kronos_err = str(e)
        print(f"[oracle] Kronos unavailable → naive fallback ({e})")
    return _predictor

# ---------------- Data: free EOD candles (Stooq) ---------------------
def fetch_candles(symbol: str):
    """Daily OHLCV for an NSE symbol from Stooq (free, no key)."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.in&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (trinetra-oracle)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for row in rows:
        try:
            out.append({
                "ts": row["Date"],
                "open": float(row["Open"]), "high": float(row["High"]),
                "low": float(row["Low"]), "close": float(row["Close"]),
                "volume": float(row.get("Volume") or 0),
            })
        except (KeyError, ValueError):
            continue
    return out[-LOOKBACK:]

# ---------------- Forecasters ----------------------------------------
def forecast_kronos(candles, horizon):
    import pandas as pd
    predictor = get_predictor()
    df = pd.DataFrame(candles)
    df["timestamps"] = pd.to_datetime(df["ts"])
    x_df = df[["open", "high", "low", "close", "volume"]]
    x_ts = df["timestamps"]
    last = x_ts.iloc[-1]
    y_ts = pd.Series(pd.bdate_range(start=last, periods=horizon + 1)[1:])
    pred = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=horizon,
        T=1.0, top_p=0.9, sample_count=SAMPLE_COUNT,
    )
    last_close = float(df["close"].iloc[-1])
    pred_close = float(pred["close"].iloc[-1])
    path = [round(float(c), 2) for c in pred["close"].tolist()]
    return {
        "ret": round((pred_close - last_close) / last_close * 100, 2),
        "path": path,
        "engine": "kronos-mini",
    }

def forecast_naive(candles, horizon):
    """Honest statistical fallback: recent drift blended with momentum,
    damped toward zero. Clearly labeled so it's never mistaken for Kronos."""
    closes = [c["close"] for c in candles]
    if len(closes) < 30:
        return None
    def ret(n):
        return (closes[-1] - closes[-n]) / closes[-n]
    drift = ret(20) / 20          # avg daily drift over ~1 month
    momo = ret(5) / 5             # short momentum
    daily = 0.5 * drift + 0.5 * momo
    daily = max(min(daily, 0.02), -0.02)  # damp outliers
    total = (1 + daily) ** horizon - 1
    last = closes[-1]
    path = [round(last * (1 + daily) ** (i + 1), 2) for i in range(horizon)]
    return {"ret": round(total * 100, 2), "path": path, "engine": "naive"}

def forecast(symbol, horizon):
    candles = fetch_candles(symbol)
    if len(candles) < 60:
        return None
    if get_predictor() is not None:
        try:
            return forecast_kronos(candles, horizon)
        except Exception as e:
            print(f"[oracle] kronos failed for {symbol}: {e}")
    return forecast_naive(candles, horizon)

# ---------------- Cache: one forecast per symbol per day -------------
_cache = {}  # symbol -> {"day": iso, "horizon": n, "data": {...}}

def cached_forecast(symbol, horizon):
    today = date.today().isoformat()
    hit = _cache.get(symbol)
    if hit and hit["day"] == today and hit["horizon"] == horizon:
        return hit["data"]
    data = forecast(symbol, horizon)
    _cache[symbol] = {"day": today, "horizon": horizon, "data": data}
    return data

# ---------------- API -------------------------------------------------
@app.get("/health")
def health():
    p = get_predictor()
    return {
        "ok": True,
        "engine": "kronos-mini" if p is not None else "naive",
        "note": None if p is not None else f"Kronos not loaded: {_kronos_err}",
        "cached": len(_cache),
    }

@app.get("/forecasts")
async def forecasts(symbols: str = Query(...), horizon: int = HORIZON_DEFAULT):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:100]
    out = {}
    for s in syms:
        try:
            data = await asyncio.to_thread(cached_forecast, s, horizon)
            if data:
                out[s] = {**data, "horizon": horizon, "asOf": date.today().isoformat()}
        except Exception as e:
            print(f"[oracle] {s}: {e}")
    return out
