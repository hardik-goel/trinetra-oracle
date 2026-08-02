"""
TRINETRA · Oracle — the forward-looking eye.

A small forecast microservice wrapping Kronos (shiyu-coder/Kronos, MIT),
an open-source foundation model for financial candlesticks. Given the
last ~200 daily candles of an NSE stock, it predicts the next N days of
OHLCV and reports the expected close-to-close return.

The Node backend consumes this as one number per symbol (fcstReturn),
which powers the "AI Forecast" criterion in the dashboard.

Daily candles come from Yahoo Finance (same feed as the backend's yahooDelayed
provider), with Stooq kept only as a fallback — Stooq now serves a bot-check
page instead of CSV. See the constants block under "Data".

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
import json
import math
import time
import asyncio
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

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

# ---------------- Data: free EOD candles ------------------------------
# One block for every endpoint/selector we depend on, so a blocked or moved
# source is a one-line patch rather than a hunt through the file.
#
# Stooq (the original source) now answers with a JavaScript proof-of-work
# bot-check page instead of CSV, so it parses to ~0 rows. Yahoo's chart API is
# the same feed the backend's working `yahooDelayed` provider uses, so it leads;
# Stooq stays wired up as a fallback in case the two swap places again.
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1y"
YAHOO_SUFFIX = ".NS"        # Yahoo's NSE convention: RELIANCE -> RELIANCE.NS
STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={sym}.in&i=d"
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "10"))
SYMBOL_DELAY = float(os.environ.get("SYMBOL_DELAY", "0.3"))  # be a good citizen
# A real browser UA: Yahoo serves an error page to obviously-scripted clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/csv,*/*",
    })
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def yahoo_symbol(symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS; pass through anything already suffixed."""
    s = symbol.strip().upper()
    return s if "." in s else s + YAHOO_SUFFIX


def fetch_yahoo(symbol: str):
    """Daily OHLCV from Yahoo's chart API."""
    payload = json.loads(_http_get(YAHOO_CHART_URL.format(sym=yahoo_symbol(symbol))))
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise ValueError(f"empty chart result ({err})")
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    o, h, l = quote.get("open") or [], quote.get("high") or [], quote.get("low") or []
    c, v = quote.get("close") or [], quote.get("volume") or []

    out = []
    for i, ts in enumerate(stamps):
        try:
            # Yahoo leaves nulls in the series for halted/missing sessions.
            row = (o[i], h[i], l[i], c[i])
            if ts is None or any(x is None for x in row):
                continue
            out.append({
                "ts": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"),
                "open": float(o[i]), "high": float(h[i]),
                "low": float(l[i]), "close": float(c[i]),
                "volume": float(v[i] or 0) if i < len(v) else 0.0,
            })
        except (IndexError, TypeError, ValueError):
            continue
    return out[-LOOKBACK:]


def fetch_stooq(symbol: str):
    """Daily OHLCV from Stooq CSV. Deprecated — kept as a fallback only."""
    text = _http_get(STOOQ_CSV_URL.format(sym=symbol.lower()))
    if not text.lstrip().lower().startswith("date"):
        raise ValueError("not CSV (bot-check page?)")
    out = []
    for row in csv.DictReader(io.StringIO(text)):
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


SOURCES = [("yahoo", fetch_yahoo), ("stooq", fetch_stooq)]


def fetch_candles(symbol: str):
    """Daily OHLCV for an NSE symbol. Returns (candles, source_name).

    Tries each source in order and takes the first that yields usable rows, so
    one blocked provider degrades to the other instead of to silence.
    """
    for name, fn in SOURCES:
        try:
            candles = fn(symbol)
        except urllib.error.HTTPError as e:
            print(f"[oracle] {symbol}: {name} HTTP {e.code}")
            continue
        except Exception as e:
            print(f"[oracle] {symbol}: {name} failed ({e})")
            continue
        finally:
            time.sleep(SYMBOL_DELAY)  # pace network hits regardless of outcome
        if candles:
            print(f"[oracle] {symbol}: {len(candles)} candles from {name}")
            return candles, name
        print(f"[oracle] {symbol}: {name} returned 0 candles")
    print(f"[oracle] {symbol}: no data source available")
    return [], None

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
    candles, source = fetch_candles(symbol)
    if len(candles) < 60:
        print(f"[oracle] {symbol}: only {len(candles)} candles (need 60) — skipping")
        return None
    out = None
    if get_predictor() is not None:
        try:
            out = forecast_kronos(candles, horizon)
        except Exception as e:
            print(f"[oracle] kronos failed for {symbol}: {e}")
    if out is None:
        out = forecast_naive(candles, horizon)
    if out is not None:
        out["source"] = source
    return out

# ---------------- Cache: one forecast per symbol per day -------------
# Successes only. Caching a None would pin a transient data outage for the rest
# of the day and report it as a "cached forecast", which it is not.
_cache = {}     # symbol -> {"day": iso, "horizon": n, "data": {...}}
_failed = {}    # symbol -> iso day of the last failed attempt (never gates retries)

def cached_forecast(symbol, horizon):
    today = date.today().isoformat()
    hit = _cache.get(symbol)
    if hit and hit["day"] == today and hit["horizon"] == horizon:
        return hit["data"]
    data = forecast(symbol, horizon)
    if data:
        _cache[symbol] = {"day": today, "horizon": horizon, "data": data}
        _failed.pop(symbol, None)
    else:
        _failed[symbol] = today  # observability only — next request retries
    return data

# ---------------- API -------------------------------------------------
@app.get("/health")
def health():
    p = get_predictor()
    today = date.today().isoformat()
    ok = sum(1 for v in _cache.values() if v["day"] == today and v["data"])
    return {
        "ok": True,
        "engine": "kronos-mini" if p is not None else "naive",
        "note": None if p is not None else f"Kronos not loaded: {_kronos_err}",
        "cached": ok,        # successful forecasts held for today
        "cached_ok": ok,
        "cached_failed": sum(1 for d in _failed.values() if d == today),
        "sources": [name for name, _ in SOURCES],
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
