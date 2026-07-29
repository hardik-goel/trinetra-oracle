# Trinetra · Oracle

Forward-looking forecast service wrapping **Kronos** (shiyu-coder/Kronos, MIT) —
an open-source foundation model for candlestick forecasting. Feeds the
"AI Forecast" criterion in Trinetra: expected % return over the next N days.

## API
- `GET /health` — engine in use (kronos-mini or naive) + cache size
- `GET /forecasts?symbols=POLYCAB,BEL&horizon=3` →
  `{ "POLYCAB": { "ret": 2.4, "path": [...], "engine": "kronos-mini", ... } }`

Forecasts are computed once per symbol per day (EOD data → daily cache),
so inference cost stays tiny even on CPU.

## Run
Naive mode (no torch, runs anywhere, free):
```
pip install -r requirements.txt
MODE=naive uvicorn app:app --port 8000
```
Full Kronos (CPU is fine, Kronos-mini is 4.1M params):
```
docker build -t trinetra-oracle .
docker run -p 8000:8000 trinetra-oracle
```
First start downloads weights from Hugging Face (~few min).

## Deploy
- **Hugging Face Spaces (Docker, free)** — easiest home for the torch image.
- Render/Railway: needs ≥1GB RAM for torch; the naive-only image fits free tiers.
- Then set `ORACLE_URL=https://…` on the Trinetra backend.

## Honest limits
- Kronos is a research model. Its forecasts are probabilistic candles, not
  guaranteed alpha — use the criterion as a *filter*, and backtest it against
  your signals before trusting it with money.
- The naive fallback is drift+momentum, clearly labeled `"engine": "naive"` —
  never silently pretends to be Kronos.
- EOD data in → the forecast updates once a day. That matches the swing
  horizon; it is not an intraday signal.
