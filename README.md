# Trinetra · Oracle

Forward-looking forecast service wrapping **Kronos** (shiyu-coder/Kronos, MIT) —
an open-source foundation model for candlestick forecasting. Feeds the
"AI Forecast" criterion in Trinetra: expected % return over the next N days.

## API
- `GET /health` — engine in use (kronos-mini or naive), data sources, and
  `cached_ok` / `cached_failed` (successes held for today vs symbols whose last
  attempt failed; failures are *not* cached and retry on the next request)
- `GET /forecasts?symbols=POLYCAB,BEL&horizon=3` →
  `{ "POLYCAB": { "ret": 2.4, "path": [...], "engine": "kronos-mini", "source": "yahoo", ... } }`

Successful forecasts are computed once per symbol per day (EOD data → daily
cache), so inference cost stays tiny even on CPU. A symbol with no usable data
is omitted from the response — the backend already handles missing forecasts.

## Data source
Daily candles come from **Yahoo Finance** (`query1.finance.yahoo.com/v8/finance/chart/<SYM>.NS`,
`interval=1d&range=1y`) — the same feed the backend's working `yahooDelayed`
provider uses. NSE symbols get the `.NS` suffix (RELIANCE → RELIANCE.NS).

**Stooq is deprecated.** It now answers `stooq.com/q/d/l/` with a JavaScript
proof-of-work bot-check page instead of CSV, so it parsed to ~0 rows and every
symbol fell below the 60-candle minimum — the Oracle returned `{}` for
everything. It is still wired up as a *fallback* behind Yahoo (`SOURCES` in
`app.py`) in case the two ever swap places. All endpoint URLs and the symbol
suffix live in one constants block near the top of `app.py`.

> The backend's default `stooqEod.js` provider has the same problem and should
> get the same swap to Yahoo daily candles.

Per-symbol candle counts and the source used are logged, so Render logs show at
a glance whether Yahoo is delivering:
`[oracle] RELIANCE: 220 candles from yahoo`

Note that Yahoo rate-limits aggressively per IP (HTTP 429). Keep `SYMBOL_DELAY`
(default 0.3s) in place; `HTTP_TIMEOUT` defaults to 10s.

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
- The data feed is free and unofficial. It can be rate-limited or blocked
  without notice — that is exactly how Stooq died. Watch the per-symbol candle
  counts in the logs and `cached_failed` on `/health`.
