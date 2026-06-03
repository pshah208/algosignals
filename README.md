# AlgoSignals

LLM-assisted daily **Buy / Sell / Hold** stock recommendation app.

AlgoSignals generates a daily ranked watchlist of recommendations from a composite
score that blends **technical/price signals, news sentiment, major events, company
financials, and earnings calls**. An optional LLM layer (backed by **GitHub Models /
GitHub Copilot**) summarizes qualitative inputs and writes a plain-English rationale
for each recommendation.

> ⚠️ **Advisory only.** AlgoSignals is a research tool. It is **not investment advice**
> and it **does not place brokerage orders**.

---

## Table of contents

- [Features](#features)
- [Scoring model](#scoring-model)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Configuration](#configuration)
- [GitHub Models / Copilot LLM](#github-models--copilot-llm)
- [Switching LLM models at runtime](#switching-llm-models-at-runtime)
- [Running the app](#running-the-app)
- [Daily schedule](#daily-schedule)
- [JSON API](#json-api)
- [Running tests](#running-tests)
- [Disclaimer](#disclaimer)

---

## Features

| Feature | Detail |
|---|---|
| **Dashboard** | Latest run table — symbol, BUY/SELL/HOLD badge, composite score, per-factor sub-scores, rationale |
| **Watchlist** | Add/remove/enable/disable symbols |
| **Config panel** | Factor weights, BUY/SELL thresholds, scheduler IST time, enable/disable |
| **Run History** | Recent run timestamps and status |
| **JSON API** | `GET /api/v1/recommendations` (read-only, API-key protected) |
| **Scheduler** | APScheduler daily job at configurable IST time |
| **LLM rationale** | GitHub Models / OpenAI-compatible — fully optional |

---

## Scoring model

Each symbol passes through **five factor modules**, each returning a normalised score
in **−1.0 (very bearish) … +1.0 (very bullish)**.  Every module degrades gracefully
to a neutral `0.0` when its data source is unavailable.

| Factor | Weight (default) | Signal source |
|---|---|---|
| `technical` | 35 % | yfinance OHLCV → 20/50 SMA crossover, RSI-14, 20-day momentum, ATR volatility |
| `news` | 25 % | RSS / NewsAPI → LLM or lexicon sentiment |
| `financials` | 20 % | yfinance `info` → P/E, revenue growth, profit margin, debt/equity |
| `events` | 10 % | yfinance dividends, splits, earnings calendar |
| `earnings` | 10 % | yfinance EPS surprise, forward EPS growth, upcoming earnings proximity |

**Composite score** = weighted average of the five factor scores.

| Composite ≥ BUY threshold (default 0.15) | → `BUY` |
|---|---|
| Composite ≤ SELL threshold (default −0.15) | → `SELL` |
| Otherwise | → `HOLD` |

Weights and thresholds are configurable from `/config`.

---

## Project layout

```
app.py                           # Flask app factory + blueprint registration + DB/scheduler init
config.py                        # Env-driven settings
requirements.txt
.env.example
database/
  db.py                          # Engine/session (NullPool), init_db()
  models.py                      # Watchlist, AppConfig, SignalRun, Recommendation
services/
  signal_service.py              # Orchestrator + composite scoring
  scheduler.py                   # APScheduler daily job
  factors/
    __init__.py                  # FactorResult dataclass
    technical_signals.py
    news_signals.py
    events_signals.py
    financials_signals.py
    earnings_signals.py
  llm/
    llm_client.py                # GitHub Models / OpenAI-compatible client (optional)
    prompts.py                   # Prompt templates
blueprints/
  dashboard.py                   # / and /runs
  watchlist.py                   # /watchlist
  config_routes.py               # /config
  api.py                         # /api/v1 (read-only JSON)
templates/
  base.html, dashboard.html, watchlist.html, config.html, runs.html
utils/
  logging.py                     # get_logger() helper
tests/
  test_factors.py
  test_scoring.py
  test_llm_optional.py
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/pshah208/algosignals.git
cd algosignals

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — see Configuration section below
```

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key` | Flask session secret — **change in production** |
| `DATABASE_URL` | `sqlite:///algosignals.db` | SQLAlchemy URL |
| `GITHUB_TOKEN` | *(empty)* | GitHub personal-access token for GitHub Models LLM (optional) |
| `GITHUB_MODELS_TOKEN` | *(empty)* | Alias for `GITHUB_TOKEN` |
| `LLM_BASE_URL` | `https://models.github.ai/inference` | LLM inference endpoint |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | Default model identifier (always first in selector) |
| `LLM_AVAILABLE_MODELS` | *(seven models — see below)* | Comma-separated list of models shown in the dashboard selector |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (alternative to GitHub Models) |
| `NEWS_API_KEY` | *(empty)* | NewsAPI key (optional — falls back to free RSS) |
| `SCHEDULE_HOUR_IST` | `9` | Daily run hour in IST |
| `SCHEDULE_MINUTE_IST` | `0` | Daily run minute in IST |
| `APP_API_KEY` | *(empty)* | API key for `GET /api/v1/recommendations` |

---

## GitHub Models / Copilot LLM

AlgoSignals uses an **OpenAI-compatible** chat endpoint for:
1. Summarising news headlines into a sentiment score.
2. Writing a plain-English rationale for each BUY/SELL/HOLD recommendation.

### Using GitHub Models (recommended, free tier available)

1. Go to <https://github.com/settings/tokens> → **Generate new token (fine-grained)**.
   No special scopes are needed; the token just authenticates you to the GitHub Models API.
2. Set it in `.env`:
   ```
   GITHUB_TOKEN=github_pat_...
   LLM_BASE_URL=https://models.github.ai/inference
   LLM_MODEL=openai/gpt-4.1-mini
   ```

### Using OpenAI directly

```
OPENAI_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### LLM disabled (no token)

If neither `GITHUB_TOKEN` nor `OPENAI_API_KEY` is set:
- Sentiment scoring uses a **lexicon-based fallback**.
- Rationales use a **template** with the numeric scores.
- A badge in the dashboard shows **"heuristic"** instead of **"LLM"**.
- The app continues to work fully — no crash, no degraded functionality.

---

## Switching LLM models at runtime

The dashboard shows an **LLM model** selector at the top.  Pick any model from
the dropdown and click **Switch** (or just change the selection — it auto-submits).
The chosen model is stored in the Flask session and used for all subsequent LLM
calls (sentiment scoring and rationale generation) without restarting the app.

### Available models (default list)

| Identifier | Provider |
|---|---|
| `openai/gpt-4.1-mini` | OpenAI via GitHub Models |
| `openai/gpt-4.1` | OpenAI via GitHub Models |
| `openai/gpt-4o-mini` | OpenAI via GitHub Models |
| `openai/gpt-4o` | OpenAI via GitHub Models |
| `meta/llama-3.3-70b-instruct` | Meta via GitHub Models |
| `microsoft/phi-4` | Microsoft via GitHub Models |
| `mistral-ai/mistral-small` | Mistral via GitHub Models |

Customise the list by setting `LLM_AVAILABLE_MODELS` (comma-separated) in `.env`.
`LLM_MODEL` is always included as the default selection.

### Model API

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | `GET` | Returns `{"available": [...], "active": "..."}` |
| `/api/models` | `POST` | Sets active model (`{"model": "..."}` JSON or form). Returns 400 for unknown models. |

---

## Running the app

```bash
python app.py
```

Then open <http://localhost:5000>.

For production use Gunicorn with **a single worker** (required for APScheduler):

```bash
gunicorn -w 1 app:app
```

### First run

1. Visit `/watchlist` → add one or more symbols (e.g. `AAPL`, `RELIANCE.NS`).
2. Return to `/` → click **▶ Run Now**.
3. Results appear in the dashboard table sorted by composite score.

---

## Daily schedule

The APScheduler job runs the full signal pipeline once a day at the configured IST time
(default: 09:00 IST = 03:30 UTC).

- Change the time and enable/disable it from `/config` without restarting the app.
- The scheduler is safe for single-process Flask; it will not double-fire with
  Werkzeug's reloader because app.py sets `use_reloader=False`.
- With Gunicorn (`-w 1`) the reloader is not used, so there is no risk.

---

## JSON API

```
GET /api/v1/recommendations
```

Returns the latest run as JSON.  **Read-only — no orders are placed.**

Authentication: pass the `APP_API_KEY` from `.env` as a bearer token or query param:

```bash
curl -H "Authorization: ******" http://localhost:5000/api/v1/recommendations
# or
curl "http://localhost:5000/api/v1/recommendations?api_key=<APP_API_KEY>"
```

If `APP_API_KEY` is not set in `.env`, the endpoint is open (development convenience).

---

## Running tests

Tests run **offline** — all network calls are mocked.  No real API keys are required.

```bash
pytest
```

The test suite covers:
- All five factor modules: valid `FactorResult` output and graceful degradation when
  the data provider is missing or raises an exception.
- Composite scoring and BUY/SELL/HOLD threshold mapping.
- LLM disabled path: correct fallback rationale and neutral sentiment score.
- LLM HTTP failure: no crash, correct fallback.

---

## Disclaimer

> **AlgoSignals is a research and educational tool only.**
>
> - It is **not investment advice**.
> - It **does not place brokerage orders** of any kind.
> - Past performance implied by any signal is not indicative of future results.
> - Always do your own due diligence before making investment decisions.
> - The authors accept no liability for financial losses arising from use of this software.

