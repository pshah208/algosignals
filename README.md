# AlgoSignals

LLM-assisted daily **Buy / Sell / Hold** stock recommendation app.

AlgoSignals generates a daily ranked watchlist of recommendations from a composite
score that blends **technical/price signals, news sentiment, major events, company
financials, and earnings calls**. An optional LLM layer (backed by an **OpenAI-compatible
provider or the official GitHub Copilot SDK**) summarizes qualitative inputs and writes a plain-English rationale
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
- [Copilot research and TradingView MCP](#copilot-research-and-tradingview-mcp)
- [Running the app](#running-the-app)
- [Daily schedule](#daily-schedule)
- [JSON API](#json-api)
- [Running tests](#running-tests)
- [Disclaimer](#disclaimer)

---

## Features

| Feature | Detail |
|---|---|
| **Dashboard** | Latest run table — symbol, current price, BUY/SELL/HOLD badge, composite score, per-factor sub-scores, rationale |
| **Watchlist** | Add/remove/enable/disable symbols |
| **Config panel** | Factor weights, BUY/SELL thresholds, scheduler IST time, enable/disable |
| **Run History** | Recent run timestamps and status |
| **Scoring Metrics page** | `/metrics` explains each factor, neutral `0.0` cases, and score thresholds |
| **Price prediction** | `/predict/<symbol>` 30-day close forecast using a 50-step lookback LSTM (with graceful fallback) |
| **JSON API** | `GET /api/v1/recommendations` (read-only, API-key protected) |
| **Scheduler** | APScheduler daily job at configurable IST time |
| **LLM rationale** | OpenAI-compatible provider or GitHub Copilot SDK — fully optional |
| **Research assistant** | `/research`: Copilot answers with saved evidence, prices, and optional read-only TradingView MCP tools |

---

## Scoring model

Each symbol passes through **five factor modules**, each returning a normalised score
in **−1.0 (very bearish) … +1.0 (very bullish)**.  Every module degrades gracefully
to a neutral `0.0` when its data source is unavailable.

| Factor | Weight (default) | Signal source |
|---|---|---|
| `technical` | 35 % | yfinance OHLCV → 20/50 SMA crossover, RSI-14, 20-day momentum, ATR volatility |
| `news` | 25 % | Google News RSS (default, no key) + optional NewsAPI → LLM or VADER/lexicon sentiment |
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
  prediction/
    lstm_predictor.py            # 30-day close forecast service
  factors/
    __init__.py                  # FactorResult dataclass
    technical_signals.py
    news_signals.py
    events_signals.py
    financials_signals.py
    earnings_signals.py
  llm/
    llm_client.py                # Optional Models-compatible / Copilot provider
    prompts.py                   # Prompt templates
blueprints/
  dashboard.py                   # /, /runs, /metrics, /research
  watchlist.py                   # /watchlist
  config_routes.py               # /config
  api.py                         # /api/v1/recommendations (read-only JSON)
  research.py                    # Copilot status and research API
  predict.py                     # /predict/<symbol>, /predict/api/<symbol>
templates/
  base.html, dashboard.html, research.html, watchlist.html, config.html, runs.html, metrics.html, predict.html
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
| `LLM_PROVIDER` | `models` | `models` for OpenAI-compatible inference; `copilot` for the official Copilot SDK |
| `GITHUB_TOKEN` | *(empty)* | Legacy GitHub Models PAT, not a Copilot login (see retirement notice below) |
| `GITHUB_MODELS_TOKEN` | *(empty)* | Alias for `GITHUB_TOKEN` |
| `LLM_BASE_URL` | `https://models.github.ai/inference` | LLM inference endpoint |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | Default model identifier (always first in selector) |
| `LLM_AVAILABLE_MODELS` | *(configured fallback list)* | Comma-separated model IDs for the Models-compatible selector; not a guaranteed catalog |
| `COPILOT_MODEL` | `gpt-4.1` | Default Copilot model ID; available models are discovered through the SDK |
| `GITHUB_CLIENT_ID` | *(empty)* | GitHub OAuth app client ID; enables browser login together with the client secret |
| `GITHUB_CLIENT_SECRET` | *(empty)* | GitHub OAuth app secret; keep server-side |
| `GITHUB_REDIRECT_URI` | *(empty)* | Explicit OAuth callback URL; configure HTTPS in production and match the registered app callback |
| `TRADINGVIEW_MCP_URL` | *(empty)* | Optional trusted self-hosted TradingView MCP HTTP endpoint, e.g. `http://127.0.0.1:8000/mcp` |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (alternative to GitHub Models) |
| `NEWS_API_KEY` | *(empty)* | NewsAPI key (optional — app works without it via RSS + VADER) |
| `SCHEDULE_HOUR_IST` | `9` | Daily run hour in IST |
| `SCHEDULE_MINUTE_IST` | `0` | Daily run minute in IST |
| `APP_API_KEY` | *(empty)* | Protects recommendations when configured; **mandatory** for Copilot status, research, and Copilot `POST /run-now` |

---

## GitHub Models / Copilot LLM

Choose one of two separate provider APIs. `LLM_PROVIDER=models` uses an
**OpenAI-compatible** chat endpoint; `LLM_PROVIDER=copilot` uses the official
`github-copilot-sdk` (pinned in requirements), not the Models inference API. Both support:
1. Summarising news headlines into a sentiment score.
2. Writing a plain-English rationale for each BUY/SELL/HOLD recommendation.

### GitHub Models legacy configuration

[GitHub's current documentation](https://docs.github.com/en/github-models) states that
GitHub Models retired on July 30, 2026, including its inference API. The legacy
`models` provider name and default endpoint remain for compatibility; do not assume
the default endpoint works. Configure a supported OpenAI-compatible endpoint instead,
or use Copilot below.

Historically, a fine-grained GitHub Models PAT needed **Models: read** (`models:read`)
permission, not “no scopes.” `GITHUB_TOKEN` / `GITHUB_MODELS_TOKEN` are Models
credentials and do not authenticate GitHub Copilot. Models and Copilot are separate
services with different model IDs, catalogs, and access rights.

### Using GitHub Copilot

After installing Python requirements, preprovision the SDK runtime:

```bash
python -m copilot download-runtime
```

Configure:

```dotenv
LLM_PROVIDER=copilot
COPILOT_MODEL=gpt-4.1
APP_API_KEY=replace-with-a-strong-unique-secret
```

#### Browser GitHub OAuth login

Register a [GitHub OAuth app](https://github.com/settings/developers) for this
deployment, with callback `/auth/github/callback`. Configure its credentials:

```dotenv
GITHUB_CLIENT_ID=your-oauth-app-client-id
GITHUB_CLIENT_SECRET=your-oauth-app-client-secret
GITHUB_REDIRECT_URI=http://localhost:5000/auth/github/callback
```

For production, set a strong `SECRET_KEY`, use HTTPS, and set
`GITHUB_REDIRECT_URI` explicitly to the HTTPS callback registered on the OAuth app.
Keep the client secret private.

Use **Sign in with GitHub** on the dashboard or research page. GitHub OAuth app
user tokens authenticate the user's Copilot session; the account still needs
Copilot access. Each browser has an isolated runtime. Tokens live only in an
in-memory server vault, never in session cookies, browser storage, or the database.
The login expires after **one hour**, and server restarts require sign-in again.
**Sign out** submits a CSRF-protected POST and clears the browser's login.
It revokes only this app's local token/session; it does not revoke the GitHub OAuth
app authorization. To revoke that authorization, use GitHub's authorized applications
settings.

When OAuth is configured, Copilot calls require browser login and **do not fall
back to the operator's CLI account**. `APP_API_KEY` is also required for status,
research, and manual Copilot signal runs. Open `/research`, enter the application
key (not a GitHub token), and check status.

#### Operator CLI fallback (OAuth not configured)

When browser OAuth is **not configured** and the browser is not logged in, the
SDK can reuse the host operator's Copilot CLI login. Install the official
[Copilot CLI](https://docs.github.com/en/copilot/get-started/cli-quickstart) and,
as the **same OS account that runs Flask**, run `copilot login` and follow its
browser/device prompts. Alternatively, start `copilot` and enter `/login`.

This fallback is a trusted **single-operator deployment**: authorized app users
invoke the operator's account and may consume its quota. Keep it private or behind
appropriate deployment access controls. `GITHUB_TOKEN` and `GH_TOKEN` are stripped
from the SDK process environment so a Models PAT cannot override Copilot login.

### Using OpenAI directly

```
LLM_PROVIDER=models
GITHUB_TOKEN=
GITHUB_MODELS_TOKEN=
OPENAI_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_AVAILABLE_MODELS=gpt-4o-mini
```

### LLM unavailable

In Models-compatible mode, if neither `GITHUB_TOKEN` nor `OPENAI_API_KEY` is set,
or inference fails:
- Sentiment scoring uses **VADER**, with lexicon fallback if VADER is unavailable.
- Rationales use a **template** with the numeric scores.
- A badge in the dashboard shows **"heuristic"** instead of **"LLM"**.
- The app continues to work fully — no crash, no degraded functionality.

Copilot configured without an authenticated GitHub login, or with inference
failures, likewise falls back to VADER/heuristic scoring and template rationales.
Each recommendation's `llm_used` flag reflects actual successful LLM calls, not
merely whether a provider was configured.
Research is Copilot-only and returns an explicit unavailable error instead of
pretending to have generated a research answer.

---

## Switching LLM models at runtime

The dashboard shows an **LLM model** selector at the top.  Pick any model from
the dropdown and click **Switch** (or just change the selection — it auto-submits).
The chosen model is stored in that browser's Flask session and used for its manual
signal runs and research without restarting the app. It does not mutate a shared
global model or another visitor's selection. Scheduled jobs use the configured default.

### Provider-specific model availability

In Copilot mode, model discovery uses the **Copilot SDK**. `COPILOT_MODEL` is the
configured default; model access depends on the authenticated user's entitlement and account
policy. Copilot IDs such as `gpt-4.1` are not interchangeable with Models IDs such
as `openai/gpt-4.1`.

In Models-compatible mode, customise `LLM_AVAILABLE_MODELS` (comma-separated) for
your actual endpoint; `LLM_MODEL` is included as the default selection. The bundled
list is a **configurable fallback**, not a verified catalog or a promise that each
entry exists on GitHub Models. In particular, a model offered by Copilot must not
be described as available through GitHub Models solely because it is in that list.

### Why News or Events can be 0.0

- **News 0.0** can be legitimate for neutral/mixed headlines or sparse coverage. By default AlgoSignals uses Google News RSS (no API key), then optional NewsAPI.
- **Events 0.0** is common when there is no recent dividend/split and no near-term earnings event in the scoring window.

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
4. Click any ticker symbol to open `/predict/<symbol>` and view the 30-day forecast chart.

Symbol processing uses a bounded four-worker pipeline while keeping ORM work on
the caller thread and preserving deterministic scoring. Current prices use a
60-second, 256-entry cache; failed price lookups are not cached.

---

## Copilot research and TradingView MCP

Open `/research` with `LLM_PROVIDER=copilot`, sign in with GitHub if OAuth is
configured, enter `APP_API_KEY`, check status,
and ask a 1–2000 character question about a single symbol. Select the model on the
dashboard first. The page sends the key in a bearer authorization header;
it does not save the key in localStorage or cookies. Assistant answers are displayed
as plain text, not executable HTML.

The assistant has two custom tools, `saved_signal_evidence` and `current_price`,
restricted to the requested symbol. Research is bounded to **eight tool calls**
and **60 seconds total**, exposes no built-in shell/file tools, and places no
trades. Answers must identify missing/stale evidence; verify timestamps and sources
rather than treating an answer as live market data.

The `copilot-cli` runtime uses an isolated working directory with configuration
and skills discovery disabled. Host/file/shell tools are disabled; an explicit
allowlist exposes only the custom tools and the four optional read-only TradingView
tools below, not the CLI's general-purpose tools.

Optionally install and run **tradingview-mcp** as a separate trusted self-hosted
HTTP server from its official source, following that project's installation and
HTTP transport instructions. It is deliberately **not** a dependency of this app.
Keep it loopback-only; the verified HTTP launch command is:

```bash
tradingview-mcp streamable-http --host 127.0.0.1 --port 8000
```

Configure the app:

```dotenv
TRADINGVIEW_MCP_URL=http://127.0.0.1:8000/mcp
```

Only four read-only tools are exposed: `yahoo_price`, `combined_analysis`,
`financial_news`, and `multi_timeframe_analysis`. Do not point this setting at an
untrusted remote server. If it is unset or unavailable, the assistant must disclose
missing optional analysis rather than fabricate it.

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
curl --oauth2-bearer '<APP_API_KEY>' http://localhost:5000/api/v1/recommendations
# or
curl "http://localhost:5000/api/v1/recommendations?api_key=<APP_API_KEY>"
```

If `APP_API_KEY` is not set in `.env`, **only the recommendations endpoint** is open
(development convenience). Copilot endpoints and Copilot Run Now fail closed.

### Copilot endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/copilot/status` | `GET` | Returns `enabled`, `authenticated`, and `available` |
| `/api/v1/research` | `POST` | JSON `symbol` and `question`; uses the browser session's model |

Both require a configured `APP_API_KEY` and a bearer authorization header.
When GitHub OAuth is configured, they also require the authenticated browser's
session; use the research page for that flow. The CLI example below applies to
operator fallback without OAuth.
Example research request (replace the placeholder locally; do not publish keys):

```bash
curl --oauth2-bearer '<APP_API_KEY>' \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","question":"What supports the saved signal, and what evidence is missing?"}' \
  http://localhost:5000/api/v1/research
```

Research returns `symbol`, `answer`, `tools_used`, `model`, and `disclaimer`.
Errors: `400` invalid input, `401` unauthorized, `503` disabled, unconfigured, or
unavailable. No model is accepted in the research payload; choose it via `/api/models`
in the same browser session. In Copilot mode, `POST /run-now` also requires the
application key, supplied by its dashboard password field.

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
