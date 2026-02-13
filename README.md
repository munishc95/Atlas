# Atlas

Adaptive Swing Trading Research + Walk-Forward + Paper Trading platform for NIFTY 500.

Disclaimer: This tool is for research and paper trading. Not financial advice. Trading involves risk. Past performance does not guarantee future results.

## Monorepo

- `apps/web`: Next.js App Router, TypeScript, Tailwind, TanStack Query, Playwright
- `apps/api`: FastAPI, SQLModel, RQ worker jobs, Optuna walk-forward tuning
- `infra/docker-compose.yml`: Postgres + Redis
- `data/sample`: sample OHLCV CSV files

## Local setup

### 1) Start infra

```powershell
docker compose -f infra/docker-compose.yml up -d
```

If Postgres `5432` is already used, change host mapping in `infra/docker-compose.yml` (example `5433:5432`).

### 2) Backend install

```powershell
python -m pip install -e "apps/api[dev]" --index-url https://pypi.org/simple
```

### 3) Start API

```powershell
$env:PYTHONPATH="apps/api"
python -m uvicorn app.main:app --reload --app-dir apps/api --host 127.0.0.1 --port 8000
```

### 4) Start RQ worker (required for queued jobs)

```powershell
$env:PYTHONPATH="apps/api"
python -m app.worker
```

### 5) Frontend install + run

```powershell
pnpm install
pnpm -C apps/web dev
```

### 6) Open app

- Web: `http://localhost:3000`
- API health: `http://127.0.0.1:8000/api/health`

### Optional: one-command stack launcher (Windows)

```powershell
pnpm dev:stack
```

Includes API + worker + web terminals and Docker infra startup.

If you want launcher to install missing Playwright Chromium:

```powershell
pnpm dev:stack:e2e
```

## Job system

Long-running APIs enqueue RQ jobs and return immediately with `job_id`:

- `POST /api/data/import`
- `POST /api/backtests/run`
- `POST /api/walkforward/run`
- `POST /api/paper/run-step`

Idempotent enqueue support:

- Send `Idempotency-Key` header with same payload to deduplicate accidental retries and reuse existing `job_id`.

Job progress stream:

- `GET /api/jobs/{job_id}/stream` (SSE: `progress`, `log`, `heartbeat` events)

## Sample workflow

1. Go to `Universe & Data`, select/create a universe bundle, and import `data/sample/NIFTY500_1d.csv`.
2. Import futures series for short support (example `data/sample/NIFTY500_FUT_1d.csv`) with:
   - `instrument_kind=STOCK_FUT`
   - `underlying=NIFTY500`
   - `lot_size` set (required for futures)
3. Go to `Strategy Lab` and run backtest.
4. Go to `Walk-Forward` and run walk-forward (Optuna trials configurable).
5. Go to `Auto Research` and run multi-template robust scan.
6. Create a policy from the research run and click `Use in Paper`.
7. Go to `Paper Trading`, enable `Autopilot`, click `Preview Signals`, then `Run Step`.

## Auto Research workflow

`Auto Research` runs systematic, explainable strategy evaluation:

- Walk-forward optimization with Optuna for each candidate tuple `(symbol, timeframe, strategy template)`
- OOS-first robustness scoring with deterministic penalties and hard gates
- Stress test evaluation and parameter stability checks
- Ranked candidates + policy preview (`regime -> strategy -> params -> risk scaling`)

From the UI:

1. Open `Auto Research`.
2. Set timeframes, templates, trial/symbol budgets, and gating constraints.
3. Click `Run Auto Research` and monitor progress via SSE logs.
4. Review candidates and explanations.
5. Click `Create Policy`, then `Use in Paper`.

## Paper Trading Autopilot (v1.5)

`POST /api/paper/run-step` now supports autonomous signal generation:

- If `auto_generate_signals=true`, Atlas generates signals from the active policy.
- If paper mode is `policy` and `signals` is empty, generation is automatic.
- Generation is bundle-scoped (policy universe bundle), deterministic, ranked, and explainable.
- Feature cache is used for indicator reads (`data/features`) with incremental refresh.
- Strategy templates are direction-aware (`long`, `short`, `both`) and can emit native SELL signals.
- For SELL candidates, execution prefers `STOCK_FUT` when available in the selected bundle.

Response now includes:

- `signals_source` (`provided` or `generated`)
- `generated_signals_count`
- `selected_signals_count`
- `selected_signals[]` and `skipped_signals[]` with reasons
- `policy_mode`, `policy_selection_reason`, and `cost_summary`
- `scan_truncated`, `scanned_symbols`, `evaluated_candidates`, `total_symbols`

Preview endpoint (no orders placed):

- `POST /api/paper/signals/preview`

Paper Trading UI includes:

- Autopilot toggle
- Preview Signals button
- Why drawer with selected/skipped reasons and cost summary
- Shortcuts: `Ctrl/Cmd+Enter` (Run Step), `P` (Preview)

## Universe Bundles (first-class scope)

`DatasetBundle` is the explicit source of truth for universe membership:

- Imports can be assigned to a bundle from `Universe & Data`.
- Auto Research runs anchor to `bundle_id` (with compatibility fallback to `dataset_id`).
- Paper Autopilot previews/runs can use `bundle_id` directly.
- Symbol scans are deterministic and can be limited to top liquid names by ADV.

## Short support (India-correct v1.5)

- `allowed_sides` runtime setting controls allowed entry sides (default: `["BUY"]`).
- Cash equity shorts (`EQUITY_CASH`) are intraday only:
  - open short positions are marked `must_exit_by_eod=true`
  - auto square-off is enforced after cutoff (`paper_short_squareoff_time`, default `15:20 IST`)
  - forced exits are logged in audit trail.
- Swing shorts use futures when available:
  - import continuous futures data locally (`*_FUT`) and map `underlying` + `lot_size`
  - paper engine sizes futures in lots (`qty_lots`) and reserves margin (`futures_initial_margin_pct`)
  - futures positions can be held overnight (no EOD forced square-off)
- Instrument selection rule for SELL:
  - prefer `STOCK_FUT` in selected bundle
  - fallback to intraday `EQUITY_CASH` short (if allowed)
  - otherwise skip with explicit reason (`no_short_instrument_available` / `shorts_disabled`)

## Universe/Bundles and futures mapping

- `DatasetBundle` is the deterministic universe boundary for research + autopilot scanning.
- Bundle membership can include both cash symbols and futures symbols.
- Futures import requires:
  - valid OHLCV columns
  - `instrument_kind=STOCK_FUT` (or `INDEX_FUT`)
  - positive `lot_size`
  - optional `underlying` (defaults from `_FUT` suffix convention).

## India-lite transaction costs

A configurable cost model is available for both backtester and paper execution:

- Delivery and intraday estimators (`brokerage`, `STT`, exchange charges, SEBI, stamp, GST)
- Futures estimator for derivative instruments (configurable rates)
- Controlled via runtime settings (`/api/settings`)
- Applied directly to paper cash flows and included in run-step `cost_summary`

## API additions in this phase

- `GET /api/jobs`
- `GET /api/strategies`
- `GET /api/universes`
- `GET /api/paper/state`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/backtests` (leaderboard filters/sort/pagination)
- `GET /api/backtests/compare`
- `GET /api/backtests/{id}/trades/export.csv`
- `GET /api/backtests/{id}/summary/export.json`
- `GET /api/walkforward/{id}/folds`
- `POST /api/research/run`
- `GET /api/research/runs`
- `GET /api/research/runs/{id}`
- `GET /api/research/runs/{id}/candidates`
- `POST /api/policies`
- `GET /api/policies`
- `GET /api/policies/{id}`
- `POST /api/policies/{id}/promote-to-paper`
- `POST /api/paper/signals/preview`

## Testing

### Backend

```powershell
$env:PYTHONPATH="apps/api"
python -m pytest apps/api/tests -q
```

### Frontend

```powershell
pnpm -C apps/web lint
pnpm -C apps/web typecheck
```

### E2E Setup (Playwright)

Check or install browser binaries:

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/setup-e2e.ps1
# or install immediately:
powershell -ExecutionPolicy Bypass -File ./scripts/setup-e2e.ps1 -Install
```

Install directly via package script:

```powershell
pnpm -C apps/web test:e2e:setup
```

Linux CI runner (if deps are missing):

```bash
pnpm -C apps/web exec playwright install --with-deps chromium
```

Run E2E smoke:

```powershell
# Ensure API is running first (worker optional if ATLAS_JOBS_INLINE=true)
pnpm -C apps/web test:e2e
```

Playwright test command auto-starts a clean Next dev server via Playwright `webServer`.

### Formatting and line endings

```powershell
pnpm format
```

This runs:

- Backend formatting and autofixes via Ruff
- Frontend formatting via Prettier
- Docs formatting for markdown files

Repository guardrails:

- `.gitattributes` enforces LF for code/config/docs files
- `.editorconfig` enforces UTF-8, LF, and whitespace defaults

## CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

- `backend-tests`: installs backend deps and runs `pytest`
- `frontend-checks`: installs frontend deps, runs lint + typecheck
- `e2e`: installs both stacks, installs Playwright Chromium, starts API, runs E2E smoke
- Failure logs are uploaded as artifacts for troubleshooting

## Developer docs

- `docs/add-strategy-template.md`
- `docs/add-data-provider.md`
