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

## Operate Mode (v1.6)

Atlas now includes an operational monitoring layer focused on trust and explainability:

- Every paper run-step persists a `PaperRun` record with:
  - signal source, scan/evaluation counts, truncation flags
  - selected/skipped reason histograms
  - cost/slippage summary
  - position/order deltas (opened/closed, new order ids)
- Policy health snapshots are computed on rolling windows (default 20d/60d):
  - return, drawdown, Calmar, Sharpe/Sortino, win rate, profit factor, turnover, exposure
  - cost ratio and tail proxy where available
- Deterministic drift rules trigger safe actions:
  - `WARNING`: risk scaling override (default `0.75x`)
  - `DEGRADED`: configurable action (`PAUSE`/`RETIRED`) and degraded risk override (default `0.25x`)
- Fallback behavior:
  - if active policy is paused/retired/degraded for the regime, Atlas selects the next best eligible policy
  - if none for the same regime, Atlas falls back to a `RISK_OFF` policy when available
  - otherwise it runs shadow-only (no new entries)
- Optional daily report generation captures:
  - summary PnL/cost/drawdown
  - explainability (selected/skipped reasons)
  - risk stats and run links for drill-down

Frontend additions:

- Dashboard `Today status` panel with active bundle/policy, latest run-step, and health badge
- New `Reports` page with daily report list, detail drawer, and CSV/JSON export links
- New policy health view at `/policies/{id}` (20d/60d snapshots + baseline deltas)
- Paper Trading shows inline health status and has a `Generate Daily Report` action

## Champion-Challenger + Replay + PDF (v1.7)

Atlas now includes reproducible policy governance and shareable tear sheets:

- Champion-challenger evaluation in shadow mode:
  - runs champion and challengers on the same bundle/window without mutating live paper state
  - deterministic recommendation with explicit reasons and guardrails
  - optional `auto_promote` remains disabled by default
- Deterministic replay runs:
  - replay summary persisted with `seed`, date range, `engine_version`, and `data_digest`
  - same input + seed yields identical ordering/summary outputs
- Professional PDF exports:
  - daily tear sheets include metrics, equity + drawdown chart, explainability reasons
  - monthly tear sheets summarize period PnL/cost/DD and reason histograms
- New UI pages:
  - `Evaluations` for champion-challenger runs and explicit active-policy apply
  - `Replay` for deterministic range replay + JSON export
  - `Reports` now supports daily/monthly PDF download actions

## Single Truth Simulator (v1.8)

Research, walk-forward, replay, and champion-challenger shadow evaluation now run through the same deterministic side/instrument-aware simulator core:

- One execution model for LONG + SHORT across:
  - backtests
  - walk-forward OOS folds + stress runs
  - auto-research candidate scoring
  - replay and shadow evaluations
- Instrument-aware execution:
  - `EQUITY_CASH`, `STOCK_FUT`, `INDEX_FUT`
  - next-bar fills, mirrored long/short stop logic, gap-through-stop handling
  - futures lot sizing + margin reserve/release
- Reproducibility metadata included in outputs:
  - `engine_version`
  - `data_digest`
  - `seed`

This removes execution-assumption drift between research recommendations and paper shadow results.

## Paper-Simulator Parity (v1.9)

Paper execution now supports a simulator adapter path that uses the same deterministic execution core as research, walk-forward, replay, and shadow evaluations.

- Paper run-step can execute through the unified simulator adapter:
  - same next-bar fill, slippage, gap-stop, and cost logic
  - same side/instrument rules for `LONG`/`SHORT` and cash/futures behavior
- Reproducibility metadata now flows into paper outputs:
  - `paper_engine`
  - `engine_version`
  - `data_digest`
  - `seed`
- Runtime toggle (with fallback path still available for migration):
  - setting: `paper_use_simulator_engine`
  - default: `true`
  - set to `false` to temporarily use the legacy paper execution path.

## Data Integrity + Safe Mode (v2.0)

Atlas now includes explicit operate-time guardrails for data integrity and operator confidence:

- Data quality service (`DataQualityReport`) checks bundle/timeframe health:
  - missing bars/gaps, duplicate/non-monotonic timestamps
  - invalid OHLC bounds
  - stale data detection and outlier/corporate-action anomaly warnings
- Operate event stream (`OperateEvent`) captures:
  - data quality WARN/FAIL
  - safe-mode activations
  - policy health actions and evaluation decisions
  - digest mismatch and job exception boundaries
- Safe mode behavior in paper run-step:
  - on quality `FAIL`, Atlas activates safe mode and blocks new entries
  - exits/stop handling remains active (`exits_only`) so risk can be reduced safely
- Reproducibility guard:
  - each paper run stores a `result_digest`
  - rerunning same day/config/seed with unchanged `data_digest` but changed result emits `digest_mismatch`

New APIs:

- `POST /api/data/quality/run`
- `GET /api/data/quality/latest?bundle_id=&timeframe=`
- `GET /api/data/quality/history?bundle_id=&timeframe=&days=`
- `GET /api/operate/events?since=&severity=&category=`
- `GET /api/operate/health`

New frontend page:

- `Ops` page (`/ops`) with current mode (`NORMAL` / `SAFE MODE`), latest quality report, recent operate events, and quick actions.

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
- `POST /api/policies/{id}/set-active`
- `GET /api/policies/health`
- `GET /api/policies/{id}/health?window_days=20|60`
- `POST /api/evaluations/run`
- `GET /api/evaluations`
- `GET /api/evaluations/{id}`
- `GET /api/evaluations/{id}/details`
- `POST /api/paper/signals/preview`
- `GET /api/operate/status`
- `POST /api/replay/run`
- `GET /api/replay/runs`
- `GET /api/replay/runs/{id}`
- `GET /api/replay/runs/{id}/export.json`
- `POST /api/reports/daily/generate`
- `GET /api/reports/daily?date=&bundle_id=&policy_id=`
- `GET /api/reports/daily/{id}`
- `GET /api/reports/daily/{id}/export.json`
- `GET /api/reports/daily/{id}/export.csv`
- `GET /api/reports/daily/{id}/export.pdf`
- `POST /api/reports/monthly/generate`
- `GET /api/reports/monthly?month=&bundle_id=&policy_id=`
- `GET /api/reports/monthly/{id}`
- `GET /api/reports/monthly/{id}/export.json`
- `GET /api/reports/monthly/{id}/export.pdf`

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
