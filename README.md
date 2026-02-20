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

## Shadow-Only Safe Mode + Local Scheduler (v2.1)

Atlas v2.1 adds operator-safe automation while keeping live paper state protected:

- True `shadow_only` safe mode:
  - when data quality fails and `operate_safe_mode_action=shadow_only`, Atlas runs full candidate selection + simulator execution
  - writes results to `PaperRun` with `mode=SHADOW`
  - persists separate `ShadowPaperState` per `(bundle_id, policy_id)`
  - does **not** mutate live `PaperState`/positions/orders/cash
- Daily reports now carry execution context:
  - `summary.mode` (`LIVE` or `SHADOW`)
  - `summary.shadow_note` for simulated-only runs
- Stale-data policy is configurable:
  - `operate_mode`: `offline` or `live`
  - `data_quality_stale_severity`: `WARN` or `FAIL`
  - in `live` mode, stale defaults to `FAIL` unless explicitly overridden
- Local scheduler in worker:
  - reads runtime settings `operate_auto_run_enabled` and `operate_auto_run_time_ist`
  - triggers on trading days (IST) once per day:
    1) data quality run
    2) paper run-step
    3) daily report generation
  - deduplicates with `operate_last_auto_run_date`
- Ops page shows scheduler status:
  - auto-run enabled/time
  - next scheduled run (IST)

## NSE Trading Calendar + Scheduler Upgrade (v2.2)

Atlas now uses a calendar-aware NSE trading day service (equities by default):

- New shared calendar module: `apps/api/app/services/trading_calendar.py`
  - `is_trading_day()`
  - `next_trading_day()`
  - `previous_trading_day()`
  - `get_session()` (open/close, special session flag, label)
- Local-first holiday files:
  - `data/calendars/nse_equities_holidays_2026.json`
  - `data/calendars/nse_equities_holidays_2027.json`
  - supports `special_sessions` (including weekend sessions such as Muhurat)
- Scheduler now uses the calendar (not weekday-only logic):
  - runs on special trading sessions even on weekends
  - skips exchange holidays
  - dedupes by trading date via `operate_last_auto_run_date`
- Data quality daily gap checks now use trading days from the same calendar (holiday-aware).
- Ops health/status now include:
  - calendar segment
  - today trading-day status
  - session window and special-session label
  - next/previous trading day

Optional calendar refresh tool:

```bash
python -m app.tools.refresh_calendar --year 2026 --segment EQUITIES
python -m app.tools.refresh_calendar --year 2026 --segment EQUITIES --from-file data/calendars/nse_equities_holidays_2026.json
```

If NSE fetch fails, Atlas keeps the existing local file intact.

## Automated Data Updates + Universe Drift Control (v2.3)

Atlas now supports local-first automated data refresh from a drop-folder inbox:

- Inbox convention:
  - `data/inbox/<bundle>/<timeframe>/*.csv|*.parquet`
  - example: `data/inbox/NIFTY500/1d/2026-02-14_bhavcopy.csv`
- Incremental + idempotent ingestion:
  - file hash dedupe (`same file twice -> skipped`)
  - per-file validation and per-run summary records
  - append/merge into existing Parquet without duplicate bars
- Persisted update records:
  - `DataUpdateRun`
  - `DataUpdateFile`

Scheduler auto-run order is now:

1. provider updates (if enabled)
2. inbox data updates (if enabled)
3. data quality
4. paper run-step
5. daily report

New runtime settings:

- `operate_auto_run_include_data_updates` (default `true`)
- `data_updates_inbox_enabled` (default `true`)
- `data_updates_max_files_per_run` (default `50`)
- `data_updates_provider_enabled` (default `false`)
- `data_updates_provider_kind` (default `UPSTOX`)
- `data_updates_provider_max_symbols_per_run`
- `data_updates_provider_max_calls_per_run`
- `data_updates_provider_timeframe_enabled` (default `1d`)
- `coverage_missing_latest_warn_pct`
- `coverage_missing_latest_fail_pct`
- `coverage_inactive_after_missing_days`

Universe drift controls:

- Coverage uses the NSE trading calendar to determine expected latest trading day.
- Missing latest bars trigger WARN/FAIL by configured thresholds.
- Symbols missing bars for configured trading-day window are marked inactive for live selection.

Frontend additions:

- `Universe & Data` page includes `Data Updates` controls and a coverage details drawer.
- `Ops` page shows latest data update status and quick action to run updates.

## Upstox Provider Updates + Inbox Fallback (v2.7)

Atlas now supports optional provider-driven OHLCV refresh with local-first fallback:

- Formal provider interface in `apps/api/app/providers`:
  - `list_symbols(bundle_id)`
  - `fetch_ohlc(symbols, timeframe, start, end)`
  - `supports_timeframes()`
- New provider implementations:
  - `UpstoxProvider` for automated historical candle pulls
  - `MockProvider` for deterministic fast-mode and test runs
- Upstox mapping table:
  - `InstrumentMap` persists `(provider, symbol, instrument_key, last_refreshed)`
  - optional env seed map via `ATLAS_UPSTOX_SYMBOL_MAP_JSON`
- Provider update persistence:
  - `ProviderUpdateRun`
  - `ProviderUpdateItem`
- Idempotent merge/upsert behavior:
  - bars are deduplicated by timestamp per `(symbol, timeframe)`
  - existing inbox flow remains unchanged
- Rate-limit-aware fetch behavior:
  - retry with exponential backoff on 429/5xx
  - per-run max API call cap
  - event emission for partial/failure outcomes
- Fast mode behavior:
  - symbol scan is hard-capped in provider updates as well
  - deterministic ordering remains seed-based

UI updates:

- `Universe & Data` includes:
  - provider updates enable/disable toggle
  - run provider update action
  - latest provider update status/calls/bars
- `Ops` includes:
  - latest provider update status + duration + API calls
  - quick action to run provider updates

Scheduler/operate order with provider enabled:

1. provider updates
2. inbox updates
3. data quality
4. paper run-step
5. daily report

### Connect Upstox (OAuth)

Atlas now provides a first-class **Connect Upstox** flow in UI:

1. Open `Settings` -> `Providers - Upstox`.
2. Click **Connect** (redirects to Upstox auth page).
3. After approval, Upstox redirects to:
   - `http://localhost:3000/providers/upstox/callback`
4. Atlas exchanges the code and stores token securely.

#### Required `.env` values

Either naming style works:

```env
ATLAS_UPSTOX_CLIENT_ID=...
ATLAS_UPSTOX_CLIENT_SECRET=...
ATLAS_UPSTOX_REDIRECT_URI=http://localhost:3000/providers/upstox/callback
```

or:

```env
ATLAS_UPSTOX_API_KEY=...
ATLAS_UPSTOX_API_SECRET=...
ATLAS_UPSTOX_REDIRECT_URI=http://localhost:3000/providers/upstox/callback
```

#### Token storage (secure local default)

- Primary storage: encrypted local credential store (`ProviderCredential` table).
- Encryption key source:
  - `ATLAS_CRED_KEY`, or
  - auto-generated `data/secrets/atlas_cred.key` (gitignored).
- Optional fallback write to `.env`:
  - controlled by `ATLAS_UPSTOX_PERSIST_ENV_FALLBACK` (default `false`).

#### Token lifecycle

- Upstox access tokens expire; Atlas exposes expiry and verification status.
- No refresh-token flow is implemented in Atlas currently.
- Atlas v3.3 adds **Auto-Renew via Access Token Request + Notifier Webhook**:
  1. Atlas requests a token approval window from Upstox.
  2. You approve in Upstox.
  3. Upstox posts the access token to Atlas notifier webhook.
- Reconnect flow is still available, but auto-renew reduces daily manual code-exchange friction.

#### Upstox Auto-Renew + Webhook Observability (v3.4-v3.5)

- Settings page now includes:
  - auto-renew toggle/time/expiry threshold
  - **Request token now**
  - **Send Test Webhook**
  - **Generate Ping URL** reachability check
  - notifier health + callback timestamp + request history
- Ops page includes one-click **Renew Upstox Token Now** with:
  - pending-request reuse when still valid
  - 60s status auto-refresh flow (request + token + notifier status)
  - copy-ready notifier URL and short approval instructions
- Preferred notifier route is now secret-path based:
  - `POST /api/providers/upstox/notifier/{secret}`
- Legacy route still works (marked less secure):
  - `POST /api/providers/upstox/notifier`
- Atlas persists webhook deliveries for diagnostics and dedupes by payload digest.
- Notifier handler is fail-safe:
  - always responds quickly with 2xx
  - validates client/nonce/secret when present
  - never returns or logs raw access tokens
  - applies local rate limiting and emits `upstox_notifier_rate_limited` warnings when flooded
- Webhook reachability diagnostics:
  - Atlas can generate a temporary ping URL
  - open the ping URL in a browser to confirm your tunnel reaches Atlas
  - inspect ping status (`SENT` / `RECEIVED` / `EXPIRED`) from Settings

Local-first notifier setup for development:

1. Start API locally on `http://127.0.0.1:8000`.
2. Open a tunnel:
   - `ngrok http 8000`
   - or `cloudflared tunnel --url http://127.0.0.1:8000`
3. In Atlas Settings -> Providers -> Upstox, copy the **recommended notifier URL** (secret path).
4. In Upstox My Apps, set notifier URL to that copied URL.
5. Click **Request token now**, approve in Upstox, keep tunnel running until callback arrives.
6. If callbacks still do not arrive:
   - verify tunnel URL is reachable
   - try another public endpoint (for debugging) like webhook.site
   - contact Upstox support for possible domain allowlisting.

Optional CLI helper:

```powershell
$env:PYTHONPATH=\"apps/api\"
python -m app.tools.upstox_auto_renew status
python -m app.tools.upstox_auto_renew request --source cli
```

#### API endpoints

- `GET /api/providers/upstox/auth-url`
- `POST /api/providers/upstox/token/exchange`
- `POST /api/providers/upstox/token/request`
- `POST /api/providers/upstox/token/renew`
- `GET /api/providers/upstox/notifier/status`
- `POST /api/providers/upstox/notifier/test`
- `POST /api/providers/upstox/notifier/ping`
- `GET /api/providers/upstox/notifier/ping/{ping_id}`
- `GET /api/providers/upstox/notifier/ping/{ping_id}/status`
- `GET /api/providers/upstox/notifier/events?limit=&offset=`
- `POST /api/providers/upstox/notifier/{secret}` (recommended)
- `POST /api/providers/upstox/notifier`
- `GET /api/providers/upstox/token/requests/latest`
- `GET /api/providers/upstox/token/requests/history`
- `GET /api/providers/upstox/token/status`
- `GET /api/providers/upstox/token/verify`
- `POST /api/providers/upstox/disconnect`

#### Safety integration

- Provider updates fail gracefully when token missing/expired:
  - `provider_token_missing`
  - `provider_token_expired`
- Operate pipeline token-invalid behavior is configurable:
  - `SKIP` (default): provider stage skipped with `provider_stage_status=SKIPPED_TOKEN_INVALID`, then continue updates/quality/paper/report.
  - `FAIL`: abort operate run at provider stage (strict mode).
- Scheduler can request renewal ahead of next session open:
  - `upstox_auto_renew_lead_hours_before_open` (default 10h).

## Instrument Map Manager + Provider Repair/Backfill + Optional 4H-ish (v2.8)

Atlas now adds local-first mapping management and self-healing provider updates:

- Instrument map manager (no env JSON editing required):
  - import mappings from local file (`csv`/`json`) via API
  - recommended drop-folder path:
    - `data/inbox/_metadata/upstox_instruments.csv`
  - persisted import audit model:
    - `MappingImportRun`
  - mapping health status:
    - mapped count, missing count, sample missing symbols
- Provider update repair/backfill:
  - calendar-aware missing trading-day detection
  - deterministic repair of last `N` trading days each run
  - backfill truncation guardrail (`provider_backfill_max_days`)
  - missing map symbols are skipped with explicit reason:
    - `missing_instrument_map`
  - run summary includes:
    - `repaired_days_used`
    - `missing_days_detected`
    - `backfill_truncated`
  - per-symbol item includes:
    - `bars_added`
    - `bars_updated`
- Optional provider `4h_ish` support:
  - intraday fetch + resample to fixed windows:
    - `09:15-13:15`
    - `13:15-15:30`
  - incomplete-day guardrail:
    - warns and skips partial day by default
    - optional override via `data_updates_provider_allow_partial_4h_ish`
- Fast mode guardrails for intraday provider runs:
  - aggressive symbol/day caps for deterministic quick smoke checks
  - defaults:
    - `fast_mode_provider_intraday_max_symbols=3`
    - `fast_mode_provider_intraday_max_days=2`

## Portfolio Risk Overlay + One-Button Operate (v2.4)

Atlas now supports a portfolio-level risk overlay on top of per-trade sizing:

- Realized portfolio vol targeting:
  - computes rolling realized vol from recent `PaperRun` returns
  - applies deterministic risk scale: `scale = clamp(target_vol / realized_vol, min_scale, max_scale)`
  - persists `PortfolioRiskSnapshot` for each run-step
- Exposure caps enforced in the simulator before accepting entries:
  - gross exposure cap
  - single-name exposure cap
  - sector exposure cap
  - optional correlation clamp with configurable reduction factor
- Explicit skip reasons for explainability:
  - `risk_overlay_gross_exposure_cap`
  - `risk_overlay_single_name_cap`
  - `risk_overlay_sector_cap`
  - `risk_overlay_corr_clamp`
- Paper run-step outputs now include:
  - `risk_overlay.risk_scale`
  - `risk_overlay.realized_vol`
  - `risk_overlay.target_vol`
  - `risk_overlay.caps_applied`

One-button Ops orchestration:

- New endpoint: `POST /api/operate/run`
- Runs in order:
  1. data updates (optional, settings-driven)
  2. data quality
  3. paper run-step
  4. daily report generation
- Returns a job with compact summary including step order, statuses, and generated report id.
- Ops page now has primary action:
  - `Run Today (Updates -> Quality -> Step -> Report)`
  - completion summary includes report id and direct PDF download.

## Closed-Loop Learning + Controlled Switching (v2.5)

Atlas now supports scheduled, explainable policy governance:

- Auto-evaluation loop:
  - compares active policy vs challengers on a rolling trading-day lookback
  - deterministic recommendation: `KEEP`, `SWITCH`, or `SHADOW_ONLY`
  - stores full audit trail in `AutoEvalRun` (`reasons_json`, `score_table_json`, `digest`)
- Safety gates before switching:
  - minimum sample/trade gates
  - score-margin + max drawdown tolerance gates
  - cooldown in trading days
  - max switches in rolling 30 days
  - data quality and safe-mode shadow-only gate checks
- Controlled switching:
  - auto-switch is off by default (`operate_auto_eval_auto_switch=false`)
  - if enabled, switching only executes in live mode and only when all gates pass
  - every switch is persisted to `PolicySwitchEvent`
- Scheduler integration:
  - calendar-aware auto-evaluation scheduling (`DAILY` or `WEEKLY`)
  - dedupe key by trading date (`YYYY-MM-DD::AUTO_EVAL`)
  - tracked via `operate_last_auto_eval_date`
- Ops UI:
  - new `Learning` section with latest recommendation, `Run Evaluation Now`, and `Apply Recommended Switch`
  - switch history drawer for recent policy transitions
- Policy detail UI:
  - shows recent auto-evaluation outcomes and keep/demote reasons

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
- `POST /api/data/updates/run`
- `GET /api/data/updates/latest?bundle_id=&timeframe=`
- `GET /api/data/updates/history?bundle_id=&timeframe=&days=`
- `POST /api/data/provider-updates/run`
- `GET /api/data/provider-updates/latest?bundle_id=&timeframe=`
- `GET /api/data/provider-updates/history?bundle_id=&timeframe=&days=`
- `POST /api/providers/upstox/mapping/import`
- `GET /api/providers/upstox/mapping/status?bundle_id=&timeframe=&sample_limit=`
- `GET /api/providers/upstox/mapping/missing?bundle_id=&timeframe=&limit=`
- `GET /api/data/coverage?bundle_id=&timeframe=&top_n=`
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
- `POST /api/operate/run`
- `POST /api/operate/auto-eval/run`
- `GET /api/operate/auto-eval/history`
- `GET /api/operate/auto-eval/{id}`
- `GET /api/operate/policy-switches`
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
# Fast deterministic smoke (default)
pnpm -C apps/web test:e2e:smoke
```

Run full E2E journey:

```powershell
pnpm -C apps/web test:e2e:full
```

Playwright test command auto-starts a clean Next dev server via Playwright `webServer`.
For deterministic local/CI speed, set:

```powershell
$env:ATLAS_E2E_FAST="1"
$env:ATLAS_FAST_MODE="1"
$env:NEXT_PUBLIC_ATLAS_FAST_MODE="1"
```

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
- `e2e-smoke`: installs both stacks, installs Playwright Chromium, runs fast smoke E2E
- Failure logs are uploaded as artifacts for troubleshooting

Nightly full E2E workflow: `.github/workflows/nightly-e2e-full.yml`

- Runs the full Playwright journey on schedule/manual dispatch
- Uploads Playwright artifacts on completion

## Developer docs

- `docs/add-strategy-template.md`
- `docs/add-data-provider.md`
