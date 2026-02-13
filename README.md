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
1. Go to `Universe & Data` and import `data/sample/NIFTY500_1d.csv`.
2. Go to `Strategy Lab` and run backtest.
3. Go to `Walk-Forward` and run walk-forward (Optuna trials configurable).
4. Promote strategy to paper.
5. Go to `Paper Trading` and run step.

## API additions in this phase
- `GET /api/jobs`
- `GET /api/strategies`
- `GET /api/paper/state`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/backtests` (leaderboard filters/sort/pagination)
- `GET /api/backtests/compare`
- `GET /api/backtests/{id}/trades/export.csv`
- `GET /api/backtests/{id}/summary/export.json`
- `GET /api/walkforward/{id}/folds`

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

## Developer docs
- `docs/add-strategy-template.md`
- `docs/add-data-provider.md`
