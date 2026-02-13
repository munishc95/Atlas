# AGENTS.md - Atlas (Adaptive Swing Trading Platform)

This file defines working agreements and quality standards for agentic coding (Codex) on this repo.
Follow these rules **strictly**. If something is unclear, make a sensible default and proceed.

---

## 0) Mission

Build **Atlas**: a local-first, production-quality **research + walk-forward + paper-trading** platform for **NIFTY 500** that feels **Apple-like**: smooth, minimal, fast, intuitive.

**Non-goals for MVP**
- No real-money trading.
- No "price prediction" claims.
- No Streamlit.

---

## 1) Golden Product Principles

### 1.1 Capital Protection First
- Backtest realism > pretty curves.
- Always include: next-bar fills, costs, slippage, max positions, ATR risk sizing, kill-switch.
- Prefer simple strategies + strong validation over complexity.

### 1.2 No Overfitting Theater
- Optimization must be validated via walk-forward.
- Never show in-sample metrics as "final performance" without OOS clearly labeled.

### 1.3 UI: Apple-like, Minimal
- Whitespace, typography hierarchy, subtle motion.
- Everything should feel **calm and premium**.
- Avoid clutter: progressive disclosure (details in drawers/modals).

---

## 2) Repo Structure Expectations

Monorepo:
- `apps/web` = Next.js (App Router) + TypeScript + Tailwind + shadcn/ui + Framer Motion
- `apps/api` = FastAPI + Pydantic + SQLModel/SQLAlchemy + Alembic
- `infra/` = docker-compose (postgres, redis)
- `src/` or `packages/` shared libs only if needed (avoid premature abstraction)

Data:
- OHLCV in **Parquet** and query via **DuckDB**
- Metadata/results in **Postgres**

---

## 3) Development Workflow (Always Do)

### 3.1 Small, Safe Steps
- Make incremental changes with frequent local checks.
- Prefer simple implementations that meet requirements.

### 3.2 Before You Commit a Step
- Backend: run tests and type checks
- Frontend: run lint and type checks
- Ensure app still starts cleanly

### 3.3 Never Break the Build
- Keep `main` runnable locally at all times.
- If you must do a large refactor, do it in multiple commits/steps.

---

## 4) Coding Standards

### 4.1 Python (apps/api)
- Use type hints everywhere (mypy-friendly).
- Use Pydantic models for API schemas.
- Keep functions small (= ~40 lines) unless necessary.
- Prefer pure functions for indicator calculations and metrics (testable).
- Add docstrings for core quant logic (backtester, sizing, metrics, regimes).

### 4.2 TypeScript (apps/web)
- Strict TypeScript.
- Prefer server components for data fetching where appropriate.
- Components should be reusable and styled consistently.
- Use `zod` for client-side validation when needed.

### 4.3 Error Handling
- Never swallow errors silently.
- API must return structured errors with `code`, `message`, and optional `details`.
- Frontend must show friendly error states and recovery actions.

---

## 5) Quant Engine Rules (Non-Negotiable)

### 5.1 Backtest Realism
- **Next-bar fills**: signals at bar close, fills at next bar open (or configured).
- **Slippage + commission** always applied.
- **Max concurrent positions** = 3 (configurable).
- **Position sizing** uses ATR-based stop distance:
  - `risk_amount = equity * 0.005`
  - `qty = floor(risk_amount / stop_distance)`
- Support: stop loss, trailing stop, time stop, optional take-profit.

### 5.2 Validation Rules
- Must support **walk-forward** with configurable train/test/step windows.
- Must separate IS vs OOS results and label clearly in UI.
- Must run stress tests:
  - costs x2
  - slippage x2
  - delayed fills (simple simulation)

### 5.3 Strategy Templates (MVP must include)
1) Trend Breakout (Daily filter + 4H-ish breakout + ATR stop/trail)
2) Pullback-in-Trend (Daily trend + 4H-ish RSI oversold pullback)
3) Volatility Squeeze Breakout (BB/KC squeeze + breakout + ATR stop)

---

## 6) UI/UX Design System Rules (Apple-like)

### 6.1 Layout
- Sidebar navigation + top bar.
- Pages use a consistent max width and spacing scale.
- Prefer cards with subtle borders/shadows.

### 6.2 Typography
- Use system font stack.
- Clear hierarchy:
  - Page title: large, bold
  - Section title: medium semibold
  - Body: comfortable line-height
- Avoid dense text blocks; use bullets and small captions.

### 6.3 Motion
- Use Framer Motion sparingly:
  - fade/slide for page transitions
  - subtle hover and expand animations
- No bouncy/cartoon animations.

### 6.4 States
- Every screen must have:
  - loading state (skeleton)
  - empty state (helpful CTA)
  - error state (actionable)

### 6.5 Dark Mode
- Must look excellent in dark and light themes.
- Avoid low-contrast text.

---

## 7) API Design Rules

### 7.1 Endpoints must be stable and predictable
- Use REST for resource reads/writes.
- Use SSE/WebSocket for job progress streaming.

### 7.2 Response shapes
- Success: `{ data: ..., meta?: ... }`
- Error: `{ error: { code, message, details? } }`

### 7.3 Jobs
- Long-running work must be async (queue worker).
- Job model includes:
  - `id`, `type`, `status`, `progress` (0-100), `started_at`, `ended_at`
  - logs as append-only lines
- Frontend shows a job drawer with streamed logs.

---

## 8) Testing Requirements (Minimum)

### 8.1 Backend (pytest)
Must include tests for:
- next-bar fill timing
- slippage/commission application
- ATR sizing calculation
- stop-loss and trailing stop behavior
- max positions enforcement (3)
- metrics sanity check

### 8.2 Frontend
- ESLint + TypeScript checks must pass.
- Add at least one Playwright smoke test:
  - open app
  - run demo backtest
  - verify results render

---

## 9) Documentation Rules

- `README.md` must include:
  - local setup steps
  - docker-compose usage
  - how to import sample data
  - how to run a demo backtest and view results
- Add developer docs:
  - "Add a strategy template"
  - "Add a data provider"

---

## 10) Performance Rules

- Avoid loading huge OHLCV blobs into the browser.
- Aggregate in backend and paginate trades.
- Use DuckDB for OHLCV queries and compute-heavy analytics.
- Cache expensive results (job output persisted in Postgres).

---

## 11) Security & Safety

- Store API keys locally and encrypted (if implemented).
- No leaking secrets in logs or UI.
- Always display disclaimer:
  "Research + paper trading only. Not financial advice."

---

## 12) Decision Defaults (If Unclear)

- Prefer long-only cash equity behavior.
- Prefer next-bar open fills.
- Prefer simple slippage model:
  - `slippage_bps = base_bps + vol_factor * ATR%`
- Prefer conservative risk:
  - enable kill-switch DD at 8% default
  - cooldown 10 trading days

---

## 13) Done Criteria (Per Step)

A step is "done" only if:
- Code compiles, runs, and passes tests/lint.
- UI has loading/empty/error states.
- Core logic is documented and covered by tests where required.
- Changes are small and reviewable.

---

## 14) Communication Style (In PR/Step Summaries)

When you finish a step, provide:
1) What changed (1-5 bullets)
2) How to run locally (commands)
3) Key files added/modified
4) Any known limitations and next steps

---

