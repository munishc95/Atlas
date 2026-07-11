#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Applications/Docker.app/Contents/Resources/bin:${PATH:-}"

BUNDLE_ID="${ATLAS_SCHEDULE_BUNDLE_ID:-}"
LOOKBACK_DAYS="${ATLAS_SCHEDULE_LOOKBACK_DAYS:-10}"
CORPORATE_ACTION_LOOKBACK_DAYS="${ATLAS_SCHEDULE_CORPORATE_ACTION_LOOKBACK_DAYS:-180}"
EVENT_RISK_LOOKBACK_DAYS="${ATLAS_SCHEDULE_EVENT_RISK_LOOKBACK_DAYS:-30}"
EVENT_RISK_FORWARD_DAYS="${ATLAS_SCHEDULE_EVENT_RISK_FORWARD_DAYS:-120}"
THROTTLE_SECONDS="${ATLAS_SCHEDULE_THROTTLE_SECONDS:-0.05}"
RUN_QUALITY=0
SKIP_EVENT_RISK=0
RUN_OPERATE=1
OPERATE_TIMEFRAME="${ATLAS_SCHEDULE_OPERATE_TIMEFRAME:-1d}"
OPERATE_REGIME="${ATLAS_SCHEDULE_OPERATE_REGIME:-TREND_UP}"
OPERATE_MAX_RUNTIME_SECONDS="${ATLAS_SCHEDULE_OPERATE_MAX_RUNTIME_SECONDS:-10800}"

usage() {
  cat <<'EOF'
Usage: scripts/daily-free-data-update-macos.sh [options]

Options:
  --bundle-id ID                         Universe bundle id. Defaults to active/latest bundle.
  --lookback-days N                      Bhavcopy lookback days. Default: 10.
  --corporate-action-lookback-days N     Corporate-action lookback days. Default: 180.
  --event-risk-lookback-days N           Event-risk lookback days. Default: 30.
  --event-risk-forward-days N            Event-risk forward days. Default: 120.
  --throttle-seconds N                   NSE request throttle. Default: 0.05.
  --run-quality                          Run data quality after bhavcopy import.
  --skip-event-risk                      Skip event-risk sync.
  --no-operate                           Refresh data only.
  --timeframe VALUE                      Operate timeframe. Default: 1d.
  --regime VALUE                         Operate regime. Default: TREND_UP.
  --help                                 Show this help.

The operate step is always shadow-only. It does not place real-money orders.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-id)
      BUNDLE_ID="$2"
      shift 2
      ;;
    --lookback-days)
      LOOKBACK_DAYS="$2"
      shift 2
      ;;
    --corporate-action-lookback-days)
      CORPORATE_ACTION_LOOKBACK_DAYS="$2"
      shift 2
      ;;
    --event-risk-lookback-days)
      EVENT_RISK_LOOKBACK_DAYS="$2"
      shift 2
      ;;
    --event-risk-forward-days)
      EVENT_RISK_FORWARD_DAYS="$2"
      shift 2
      ;;
    --throttle-seconds)
      THROTTLE_SECONDS="$2"
      shift 2
      ;;
    --run-quality)
      RUN_QUALITY=1
      shift
      ;;
    --skip-event-risk)
      SKIP_EVENT_RISK=1
      shift
      ;;
    --no-operate)
      RUN_OPERATE=0
      shift
      ;;
    --timeframe)
      OPERATE_TIMEFRAME="$2"
      shift 2
      ;;
    --regime)
      OPERATE_REGIME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 or .venv/bin/python is required" >&2
  exit 1
fi

DEFAULT_SQLITE_URL="sqlite:///${REPO_ROOT}/apps/api/.atlas/atlas.db"
export ATLAS_DATABASE_URL="${ATLAS_DATABASE_URL:-$DEFAULT_SQLITE_URL}"
export ATLAS_REDIS_URL="${ATLAS_REDIS_URL:-redis://127.0.0.1:6379/0}"
export ATLAS_PARQUET_ROOT="${ATLAS_PARQUET_ROOT:-${REPO_ROOT}/data/parquet}"
export ATLAS_DUCKDB_PATH="${ATLAS_DUCKDB_PATH:-${REPO_ROOT}/apps/api/.atlas/ohlcv.duckdb}"
export ATLAS_FEATURE_CACHE_ROOT="${ATLAS_FEATURE_CACHE_ROOT:-${REPO_ROOT}/data/features}"
export ATLAS_CALENDAR_DATA_ROOT="${ATLAS_CALENDAR_DATA_ROOT:-${REPO_ROOT}/data/calendars}"
export ATLAS_DATA_INBOX_ROOT="${ATLAS_DATA_INBOX_ROOT:-${REPO_ROOT}/data/inbox}"
export PYTHONPATH="${REPO_ROOT}/apps/api"

LOGS_ROOT="${REPO_ROOT}/data/logs"
mkdir -p "$LOGS_ROOT"
RUN_STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_PATH="${LOGS_ROOT}/daily-free-data-update-macos-${RUN_STAMP}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

ensure_docker_stack_if_needed() {
  case "$ATLAS_DATABASE_URL" in
    postgresql:*|postgresql+*)
      ;;
    *)
      log "Using local SQLite state; Docker is not required for this scheduled inline run."
      return 0
      ;;
  esac

  if ! command -v docker >/dev/null 2>&1; then
    log "Docker CLI is required for Postgres database URL: $ATLAS_DATABASE_URL"
    return 1
  fi

  if ! docker info >/dev/null 2>&1; then
    log "Docker daemon is not running; opening Docker Desktop."
    open -a Docker >/dev/null 2>&1 || true
    for _ in {1..60}; do
      if docker info >/dev/null 2>&1; then
        break
      fi
      sleep 2
    done
  fi

  docker info >/dev/null 2>&1
  log "Starting Atlas Docker infra."
  docker compose -f "${REPO_ROOT}/infra/docker-compose.yml" up -d
}

resolve_bundle_id() {
  if [[ -n "$BUNDLE_ID" ]]; then
    printf '%s\n' "$BUNDLE_ID"
    return 0
  fi

  "$PYTHON_BIN" - <<'PY'
from sqlmodel import Session

from app.core.config import get_settings
from app.db.bootstrap import seed_defaults
from app.db.models import PaperState
from app.db.session import engine, init_db
from app.services.operate_context import resolve_active_bundle_id

settings = get_settings()
init_db()
with Session(engine) as session:
    seed_defaults(session, settings)
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}
    bundle_id = resolve_active_bundle_id(session, state_settings=state_settings)
    if bundle_id is None:
        raise SystemExit("No dataset bundle found. Create/import a universe bundle first.")
    print(int(bundle_id))
PY
}

run_python() {
  log "python $*"
  "$PYTHON_BIN" "$@"
}

read -r START_DATE CORPORATE_ACTION_START_DATE EVENT_RISK_START_DATE EVENT_RISK_END_DATE END_DATE < <(
  "$PYTHON_BIN" - "$LOOKBACK_DAYS" "$CORPORATE_ACTION_LOOKBACK_DAYS" "$EVENT_RISK_LOOKBACK_DAYS" "$EVENT_RISK_FORWARD_DAYS" <<'PY'
from datetime import date, timedelta
import sys

lookback = max(1, int(sys.argv[1]))
corporate = max(1, int(sys.argv[2]))
event_back = max(1, int(sys.argv[3]))
event_forward = max(1, int(sys.argv[4]))
today = date.today()
print(
    (today - timedelta(days=lookback)).isoformat(),
    (today - timedelta(days=corporate)).isoformat(),
    (today - timedelta(days=event_back)).isoformat(),
    (today + timedelta(days=event_forward)).isoformat(),
    today.isoformat(),
)
PY
)

cd "$REPO_ROOT"
ensure_docker_stack_if_needed
BUNDLE_ID="$(resolve_bundle_id)"

IMPORT_ARGS=(
  "scripts/free_nse_bhavcopy_backfill.py"
  "--bundle-id" "$BUNDLE_ID"
  "--start-date" "$START_DATE"
  "--end-date" "$END_DATE"
  "--throttle-seconds" "$THROTTLE_SECONDS"
)
if [[ "$RUN_QUALITY" -eq 0 ]]; then
  IMPORT_ARGS+=("--skip-quality")
fi

log "Atlas free NSE daily update"
log "Repo: $REPO_ROOT"
log "Database: $ATLAS_DATABASE_URL"
log "Bundle: $BUNDLE_ID"
log "Window: $START_DATE to $END_DATE"
log "Corporate action window: $CORPORATE_ACTION_START_DATE to $END_DATE"
if [[ "$SKIP_EVENT_RISK" -eq 0 ]]; then
  log "Event risk window: $EVENT_RISK_START_DATE to $EVENT_RISK_END_DATE"
fi
log "Log: $LOG_PATH"

run_python "${IMPORT_ARGS[@]}"
run_python \
  "scripts/free_nse_corporate_actions_import.py" \
  "--bundle-id" "$BUNDLE_ID" \
  "--start-date" "$CORPORATE_ACTION_START_DATE" \
  "--end-date" "$END_DATE" \
  "--mode" "UPSERT"

if [[ "$SKIP_EVENT_RISK" -eq 0 ]]; then
  run_python \
    "scripts/free_event_risk_sync.py" \
    "--bundle-id" "$BUNDLE_ID" \
    "--start-date" "$EVENT_RISK_START_DATE" \
    "--end-date" "$EVENT_RISK_END_DATE"
fi

if [[ "$RUN_OPERATE" -eq 1 ]]; then
  log "Operate run: enabled ($OPERATE_TIMEFRAME, $OPERATE_REGIME, shadow-only)"
  run_python \
    "scripts/run_operate_inline.py" \
    "--bundle-id" "$BUNDLE_ID" \
    "--timeframe" "$OPERATE_TIMEFRAME" \
    "--regime" "$OPERATE_REGIME" \
    "--date" "$END_DATE" \
    "--source" "macos_launchd_daily_free_data_task" \
    "--max-runtime-seconds" "$OPERATE_MAX_RUNTIME_SECONDS" \
    "--include-data-updates" \
    "--shadow-only" \
    "--skip-if-auto-run-date-marked" \
    "--mark-auto-run-date"
fi

log "Atlas free NSE daily update finished"
