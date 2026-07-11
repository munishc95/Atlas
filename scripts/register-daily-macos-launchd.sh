#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Applications/Docker.app/Contents/Resources/bin:${PATH:-}"

LABEL="com.atlas.daily-paper"
START_TIME="18:45"
DATABASE_URL="${ATLAS_DATABASE_URL:-sqlite:///${REPO_ROOT}/apps/api/.atlas/atlas.db}"
BUNDLE_ID="${ATLAS_SCHEDULE_BUNDLE_ID:-}"
UNLOAD_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/register-daily-macos-launchd.sh [options]

Options:
  --label VALUE          launchd label. Default: com.atlas.daily-paper.
  --start-time HH:MM     Local Mac time. Default: 18:45.
  --database-url URL     Atlas DB URL. Default: existing local SQLite DB.
  --bundle-id ID         Optional fixed universe bundle id.
  --unload-only          Stop and remove the LaunchAgent.
  --help                 Show this help.

The registered task runs Monday-Friday and executes shadow-only operate after data refresh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    --start-time)
      START_TIME="$2"
      shift 2
      ;;
    --database-url)
      DATABASE_URL="$2"
      shift 2
      ;;
    --bundle-id)
      BUNDLE_ID="$2"
      shift 2
      ;;
    --unload-only)
      UNLOAD_ONLY=1
      shift
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

if [[ ! "$START_TIME" =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
  echo "--start-time must be HH:MM" >&2
  exit 2
fi
HOUR="${START_TIME%%:*}"
MINUTE="${START_TIME##*:}"
if (( HOUR < 0 || HOUR > 23 || MINUTE < 0 || MINUTE > 59 )); then
  echo "--start-time must be a valid 24-hour local time" >&2
  exit 2
fi

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 or .venv/bin/python is required" >&2
  exit 1
fi

UPDATE_SCRIPT="${REPO_ROOT}/scripts/daily-free-data-update-macos.sh"
if [[ ! -f "$UPDATE_SCRIPT" ]]; then
  echo "Missing update script: $UPDATE_SCRIPT" >&2
  exit 1
fi

LOGS_ROOT="${REPO_ROOT}/data/logs"
mkdir -p "$LOGS_ROOT" "${HOME}/Library/LaunchAgents"

PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

bootout_existing() {
  launchctl bootout "$DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl remove "$LABEL" >/dev/null 2>&1 || true
}

if [[ "$UNLOAD_ONLY" -eq 1 ]]; then
  bootout_existing
  rm -f "$PLIST_PATH"
  echo "Unloaded and removed $PLIST_PATH"
  exit 0
fi

export ATLAS_LAUNCHD_LABEL="$LABEL"
export ATLAS_LAUNCHD_REPO_ROOT="$REPO_ROOT"
export ATLAS_LAUNCHD_UPDATE_SCRIPT="$UPDATE_SCRIPT"
export ATLAS_LAUNCHD_START_HOUR="$HOUR"
export ATLAS_LAUNCHD_START_MINUTE="$MINUTE"
export ATLAS_LAUNCHD_DATABASE_URL="$DATABASE_URL"
export ATLAS_LAUNCHD_BUNDLE_ID="$BUNDLE_ID"
export ATLAS_LAUNCHD_STDOUT="${LOGS_ROOT}/atlas-launchd.out.log"
export ATLAS_LAUNCHD_STDERR="${LOGS_ROOT}/atlas-launchd.err.log"

"$PYTHON_BIN" - "$PLIST_PATH" <<'PY'
from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
repo_root = os.environ["ATLAS_LAUNCHD_REPO_ROOT"]
arguments = [
    "/bin/bash",
    os.environ["ATLAS_LAUNCHD_UPDATE_SCRIPT"],
]
bundle_id = os.environ.get("ATLAS_LAUNCHD_BUNDLE_ID", "").strip()
if bundle_id:
    arguments.extend(["--bundle-id", bundle_id])

hour = int(os.environ["ATLAS_LAUNCHD_START_HOUR"])
minute = int(os.environ["ATLAS_LAUNCHD_START_MINUTE"])
# launchd follows cron semantics: 0 and 7 are Sunday, so 1-5 is Monday-Friday.
weekdays = [1, 2, 3, 4, 5]

plist = {
    "Label": os.environ["ATLAS_LAUNCHD_LABEL"],
    "ProgramArguments": arguments,
    "WorkingDirectory": repo_root,
    "EnvironmentVariables": {
        "PATH": (
            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:"
            "/Applications/Docker.app/Contents/Resources/bin"
        ),
        "PYTHONPATH": f"{repo_root}/apps/api",
        "ATLAS_DATABASE_URL": os.environ["ATLAS_LAUNCHD_DATABASE_URL"],
        "ATLAS_REDIS_URL": os.environ.get("ATLAS_REDIS_URL", "redis://127.0.0.1:6379/0"),
        "ATLAS_PARQUET_ROOT": os.environ.get("ATLAS_PARQUET_ROOT", f"{repo_root}/data/parquet"),
        "ATLAS_DUCKDB_PATH": os.environ.get(
            "ATLAS_DUCKDB_PATH", f"{repo_root}/apps/api/.atlas/ohlcv.duckdb"
        ),
        "ATLAS_FEATURE_CACHE_ROOT": os.environ.get(
            "ATLAS_FEATURE_CACHE_ROOT", f"{repo_root}/data/features"
        ),
        "ATLAS_CALENDAR_DATA_ROOT": os.environ.get(
            "ATLAS_CALENDAR_DATA_ROOT", f"{repo_root}/data/calendars"
        ),
        "ATLAS_DATA_INBOX_ROOT": os.environ.get(
            "ATLAS_DATA_INBOX_ROOT", f"{repo_root}/data/inbox"
        ),
    },
    "StartCalendarInterval": [
        {"Weekday": weekday, "Hour": hour, "Minute": minute} for weekday in weekdays
    ],
    "RunAtLoad": False,
    "StandardOutPath": os.environ["ATLAS_LAUNCHD_STDOUT"],
    "StandardErrorPath": os.environ["ATLAS_LAUNCHD_STDERR"],
    "ProcessType": "Background",
}

with path.open("wb") as handle:
    plistlib.dump(plist, handle, sort_keys=False)
PY

plutil -lint "$PLIST_PATH" >/dev/null
bootout_existing
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"
launchctl enable "${DOMAIN}/${LABEL}"

cat <<EOF
Registered ${LABEL}
Plist: ${PLIST_PATH}
Schedule: Monday-Friday at ${START_TIME} local time
Database: ${DATABASE_URL}
Bundle: ${BUNDLE_ID:-active/latest}
Logs: ${LOGS_ROOT}
EOF
