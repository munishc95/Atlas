import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "../..");
const localPython = path.join(
  repoRoot,
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const pythonExecutable =
  process.env.PLAYWRIGHT_PYTHON ?? (existsSync(localPython) ? localPython : "python");
const nextExecutable = path.join(configDir, "node_modules", "next", "dist", "bin", "next");
const apiPort = process.env.PLAYWRIGHT_API_PORT ?? "8000";
const webPort = process.env.PLAYWRIGHT_WEB_PORT ?? "3000";
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? `http://127.0.0.1:${apiPort}`;
const webBase = process.env.PLAYWRIGHT_BASE_URL ?? `http://127.0.0.1:${webPort}`;
const configuredE2eRoot = process.env.ATLAS_E2E_ROOT;
const e2eRoot =
  configuredE2eRoot ??
  path.join(configDir, ".atlas", `e2e-${Date.now()}-${process.pid.toString()}`);
const e2ePaths = {
  ATLAS_ENVIRONMENT: "test",
  ATLAS_DATABASE_URL: `sqlite:///${path.join(e2eRoot, "atlas.db")}`,
  ATLAS_DUCKDB_PATH: path.join(e2eRoot, "ohlcv.duckdb"),
  ATLAS_PARQUET_ROOT: path.join(e2eRoot, "parquet"),
  ATLAS_FEATURE_CACHE_ROOT: path.join(e2eRoot, "features"),
  ATLAS_DATA_INBOX_ROOT: path.join(e2eRoot, "inbox"),
  ATLAS_SECRETS_ROOT: path.join(e2eRoot, "secrets"),
  ATLAS_TRAIN_DATASETS_ROOT: path.join(e2eRoot, "train_datasets"),
  ATLAS_NSE_BHAVCOPY_CACHE_DIR: path.join(e2eRoot, "nse-bhavcopy-cache"),
  ATLAS_CRED_KEY_PATH: path.join(e2eRoot, "secrets", "atlas-cred.key"),
  ATLAS_EVENT_RISK_GENERATED_CALENDAR_PATH: path.join(e2eRoot, "inbox", "event-risk-generated.csv"),
  ATLAS_EVENT_RISK_SYNC_META_PATH: path.join(e2eRoot, "inbox", "event-risk-sync.json"),
  ATLAS_OPTUNA_STORAGE_URL: `sqlite:///${path.join(e2eRoot, "optuna.db")}`,
  ATLAS_REDIS_URL: "redis://127.0.0.1:1/0",
  ATLAS_TELEGRAM_ENABLED: "false",
};

Object.assign(process.env, e2ePaths, {
  ATLAS_E2E_ROOT: e2eRoot,
  ATLAS_E2E_EPHEMERAL_ROOT: configuredE2eRoot ? "0" : "1",
});

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  globalSetup: "./tests/global-setup.ts",
  globalTeardown: "./tests/global-teardown.ts",
  use: {
    baseURL: webBase,
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: `"${pythonExecutable}" -m uvicorn app.main:app --host 127.0.0.1 --port ${apiPort}`,
      url: process.env.PLAYWRIGHT_API_HEALTH_URL ?? `${apiBase}/api/health`,
      cwd: repoRoot,
      reuseExistingServer: false,
      timeout: 180_000,
      env: {
        ...e2ePaths,
        PYTHONPATH: "apps/api",
        ATLAS_JOBS_INLINE: process.env.ATLAS_JOBS_INLINE ?? "true",
        ATLAS_E2E_FAST: process.env.ATLAS_E2E_FAST ?? "1",
        ATLAS_FAST_MODE: process.env.ATLAS_FAST_MODE ?? "1",
      },
    },
    {
      command: `"${process.execPath}" "${nextExecutable}" dev -p ${webPort}`,
      url: webBase,
      cwd: configDir,
      reuseExistingServer: false,
      timeout: 180_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: apiBase,
        NEXT_PUBLIC_FORCE_INLINE_JOBS: process.env.NEXT_PUBLIC_FORCE_INLINE_JOBS ?? "true",
        NEXT_PUBLIC_ATLAS_FAST_MODE: process.env.NEXT_PUBLIC_ATLAS_FAST_MODE ?? "1",
      },
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
