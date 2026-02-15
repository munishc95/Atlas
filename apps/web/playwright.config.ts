import { defineConfig, devices } from "@playwright/test";

const apiPort = process.env.PLAYWRIGHT_API_PORT ?? "8000";
const webPort = process.env.PLAYWRIGHT_WEB_PORT ?? "3000";
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? `http://127.0.0.1:${apiPort}`;
const webBase = process.env.PLAYWRIGHT_BASE_URL ?? `http://127.0.0.1:${webPort}`;

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: webBase,
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: `python -m uvicorn app.main:app --host 127.0.0.1 --port ${apiPort}`,
      url: process.env.PLAYWRIGHT_API_HEALTH_URL ?? `${apiBase}/api/health`,
      cwd: "../..",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        PYTHONPATH: "apps/api",
        ATLAS_JOBS_INLINE: process.env.ATLAS_JOBS_INLINE ?? "true",
      },
    },
    {
      command: `pnpm dev -p ${webPort}`,
      url: webBase,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: apiBase,
        NEXT_PUBLIC_FORCE_INLINE_JOBS: process.env.NEXT_PUBLIC_FORCE_INLINE_JOBS ?? "true",
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
