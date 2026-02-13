import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:3000",
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
      url: process.env.PLAYWRIGHT_API_HEALTH_URL ?? "http://127.0.0.1:8000/api/health",
      cwd: "../..",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        PYTHONPATH: "apps/api",
        ATLAS_JOBS_INLINE: process.env.ATLAS_JOBS_INLINE ?? "true",
      },
    },
    {
      command: "pnpm dev",
      url: process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000",
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
