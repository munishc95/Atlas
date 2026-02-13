import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";

import { expect, test, type APIRequestContext } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sampleCsv = path.resolve(__dirname, "../../../data/sample/NIFTY500_1d.csv");

async function waitForJob(
  request: APIRequestContext,
  apiBase: string,
  jobId: string,
  timeoutMs = 120_000,
): Promise<Record<string, unknown>> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const statusRes = await request.get(`${apiBase}/api/jobs/${jobId}`);
    const statusBody = await statusRes.json();
    const data = (statusBody?.data ?? {}) as Record<string, unknown>;
    const state = data.status as string | undefined;
    if (state === "SUCCEEDED" || state === "DONE") {
      return data;
    }
    if (state === "FAILED") {
      throw new Error(`Job ${jobId} failed`);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for job ${jobId}`);
}

async function waitForImportReady(
  request: APIRequestContext,
  apiBase: string,
  jobId: string,
): Promise<void> {
  const started = Date.now();
  while (Date.now() - started < 180_000) {
    const statusRes = await request.get(`${apiBase}/api/jobs/${jobId}`);
    const statusBody = await statusRes.json();
    const state = statusBody?.data?.status as string | undefined;
    if (state === "SUCCEEDED" || state === "DONE") {
      return;
    }
    if (state === "FAILED") {
      throw new Error(`Import job ${jobId} failed`);
    }

    const dataStatusRes = await request.get(`${apiBase}/api/data/status`);
    if (dataStatusRes.ok()) {
      const dataStatusBody = await dataStatusRes.json();
      const rows = (dataStatusBody?.data ?? []) as Array<{ symbol?: string; timeframe?: string }>;
      if (rows.some((row) => row.symbol === "NIFTY500" && row.timeframe === "1d")) {
        return;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw new Error(`Timed out waiting for import readiness (job ${jobId})`);
}

test("smoke: import -> backtest -> walk-forward -> auto research -> policy -> paper", async ({
  page,
  request,
}) => {
  test.setTimeout(420_000);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

  const importRes = await request.post(`${apiBase}/api/data/import`, {
    multipart: {
      symbol: "NIFTY500",
      timeframe: "1d",
      provider: "csv",
      file: {
        name: "NIFTY500_1d.csv",
        mimeType: "text/csv",
        buffer: readFileSync(sampleCsv),
      },
    },
  });
  expect(importRes.ok()).toBeTruthy();
  const importBody = await importRes.json();
  await waitForImportReady(request, apiBase, importBody.data.job_id);

  await page.goto("/universe-data");
  await expect(page.getByRole("heading", { name: "Universe & Data" })).toBeVisible({
    timeout: 20_000,
  });

  await page.goto("/strategy-lab");
  await expect(page.getByRole("heading", { name: "Strategy Lab" })).toBeVisible({
    timeout: 20_000,
  });

  const runBacktestRes1 = page.waitForResponse(
    (res) => res.url().includes("/api/backtests/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Backtest" }).click();
  const runBacktestBody1 = await (await runBacktestRes1).json();
  const backtestJob1 = await waitForJob(request, apiBase, runBacktestBody1.data.job_id);
  const backtestResult1 = (backtestJob1.result_json ?? {}) as Record<string, unknown>;
  const backtestId1 = Number(backtestResult1.backtest_id);
  expect(Number.isFinite(backtestId1) && backtestId1 > 0).toBeTruthy();

  const csvRes = await request.get(`${apiBase}/api/backtests/${backtestId1}/trades/export.csv`);
  expect(csvRes.ok()).toBeTruthy();
  expect(await csvRes.text()).toContain("symbol");

  const jsonRes = await request.get(`${apiBase}/api/backtests/${backtestId1}/summary/export.json`);
  expect(jsonRes.ok()).toBeTruthy();
  const jsonBody = await jsonRes.json();
  expect(jsonBody?.data?.id).toBe(backtestId1);

  await page.getByRole("button", { name: "Pullback in Trend" }).click();
  const runBacktestRes2 = page.waitForResponse(
    (res) => res.url().includes("/api/backtests/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Backtest" }).click();
  const runBacktestBody2 = await (await runBacktestRes2).json();
  await waitForJob(request, apiBase, runBacktestBody2.data.job_id);

  await expect(page.getByRole("heading", { name: "Backtest leaderboard" })).toBeVisible();
  const compareBoxes = page.locator(
    'section:has-text("Backtest leaderboard") tbody input[type="checkbox"]',
  );
  await expect(compareBoxes.first()).toBeVisible();
  await compareBoxes.nth(0).check();
  await compareBoxes.nth(1).check();
  await expect(page.getByRole("heading", { name: "Compare runs" })).toBeVisible();
  const compareSection = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Compare runs" }),
  });
  await expect(compareSection.getByLabel("Equity chart")).toBeVisible({ timeout: 20_000 });

  await page.goto("/walk-forward");
  await expect(page.getByRole("heading", { name: "Walk-Forward & Robustness" })).toBeVisible({
    timeout: 20_000,
  });

  await page.getByLabel("Optuna trials").fill(process.env.PW_WF_TRIALS ?? "5");
  const runWalkRes = page.waitForResponse(
    (res) => res.url().includes("/api/walkforward/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Walk-Forward" }).click();
  const runWalkBody = await (await runWalkRes).json();
  const walkJob = await waitForJob(request, apiBase, runWalkBody.data.job_id);
  const walkResult = (walkJob.result_json ?? {}) as Record<string, unknown>;
  const walkRunId = Number(walkResult.run_id);
  expect(Number.isFinite(walkRunId) && walkRunId > 0).toBeTruthy();
  const walkDetail = await request.get(`${apiBase}/api/walkforward/${walkRunId}`);
  expect(walkDetail.ok()).toBeTruthy();
  await expect(page.getByRole("heading", { name: "Fold summary" })).toBeVisible();

  const promoteButton = page.getByRole("button", { name: "Promote to Paper" });
  if (await promoteButton.isEnabled()) {
    await promoteButton.click();
    await expect(page.getByText("Strategy promoted to paper mode")).toBeVisible({
      timeout: 20_000,
    });
  }

  await page.goto("/auto-research");
  await expect(page.getByRole("heading", { name: "Auto Research" })).toBeVisible({
    timeout: 20_000,
  });
  const bundlesRes = await request.get(`${apiBase}/api/universes`);
  expect(bundlesRes.ok()).toBeTruthy();
  const bundlesBody = await bundlesRes.json();
  const bundles = (bundlesBody?.data ?? []) as Array<{ id?: number; symbols?: string[] }>;
  const targetBundle = bundles.find((bundle) => (bundle.symbols ?? []).includes("NIFTY500")) ?? bundles[0];
  const bundleId = Number(targetBundle?.id ?? 0);
  expect(bundleId > 0).toBeTruthy();

  const runResearchApi = await request.post(`${apiBase}/api/research/run`, {
    data: {
      bundle_id: bundleId,
      timeframes: ["1d"],
      strategy_templates: ["trend_breakout"],
      symbol_scope: "liquid",
      config: {
        trials_per_strategy: Number(process.env.PW_RESEARCH_TRIALS ?? "1"),
        max_symbols: 1,
        max_evaluations: 1,
        min_trades: 1,
        stress_pass_rate_threshold: 0,
        sampler: "random",
        pruner: "none",
        seed: 19,
      },
    },
  });
  expect(runResearchApi.ok()).toBeTruthy();
  const runResearchBody = await runResearchApi.json();
  const researchJob = await waitForJob(request, apiBase, runResearchBody.data.job_id, 300_000);
  const researchResult = (researchJob.result_json ?? {}) as Record<string, unknown>;
  const researchRunId = Number(researchResult.run_id);
  expect(Number.isFinite(researchRunId) && researchRunId > 0).toBeTruthy();
  await page.reload();

  await page.getByRole("button", { name: "Candidates" }).click();
  const candidateSection = page.locator("article").filter({
    has: page.getByRole("heading", { name: "Results" }),
  });
  const candidateRows = candidateSection.locator("table tbody tr");
  await expect(candidateRows.first()).toBeVisible({ timeout: 120_000 });

  const createPolicyRes = await request.post(`${apiBase}/api/policies`, {
    data: { research_run_id: researchRunId, name: "E2E Auto Policy" },
  });
  expect(createPolicyRes.ok()).toBeTruthy();
  const createPolicyBody = await createPolicyRes.json();
  const policyId = Number(createPolicyBody?.data?.id);
  expect(Number.isFinite(policyId) && policyId > 0).toBeTruthy();

  const promotePolicyRes = await request.post(`${apiBase}/api/policies/${policyId}/promote-to-paper`);
  expect(promotePolicyRes.ok()).toBeTruthy();
  await page.goto("/paper-trading");
  await expect(page).toHaveURL(/\/paper-trading/);
  await expect(page.getByRole("button", { name: "Run Step" })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText("Universe bundle")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("heading", { name: "Promoted strategies" })).toBeVisible({
    timeout: 20_000,
  });

  await page.getByRole("button", { name: /Preview Signals/ }).click();
  await expect(page.getByText("Preview ready")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("dialog", { name: "Signal preview" })).toBeVisible();
  await page
    .getByRole("dialog", { name: "Signal preview" })
    .getByRole("button", { name: "Close" })
    .click();

  const runStepRes = page.waitForResponse(
    (res) => res.url().includes("/api/paper/run-step") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Step" }).click();
  const runStepBody = await (await runStepRes).json();
  await waitForJob(request, apiBase, runStepBody.data.job_id, 180_000);

  const paperStateRes = await request.get(`${apiBase}/api/paper/state`);
  const paperState = await paperStateRes.json();
  const openPositions = (paperState?.data?.positions ?? []) as Array<unknown>;
  if (openPositions.length > 0) {
    await expect(page.getByRole("heading", { name: "Open positions" })).toBeVisible();
  } else {
    await page.getByRole("button", { name: "Why" }).click();
    await expect(page.getByRole("dialog", { name: "Why this run step" })).toBeVisible();
    await expect(page.getByText("Skipped reasons")).toBeVisible();
  }

  const settingsRes = await request.put(`${apiBase}/api/settings`, {
    data: { allowed_sides: ["BUY"], paper_mode: "strategy", active_policy_id: null },
  });
  expect(settingsRes.ok()).toBeTruthy();
  const sellStepRes = await request.post(`${apiBase}/api/paper/run-step`, {
    data: {
      regime: "TREND_UP",
      signals: [
        {
          symbol: "NIFTY500",
          side: "SELL",
          template: "trend_breakout",
          instrument_kind: "EQUITY_CASH",
          price: 100,
          stop_distance: 5,
          signal_strength: 0.9,
          adv: 10000000000,
          vol_scale: 0,
        },
      ],
      mark_prices: {},
    },
  });
  expect(sellStepRes.ok()).toBeTruthy();
  const sellStepBody = await sellStepRes.json();
  const sellStepJob = await waitForJob(request, apiBase, sellStepBody.data.job_id);
  const sellResult = (sellStepJob.result_json ?? {}) as Record<string, unknown>;
  const sellSkipped = (sellResult.skipped_signals ?? []) as Array<{ reason?: string }>;
  expect(sellSkipped.some((item) => item.reason === "shorts_disabled")).toBeTruthy();
  await page.reload();
  await expect(page.getByText("Could not load paper trading state")).toHaveCount(0);
});
