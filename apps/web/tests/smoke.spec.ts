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
  await expect(page.getByRole("heading", { name: "Universe & Data" })).toBeVisible();

  await page.getByRole("link", { name: "Strategy Lab" }).click();
  await expect(page.getByRole("heading", { name: "Strategy Lab" })).toBeVisible();

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

  await page.getByRole("link", { name: "Walk-Forward" }).click();
  await expect(page.getByRole("heading", { name: "Walk-Forward & Robustness" })).toBeVisible();

  await page.getByLabel("Optuna trials").fill(process.env.PW_WF_TRIALS ?? "5");
  const runWalkRes = page.waitForResponse(
    (res) => res.url().includes("/api/walkforward/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Walk-Forward" }).click();
  const runWalkBody = await (await runWalkRes).json();
  await waitForJob(request, apiBase, runWalkBody.data.job_id);
  const foldsSection = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Fold summary" }),
  });
  const foldRows = foldsSection.locator("table tbody tr");
  if ((await foldRows.count()) > 0) {
    await expect(foldRows.first()).toBeVisible({ timeout: 120_000 });
  } else {
    await expect(foldsSection.getByText("No folds yet")).toBeVisible({ timeout: 120_000 });
  }

  const promoteButton = page.getByRole("button", { name: "Promote to Paper" });
  if (await promoteButton.isEnabled()) {
    await promoteButton.click();
    await expect(page.getByText("Strategy promoted to paper mode")).toBeVisible({
      timeout: 20_000,
    });
  }

  await page.getByRole("link", { name: "Auto Research" }).click();
  await expect(page.getByRole("heading", { name: "Auto Research" })).toBeVisible();

  await page.getByLabel("Trials per strategy").fill(process.env.PW_RESEARCH_TRIALS ?? "1");
  await page.getByLabel("Max symbols sampled").fill("1");
  await page.getByLabel("Max evaluations (0 = no cap)").fill("1");
  await page.getByLabel("Minimum average trades").fill("1");
  await page.getByLabel("Stress pass threshold").fill("0");

  const runResearchRes = page.waitForResponse(
    (res) => res.url().includes("/api/research/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Auto Research" }).click();
  const runResearchBody = await (await runResearchRes).json();
  const researchJob = await waitForJob(request, apiBase, runResearchBody.data.job_id, 300_000);
  const researchResult = (researchJob.result_json ?? {}) as Record<string, unknown>;
  const researchRunId = Number(researchResult.run_id);
  expect(Number.isFinite(researchRunId) && researchRunId > 0).toBeTruthy();

  await page.getByRole("button", { name: "Candidates" }).click();
  const candidateSection = page.locator("article").filter({
    has: page.getByRole("heading", { name: "Results" }),
  });
  const candidateRows = candidateSection.locator("table tbody tr");
  await expect(candidateRows.first()).toBeVisible({ timeout: 120_000 });

  await page.getByRole("button", { name: "Policy" }).click();
  await page.getByRole("button", { name: "Create Policy" }).click();
  const policyDialog = page.getByRole("dialog", { name: "Create policy" });
  await policyDialog.getByPlaceholder("Atlas Policy - 2026-02-13").fill("E2E Auto Policy");
  await policyDialog.getByRole("button", { name: "Create", exact: true }).click();
  await expect(page.getByText("Policy created")).toBeVisible({ timeout: 20_000 });

  const useInPaperButton = page.getByRole("button", { name: "Use in Paper" });
  await expect(useInPaperButton).toBeEnabled({ timeout: 30_000 });
  await useInPaperButton.click();

  await expect(page.getByRole("heading", { name: "Paper Trading" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Promoted strategies" })).toBeVisible();
});
