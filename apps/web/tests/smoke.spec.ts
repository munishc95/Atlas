import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";

import { expect, test, type APIRequestContext } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sampleCsv = path.resolve(__dirname, "../../../data/sample/NIFTY500_1d.csv");
const sampleFutCsv = path.resolve(__dirname, "../../../data/sample/NIFTY500_FUT_1d.csv");
const sampleBadCsv = path.resolve(__dirname, "../../../data/sample/NIFTY500_BAD_1d.csv");

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
  symbol = "NIFTY500",
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
      if (rows.some((row) => row.symbol === symbol && row.timeframe === "1d")) {
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
  const futImportRes = await request.post(`${apiBase}/api/data/import`, {
    multipart: {
      symbol: "NIFTY500_FUT",
      timeframe: "1d",
      provider: "csv",
      instrument_kind: "STOCK_FUT",
      underlying: "NIFTY500",
      lot_size: "50",
      file: {
        name: "NIFTY500_FUT_1d.csv",
        mimeType: "text/csv",
        buffer: readFileSync(sampleFutCsv),
      },
    },
  });
  expect(futImportRes.ok()).toBeTruthy();
  const futImportBody = await futImportRes.json();
  await waitForImportReady(request, apiBase, futImportBody.data.job_id, "NIFTY500_FUT");

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
  const compareCount = await compareBoxes.count();
  if (compareCount >= 2) {
    await expect(compareBoxes.first()).toBeVisible();
    await compareBoxes.nth(0).check();
    await compareBoxes.nth(1).check();
    await expect(page.getByRole("heading", { name: "Compare runs" })).toBeVisible();
    const compareSection = page.locator("section").filter({
      has: page.getByRole("heading", { name: "Compare runs" }),
    });
    await expect(compareSection.getByLabel("Equity chart")).toBeVisible({ timeout: 20_000 });
  }

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
  const targetBundle =
    bundles.find((bundle) => {
      const symbols = bundle.symbols ?? [];
      return symbols.includes("NIFTY500") && symbols.includes("NIFTY500_FUT");
    }) ??
    bundles.find((bundle) => (bundle.symbols ?? []).includes("NIFTY500")) ??
    bundles[0];
  const bundleId = Number(targetBundle?.id ?? 0);
  expect(bundleId > 0).toBeTruthy();
  const qualityRunRes = await request.post(`${apiBase}/api/data/quality/run`, {
    data: { bundle_id: bundleId, timeframe: "1d" },
  });
  expect(qualityRunRes.ok()).toBeTruthy();
  const qualityRunBody = await qualityRunRes.json();
  await waitForJob(request, apiBase, qualityRunBody.data.job_id, 180_000);

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
  await expect(page.getByText(/Shorts: Allowed sides/i)).toBeVisible({ timeout: 20_000 });

  const enableSellRes = await request.put(`${apiBase}/api/settings`, {
    data: { allowed_sides: ["BUY", "SELL"], paper_mode: "strategy", active_policy_id: null },
  });
  expect(enableSellRes.ok()).toBeTruthy();
  const futSellStepRes = await request.post(`${apiBase}/api/paper/run-step`, {
    data: {
      regime: "TREND_UP",
      bundle_id: bundleId,
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
  expect(futSellStepRes.ok()).toBeTruthy();
  const futSellStepBody = await futSellStepRes.json();
  const futSellJob = await waitForJob(request, apiBase, futSellStepBody.data.job_id);
  const futSellResult = (futSellJob.result_json ?? {}) as Record<string, unknown>;
  if (Number(futSellResult.selected_signals_count ?? 0) > 0) {
    const futPositions = (futSellResult.positions ?? []) as Array<{ instrument_kind?: string }>;
    expect(futPositions.some((row) => row.instrument_kind === "STOCK_FUT")).toBeTruthy();
  } else {
    const skipped = (futSellResult.skipped_signals ?? []) as Array<{ reason?: string }>;
    const killSwitch = String(futSellResult.status ?? "") === "kill_switch_active";
    if (!killSwitch) {
      expect(skipped.length).toBeGreaterThan(0);
    }
  }

  const restorePolicyRes = await request.put(`${apiBase}/api/settings`, {
    data: { allowed_sides: ["BUY", "SELL"], paper_mode: "policy", active_policy_id: policyId },
  });
  expect(restorePolicyRes.ok()).toBeTruthy();

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

  const reportGenerateRes = await request.post(`${apiBase}/api/reports/daily/generate`, {
    data: {
      bundle_id: bundleId,
      policy_id: policyId,
    },
  });
  expect(reportGenerateRes.ok()).toBeTruthy();
  const reportGenerateBody = await reportGenerateRes.json();
  const reportJob = await waitForJob(request, apiBase, reportGenerateBody.data.job_id, 120_000);
  const reportResult = (reportJob.result_json ?? {}) as Record<string, unknown>;
  const reportId = Number(reportResult.id);
  expect(Number.isFinite(reportId) && reportId > 0).toBeTruthy();
  const dailyPdfRes = await request.get(`${apiBase}/api/reports/daily/${reportId}/export.pdf`);
  expect(dailyPdfRes.ok()).toBeTruthy();
  expect(dailyPdfRes.headers()["content-type"] ?? "").toContain("application/pdf");
  expect((await dailyPdfRes.body()).byteLength).toBeGreaterThan(1000);

  const month = new Date().toISOString().slice(0, 7);
  const monthlyGenerateRes = await request.post(`${apiBase}/api/reports/monthly/generate`, {
    data: {
      month,
      bundle_id: bundleId,
      policy_id: policyId,
    },
  });
  expect(monthlyGenerateRes.ok()).toBeTruthy();
  const monthlyGenerateBody = await monthlyGenerateRes.json();
  const monthlyJob = await waitForJob(request, apiBase, monthlyGenerateBody.data.job_id, 120_000);
  const monthlyResult = (monthlyJob.result_json ?? {}) as Record<string, unknown>;
  const monthlyId = Number(monthlyResult.id);
  expect(Number.isFinite(monthlyId) && monthlyId > 0).toBeTruthy();
  const monthlyPdfRes = await request.get(`${apiBase}/api/reports/monthly/${monthlyId}/export.pdf`);
  expect(monthlyPdfRes.ok()).toBeTruthy();
  expect(monthlyPdfRes.headers()["content-type"] ?? "").toContain("application/pdf");
  expect((await monthlyPdfRes.body()).byteLength).toBeGreaterThan(1000);

  const evaluationRunRes = await request.post(`${apiBase}/api/evaluations/run`, {
    data: {
      bundle_id: bundleId,
      champion_policy_id: policyId,
      window_days: 20,
      seed: 7,
    },
  });
  expect(evaluationRunRes.ok()).toBeTruthy();
  const evaluationRunBody = await evaluationRunRes.json();
  const evaluationJob = await waitForJob(request, apiBase, evaluationRunBody.data.job_id, 240_000);
  const evaluationResult = (evaluationJob.result_json ?? {}) as Record<string, unknown>;
  const evaluationId = Number(evaluationResult.evaluation_id);
  expect(Number.isFinite(evaluationId) && evaluationId > 0).toBeTruthy();
  const evaluationDetailRes = await request.get(`${apiBase}/api/evaluations/${evaluationId}`);
  expect(evaluationDetailRes.ok()).toBeTruthy();

  const policyHealthRes = await request.get(
    `${apiBase}/api/policies/${policyId}/health?window_days=20&refresh=true`,
  );
  expect(policyHealthRes.ok()).toBeTruthy();
  const policyHealthBody = await policyHealthRes.json();
  expect(policyHealthBody?.data?.policy_id).toBe(policyId);
  expect(policyHealthBody?.data?.status).toBeTruthy();

  await page.goto("/ops");
  await expect(page.getByRole("heading", { name: "Operate Mode" })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByRole("heading", { name: "Recent operate events" })).toBeVisible({
    timeout: 20_000,
  });
  const runTodayButton = page.getByRole("button", {
    name: "Run Today (Updates -> Quality -> Step -> Report)",
  });
  await expect(runTodayButton).toBeVisible({
    timeout: 20_000,
  });
  const operateRunRes = page.waitForResponse(
    (res) => res.url().includes("/api/operate/run") && res.request().method() === "POST",
  );
  await runTodayButton.click();
  const operateRunBody = await (await operateRunRes).json();
  const operateRunJob = await waitForJob(request, apiBase, operateRunBody.data.job_id, 240_000);
  const operateResult = (operateRunJob.result_json ?? {}) as Record<string, unknown>;
  const operateSummary = (operateResult.summary ?? {}) as Record<string, unknown>;
  const operateReport = (operateSummary.daily_report ?? {}) as Record<string, unknown>;
  const operateReportId = Number(operateReport.id);
  expect(Number.isFinite(operateReportId) && operateReportId > 0).toBeTruthy();
  const operatePdfRes = await request.get(
    `${apiBase}/api/reports/daily/${operateReportId}/export.pdf`,
  );
  expect(operatePdfRes.ok()).toBeTruthy();
  expect(operatePdfRes.headers()["content-type"] ?? "").toContain("application/pdf");

  await page.goto("/reports");
  await expect(page.getByRole("heading", { name: "Reports", exact: true })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByRole("heading", { name: "Daily reports", exact: true })).toBeVisible();
  const viewButton = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Daily reports" }) })
    .getByRole("button", { name: "View" })
    .first();
  await expect(viewButton).toBeVisible();
  await viewButton.click();
  await expect(page.getByRole("dialog", { name: /Daily Report/i })).toBeVisible();
  await page.getByRole("dialog", { name: /Daily Report/i }).getByRole("button", { name: "Close" }).click();

  await page.goto("/evaluations");
  await expect(
    page.getByRole("heading", { name: "Champion-Challenger Evaluations" }),
  ).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByRole("heading", { name: "Recent evaluations" })).toBeVisible();

  const paperStateRes = await request.get(`${apiBase}/api/paper/state`);
  const paperState = await paperStateRes.json();
  const openPositions = (paperState?.data?.positions ?? []) as Array<unknown>;
  await page.goto("/paper-trading");
  if (openPositions.length > 0) {
    await expect(page.getByRole("heading", { name: "Open positions" })).toBeVisible();
  } else {
    await expect(page.getByRole("button", { name: "Why" })).toBeVisible();
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
  if (String(sellResult.status ?? "") !== "kill_switch_active") {
    expect(
      sellSkipped.some((item) =>
        [
          "shorts_disabled",
          "already_open",
          "no_short_instrument_available",
          "max_positions_reached",
          "sector_concentration",
        ].includes(String(item.reason ?? "")),
      ),
    ).toBeTruthy();
  }

  const badImportRes = await request.post(`${apiBase}/api/data/import`, {
    multipart: {
      symbol: "NIFTY500_BAD",
      timeframe: "1d",
      provider: "csv",
      file: {
        name: "NIFTY500_BAD_1d.csv",
        mimeType: "text/csv",
        buffer: readFileSync(sampleBadCsv),
      },
    },
  });
  expect(badImportRes.ok()).toBeTruthy();
  const badImportBody = await badImportRes.json();
  await waitForImportReady(request, apiBase, badImportBody.data.job_id, "NIFTY500_BAD");

  const bundlesAfterBadRes = await request.get(`${apiBase}/api/universes`);
  expect(bundlesAfterBadRes.ok()).toBeTruthy();
  const bundlesAfterBadBody = await bundlesAfterBadRes.json();
  const bundlesAfterBad = (bundlesAfterBadBody?.data ?? []) as Array<{ id?: number; symbols?: string[] }>;
  const badBundle =
    bundlesAfterBad.find((bundle) => (bundle.symbols ?? []).includes("NIFTY500_BAD")) ?? null;
  expect(badBundle?.id).toBeTruthy();
  const badBundleId = Number(badBundle?.id ?? 0);

  const stateBeforeShadowRes = await request.get(`${apiBase}/api/paper/state`);
  expect(stateBeforeShadowRes.ok()).toBeTruthy();
  const stateBeforeShadowBody = await stateBeforeShadowRes.json();
  const stateBeforeShadow = stateBeforeShadowBody?.data?.state ?? {};

  const enableShadowSettingsRes = await request.put(`${apiBase}/api/settings`, {
    data: {
      paper_mode: "strategy",
      active_policy_id: null,
      allowed_sides: ["BUY", "SELL"],
      operate_safe_mode_on_fail: true,
      operate_safe_mode_action: "shadow_only",
      operate_mode: "live",
      data_quality_stale_severity: "FAIL",
      paper_use_simulator_engine: true,
    },
  });
  expect(enableShadowSettingsRes.ok()).toBeTruthy();

  const shadowStepRes = await request.post(`${apiBase}/api/paper/run-step`, {
    data: {
      regime: "TREND_UP",
      bundle_id: badBundleId,
      timeframes: ["1d"],
      signals: [
        {
          symbol: "NIFTY500_BAD",
          side: "BUY",
          template: "trend_breakout",
          instrument_kind: "EQUITY_CASH",
          price: 100,
          stop_distance: 5,
          signal_strength: 0.7,
          adv: 10000000000,
          vol_scale: 0.01,
        },
      ],
      mark_prices: {},
    },
  });
  expect(shadowStepRes.ok()).toBeTruthy();
  const shadowStepBody = await shadowStepRes.json();
  const shadowStepJob = await waitForJob(request, apiBase, shadowStepBody.data.job_id);
  const shadowResult = (shadowStepJob.result_json ?? {}) as Record<string, unknown>;
  expect(String(shadowResult.execution_mode ?? "")).toBe("SHADOW");
  expect(Boolean(shadowResult.live_state_mutated)).toBeFalsy();

  const stateAfterShadowRes = await request.get(`${apiBase}/api/paper/state`);
  expect(stateAfterShadowRes.ok()).toBeTruthy();
  const stateAfterShadowBody = await stateAfterShadowRes.json();
  const stateAfterShadow = stateAfterShadowBody?.data?.state ?? {};
  expect(Number(stateAfterShadow.cash ?? 0)).toBe(Number(stateBeforeShadow.cash ?? 0));
  expect(Number(stateAfterShadow.equity ?? 0)).toBe(Number(stateBeforeShadow.equity ?? 0));

  const shadowReportGenerateRes = await request.post(`${apiBase}/api/reports/daily/generate`, {
    data: {
      date: new Date().toISOString().slice(0, 10),
      bundle_id: badBundleId,
      policy_id: null,
    },
  });
  expect(shadowReportGenerateRes.ok()).toBeTruthy();
  const shadowReportGenerateBody = await shadowReportGenerateRes.json();
  const shadowReportJob = await waitForJob(request, apiBase, shadowReportGenerateBody.data.job_id, 120_000);
  const shadowReportResult = (shadowReportJob.result_json ?? {}) as Record<string, unknown>;
  const shadowReportId = Number(shadowReportResult.id);
  expect(Number.isFinite(shadowReportId) && shadowReportId > 0).toBeTruthy();
  const shadowReportDetailRes = await request.get(`${apiBase}/api/reports/daily/${shadowReportId}`);
  expect(shadowReportDetailRes.ok()).toBeTruthy();
  const shadowReportDetailBody = await shadowReportDetailRes.json();
  expect(String(shadowReportDetailBody?.data?.content_json?.summary?.mode ?? "")).toBe("SHADOW");
  expect(
    String(shadowReportDetailBody?.data?.content_json?.summary?.shadow_note ?? ""),
  ).toContain("Shadow-only");

  await page.goto("/reports");
  const shadowViewButton = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Daily reports" }) })
    .getByRole("button", { name: "View" })
    .first();
  await shadowViewButton.click();
  const shadowDialog = page.getByRole("dialog", { name: /Daily Report/i });
  await expect(shadowDialog).toContainText("Mode:");
  await shadowDialog.getByRole("button", { name: "Close" }).click();

  await request.put(`${apiBase}/api/settings`, {
    data: {
      operate_mode: "offline",
      operate_safe_mode_action: "exits_only",
      paper_mode: "policy",
      active_policy_id: policyId,
    },
  });
  await page.reload();
  await expect(page.getByText("Could not load paper trading state")).toHaveCount(0);
});
