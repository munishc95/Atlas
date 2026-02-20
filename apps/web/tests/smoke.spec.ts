import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { expect, test, type APIRequestContext } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sampleCsv = path.resolve(__dirname, "../../../data/sample/NIFTY500_1d.csv");

async function waitForJob(
  request: APIRequestContext,
  apiBase: string,
  jobId: string,
  timeoutMs = 180_000,
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
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  throw new Error(`Timed out waiting for job ${jobId}`);
}

async function ensureSampleImport(
  request: APIRequestContext,
  apiBase: string,
  symbol: string,
): Promise<void> {
  const statusRes = await request.get(`${apiBase}/api/data/status`);
  if (statusRes.ok()) {
    const statusBody = await statusRes.json();
    const rows = (statusBody?.data ?? []) as Array<{ symbol?: string; timeframe?: string }>;
    if (rows.some((row) => row.symbol === symbol && row.timeframe === "1d")) {
      return;
    }
  }

  const importRes = await request.post(`${apiBase}/api/data/import`, {
    multipart: {
      symbol,
      timeframe: "1d",
      provider: "csv",
      file: {
        name: `${symbol}_1d.csv`,
        mimeType: "text/csv",
        buffer: readFileSync(sampleCsv),
      },
    },
  });
  expect(importRes.ok()).toBeTruthy();
  const importBody = await importRes.json();
  await waitForJob(request, apiBase, String(importBody?.data?.job_id));
}

test("@smoke fast operate run + report pdf + ops health", async ({ page, request }) => {
  test.setTimeout(240_000);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

  await ensureSampleImport(request, apiBase, "NIFTY500");

  const healthRes = await request.get(`${apiBase}/api/operate/health`);
  expect(healthRes.ok()).toBeTruthy();
  const healthBody = await healthRes.json();
  expect(Boolean(healthBody?.data?.fast_mode_enabled)).toBeTruthy();

  const bundlesRes = await request.get(`${apiBase}/api/universes`);
  expect(bundlesRes.ok()).toBeTruthy();
  const bundlesBody = await bundlesRes.json();
  const bundles = (bundlesBody?.data ?? []) as Array<{ id?: number; symbols?: string[] }>;
  const targetBundle =
    bundles.find((bundle) => (bundle.symbols ?? []).includes("NIFTY500")) ?? bundles[0];
  const bundleId = Number(targetBundle?.id ?? 0);
  expect(bundleId > 0).toBeTruthy();

  const settingsRes = await request.put(`${apiBase}/api/settings`, {
    data: {
      operate_auto_run_include_data_updates: false,
      paper_mode: "strategy",
      active_policy_id: null,
      data_updates_provider_enabled: true,
      data_updates_provider_kind: "MOCK",
      data_updates_provider_timeframe_enabled: "1d",
      data_updates_provider_max_symbols_per_run: 5,
      data_updates_provider_max_calls_per_run: 20,
      data_updates_provider_timeframes: ["1d"],
      upstox_auto_renew_enabled: true,
      upstox_auto_renew_time_ist: "06:30",
      upstox_auto_renew_if_expires_within_hours: 12,
      upstox_auto_renew_only_when_provider_enabled: true,
      data_updates_provider_repair_last_n_trading_days: 2,
      data_updates_provider_backfill_max_days: 5,
      no_trade_enabled: true,
      no_trade_regimes: ["TREND_UP", "RANGE", "HIGH_VOL", "RISK_OFF"],
      no_trade_max_realized_vol_annual: 10,
      no_trade_min_breadth_pct: 0,
      no_trade_min_trend_strength: 0,
      no_trade_cooldown_trading_days: 0,
    },
  });
  expect(settingsRes.ok()).toBeTruthy();

  const tokenRequestRes = await request.post(`${apiBase}/api/providers/upstox/token/request`, {
    data: {
      source: "e2e_smoke",
    },
  });
  expect(tokenRequestRes.ok()).toBeTruthy();
  const tokenRequestBody = await tokenRequestRes.json();
  expect(String(tokenRequestBody?.data?.run?.status ?? "")).toMatch(/REQUESTED|APPROVED/);
  const statusRes = await request.get(`${apiBase}/api/providers/upstox/token/status`);
  expect(statusRes.ok()).toBeTruthy();
  expect(Boolean((await statusRes.json())?.data?.connected)).toBeTruthy();

  const mappingDir = path.resolve(__dirname, "../../../data/inbox/_metadata");
  mkdirSync(mappingDir, { recursive: true });
  const mappingPath = path.resolve(mappingDir, "upstox_instruments.csv");
  writeFileSync(mappingPath, "symbol,instrument_key\nNIFTY500,NSE_EQ|NIFTY500\n", "utf-8");

  const mappingImportRes = await request.post(`${apiBase}/api/providers/upstox/mapping/import`, {
    data: {
      path: mappingPath,
      mode: "UPSERT",
    },
    params: { bundle_id: bundleId },
  });
  expect(mappingImportRes.ok()).toBeTruthy();

  const providerUpdateRes = await request.post(`${apiBase}/api/data/provider-updates/run`, {
    data: {
      bundle_id: bundleId,
      timeframe: "1d",
    },
  });
  expect(providerUpdateRes.ok()).toBeTruthy();
  const providerUpdateBody = await providerUpdateRes.json();
  const providerUpdateJob = await waitForJob(
    request,
    apiBase,
    String(providerUpdateBody?.data?.job_id),
    180_000,
  );
  expect(providerUpdateJob.status).toBe("SUCCEEDED");

  const runResearchRes = await request.post(`${apiBase}/api/research/run`, {
    data: {
      bundle_id: bundleId,
      timeframes: ["1d"],
      strategy_templates: ["trend_breakout"],
      symbol_scope: "liquid",
      config: {
        trials_per_strategy: 1,
        max_symbols: 1,
        max_evaluations: 1,
        min_trades: 1,
        stress_pass_rate_threshold: 0,
        sampler: "random",
        pruner: "none",
        seed: 17,
      },
    },
  });
  expect(runResearchRes.ok()).toBeTruthy();
  const runResearchBody = await runResearchRes.json();
  const researchJob = await waitForJob(request, apiBase, String(runResearchBody?.data?.job_id), 240_000);
  const researchResult = (researchJob.result_json ?? {}) as Record<string, unknown>;
  const researchRunId = Number(researchResult.run_id ?? 0);
  expect(researchRunId > 0).toBeTruthy();

  const createPolicyARes = await request.post(`${apiBase}/api/policies`, {
    data: { research_run_id: researchRunId, name: "Smoke Ensemble A" },
  });
  expect(createPolicyARes.ok()).toBeTruthy();
  const createPolicyABody = await createPolicyARes.json();
  const policyAId = Number(createPolicyABody?.data?.id ?? 0);
  expect(policyAId > 0).toBeTruthy();

  const createPolicyBRes = await request.post(`${apiBase}/api/policies`, {
    data: { research_run_id: researchRunId, name: "Smoke Ensemble B" },
  });
  expect(createPolicyBRes.ok()).toBeTruthy();
  const createPolicyBBody = await createPolicyBRes.json();
  const policyBId = Number(createPolicyBBody?.data?.id ?? 0);
  expect(policyBId > 0).toBeTruthy();

  const createEnsembleRes = await request.post(`${apiBase}/api/ensembles`, {
    data: {
      name: "Smoke Ensemble",
      bundle_id: bundleId,
      is_active: false,
    },
  });
  expect(createEnsembleRes.ok()).toBeTruthy();
  const createEnsembleBody = await createEnsembleRes.json();
  const ensembleId = Number(createEnsembleBody?.data?.id ?? 0);
  expect(ensembleId > 0).toBeTruthy();

  const upsertMembersRes = await request.post(`${apiBase}/api/ensembles/${ensembleId}/members`, {
    data: {
      members: [
        { policy_id: policyAId, weight: 0.6, enabled: true },
        { policy_id: policyBId, weight: 0.4, enabled: true },
      ],
    },
  });
  expect(upsertMembersRes.ok()).toBeTruthy();

  const setActiveEnsembleRes = await request.post(
    `${apiBase}/api/ensembles/${ensembleId}/set-active`,
    { data: {} },
  );
  expect(setActiveEnsembleRes.ok()).toBeTruthy();
  const enforcePolicyModeRes = await request.put(`${apiBase}/api/settings`, {
    data: {
      paper_mode: "policy",
      active_policy_id: null,
      active_ensemble_id: ensembleId,
      operate_mode: "offline",
      operate_safe_mode_on_fail: false,
      data_quality_stale_severity: "WARN",
      data_quality_stale_severity_override: true,
      coverage_missing_latest_warn_pct: 100,
      coverage_missing_latest_fail_pct: 100,
    },
  });
  expect(enforcePolicyModeRes.ok()).toBeTruthy();

  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Providers - Upstox")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Connected")).toBeVisible({ timeout: 20_000 });

  await page.goto("/ops");
  await expect(page.getByRole("heading", { name: "Operate Mode" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Fast mode:/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Mapping missing:/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Provider updates are enabled but Upstox token is/i)).toHaveCount(0);

  const runOperateRes = await request.post(`${apiBase}/api/operate/run`, {
    data: {
      bundle_id: bundleId,
      timeframe: "1d",
      policy_id: null,
      date: new Date().toISOString().slice(0, 10),
    },
  });
  expect(runOperateRes.ok()).toBeTruthy();
  const runOperateBody = await runOperateRes.json();
  const operateJob = await waitForJob(request, apiBase, String(runOperateBody?.data?.job_id), 240_000);
  const operateResult = (operateJob.result_json ?? {}) as Record<string, unknown>;
  const operateSummary = (operateResult.summary ?? {}) as Record<string, unknown>;
  const reportPayload = (operateSummary.daily_report ?? {}) as Record<string, unknown>;
  const reportId = Number(reportPayload.id ?? 0);
  expect(reportId > 0).toBeTruthy();

  const reportRes = await request.get(`${apiBase}/api/reports/daily/${reportId}`);
  expect(reportRes.ok()).toBeTruthy();
  const reportBody = await reportRes.json();
  const reportData = (reportBody?.data ?? {}) as Record<string, unknown>;
  const reportSummary = (reportData.content_json as Record<string, unknown> | undefined)?.summary as
    | Record<string, unknown>
    | undefined;
  const reportExplainability = (reportData.content_json as Record<string, unknown> | undefined)
    ?.explainability as Record<string, unknown> | undefined;
  expect(
    Object.prototype.hasOwnProperty.call(reportSummary ?? {}, "ensemble_active"),
  ).toBeTruthy();
  expect(
    Object.prototype.hasOwnProperty.call(
      reportExplainability ?? {},
      "ensemble_selected_counts_by_policy",
    ),
  ).toBeTruthy();
  expect(reportSummary?.no_trade_triggered).toBeTruthy();
  expect(Array.isArray(reportSummary?.no_trade_reasons)).toBeTruthy();
  expect((reportSummary?.no_trade_reasons as unknown[]).length).toBeGreaterThan(0);

  await page.reload();
  await expect(page.getByText(/Active ensemble:/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/No-trade gate:/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/No-trade reason:/i)).toBeVisible({ timeout: 20_000 });

  const pdfRes = await request.get(`${apiBase}/api/reports/daily/${reportId}/export.pdf`);
  expect(pdfRes.ok()).toBeTruthy();
  expect(pdfRes.headers()["content-type"] ?? "").toContain("application/pdf");
  expect((await pdfRes.body()).byteLength).toBeGreaterThan(1000);

  await expect(page.getByRole("heading", { name: "Latest operate run" })).toBeVisible({
    timeout: 20_000,
  });
});
