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

  const enforceStrategyModeRes = await request.put(`${apiBase}/api/settings`, {
    data: {
      paper_mode: "strategy",
      active_policy_id: null,
      active_ensemble_id: null,
      operate_mode: "live",
      operate_safe_mode_on_fail: false,
      data_quality_stale_severity: "WARN",
      data_quality_stale_severity_override: true,
      coverage_missing_latest_warn_pct: 100,
      coverage_missing_latest_fail_pct: 100,
      no_trade_enabled: false,
      confidence_gate_enabled: true,
      confidence_gate_avg_threshold: 99,
      confidence_gate_low_symbol_threshold: 99,
      confidence_gate_low_pct_threshold: 0.1,
      confidence_gate_fallback_pct_threshold: 0.0,
      confidence_gate_hard_floor: 40,
      confidence_gate_action_on_trigger: "SHADOW_ONLY",
      confidence_gate_lookback_days: 1,
    },
  });
  expect(enforceStrategyModeRes.ok()).toBeTruthy();

  await page.goto("/ops");
  await expect(page.getByRole("heading", { name: "Operate Mode" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Fast mode:/i)).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Renew Upstox Token Now" }).click();
  await expect
    .poll(async () => {
      const tokenRes = await request.get(`${apiBase}/api/providers/upstox/token/status`);
      const tokenBody = await tokenRes.json();
      const tokenData = (tokenBody?.data ?? {}) as Record<string, unknown>;
      return Boolean(tokenData.connected) && !Boolean(tokenData.is_expired);
    })
    .toBeTruthy();

  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Providers - Upstox")).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Generate Ping URL" }).click();
  await expect(page.getByText(/Ping status:/i)).toBeVisible({ timeout: 20_000 });
  const pingLine = page.getByText(/Ping URL:/i).first();
  await expect(pingLine).toBeVisible({ timeout: 20_000 });
  const pingUrlRaw = (await pingLine.textContent()) ?? "";
  const pingUrl = pingUrlRaw.replace(/^Ping URL:\s*/i, "").trim();
  expect(pingUrl.length > 0).toBeTruthy();
  const pingPage = await page.context().newPage();
  await pingPage.goto(pingUrl, { waitUntil: "domcontentloaded" });
  await pingPage.close();
  const pingPath = new URL(pingUrl).pathname;
  const pingId = pingPath.split("/").at(-1) ?? "";
  expect(pingId.length > 0).toBeTruthy();
  await expect
    .poll(async () => {
      const pingStatusRes = await request.get(
        `${apiBase}/api/providers/upstox/notifier/ping/${pingId}/status`,
      );
      const pingStatusBody = await pingStatusRes.json();
      return String(pingStatusBody?.data?.status ?? "");
    })
    .toBe("RECEIVED");

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
  expect(String(operateSummary.mode ?? "")).toBe("SHADOW");
  const reportPayload = (operateSummary.daily_report ?? {}) as Record<string, unknown>;
  const reportId = Number(reportPayload.id ?? 0);
  expect(reportId > 0).toBeTruthy();

  const reportRes = await request.get(`${apiBase}/api/reports/daily/${reportId}`);
  expect(reportRes.ok()).toBeTruthy();
  const reportBody = await reportRes.json();
  const reportData = (reportBody?.data ?? {}) as Record<string, unknown>;
  expect(typeof reportData.content_json).toBe("object");
  const reportContent = (reportData.content_json ?? {}) as Record<string, unknown>;
  const confidenceGate = (reportContent.confidence_gate ?? {}) as Record<string, unknown>;
  const confidenceRiskScaling = (reportContent.confidence_risk_scaling ?? {}) as Record<
    string,
    unknown
  >;
  expect(String(confidenceGate.decision ?? "")).toBe("SHADOW_ONLY");
  expect(Number(confidenceRiskScaling.scale ?? 1)).toBeLessThan(1);
  expect(String((reportContent.summary as Record<string, unknown> | undefined)?.mode ?? "")).toBe(
    "SHADOW",
  );

  const disconnectRes = await request.post(`${apiBase}/api/providers/upstox/disconnect`, {
    data: {},
  });
  expect(disconnectRes.ok()).toBeTruthy();

  const fallbackSettingsRes = await request.put(`${apiBase}/api/settings`, {
    data: {
      data_updates_provider_enabled: true,
      data_updates_provider_kind: "UPSTOX",
      data_updates_provider_mode: "FALLBACK",
      data_updates_provider_priority_order: ["UPSTOX", "NSE_EOD", "INBOX"],
      data_updates_provider_nse_eod_enabled: true,
      data_updates_provider_timeframes: ["1d"],
      data_updates_provider_max_symbols_per_run: 5,
      data_updates_provider_max_calls_per_run: 20,
    },
  });
  expect(fallbackSettingsRes.ok()).toBeTruthy();

  const fallbackProviderRunRes = await request.post(`${apiBase}/api/data/provider-updates/run`, {
    data: {
      bundle_id: bundleId,
      timeframe: "1d",
      provider_kind: "UPSTOX",
      provider_mode: "FALLBACK",
      provider_priority_order: ["UPSTOX", "NSE_EOD", "INBOX"],
    },
  });
  expect(fallbackProviderRunRes.ok()).toBeTruthy();
  const fallbackProviderRunBody = await fallbackProviderRunRes.json();
  await waitForJob(request, apiBase, String(fallbackProviderRunBody?.data?.job_id), 180_000);
  const fallbackLatestRes = await request.get(
    `${apiBase}/api/data/provider-updates/latest?bundle_id=${bundleId}&timeframe=1d`,
  );
  expect(fallbackLatestRes.ok()).toBeTruthy();
  const fallbackLatestBody = await fallbackLatestRes.json();
  const fallbackLatest = (fallbackLatestBody?.data ?? {}) as Record<string, unknown>;
  const byProvider = (fallbackLatest.by_provider_count_json ?? {}) as Record<string, number>;
  expect(Number(byProvider.NSE_EOD ?? 0)).toBeGreaterThan(0);

  await page.reload();
  await page.goto("/universe-data");
  await expect(page.getByRole("heading", { name: "Universe & Data" })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByText(/Provider status/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/NSE_EOD/i).first()).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "View provenance" }).first().click();
  await expect(page.getByText(/Data Provenance/i)).toBeVisible({ timeout: 20_000 });

  const pdfRes = await request.get(`${apiBase}/api/reports/daily/${reportId}/export.pdf`);
  expect(pdfRes.ok()).toBeTruthy();
  expect(pdfRes.headers()["content-type"] ?? "").toContain("application/pdf");
  expect((await pdfRes.body()).byteLength).toBeGreaterThan(1000);

  await page.goto("/ops");
  await expect(page.getByRole("heading", { name: "Operate Mode" })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByRole("button", { name: "Context" }).first()).toBeVisible({
    timeout: 20_000,
  });
  await page.getByRole("button", { name: "Context" }).first().click();
  await expect(page.getByText(/Effective Trading Context/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/trading_date/i).first()).toBeVisible({ timeout: 20_000 });
  await page.keyboard.press("Escape");

  await expect(page.getByRole("heading", { name: "Data Confidence" })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByText(/Confidence risk scale:/i)).toBeVisible({ timeout: 20_000 });
  const confidenceCard = page
    .locator("div.rounded-xl.border")
    .filter({ has: page.getByRole("heading", { name: "Data Confidence" }) })
    .first();
  await expect(confidenceCard.getByText("SHADOW", { exact: true })).toBeVisible({
    timeout: 20_000,
  });
  await page.getByRole("button", { name: "View trend" }).click();
  await expect(page.getByText(/Data Confidence Trend/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Risk scale/i).first()).toBeVisible({ timeout: 20_000 });
  await page.locator("button").filter({ hasText: /Avg confidence/i }).first().click();
  await expect(page.getByText(/Confidence Drilldown/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Worst symbols/i)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Provider mix delta:/i)).toBeVisible({ timeout: 20_000 });
});
