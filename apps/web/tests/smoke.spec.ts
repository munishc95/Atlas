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
    },
  });
  expect(settingsRes.ok()).toBeTruthy();

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

  await page.goto("/ops");
  await expect(page.getByRole("heading", { name: "Operate Mode" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Fast mode:/i)).toBeVisible({ timeout: 20_000 });

  const runOperateRes = page.waitForResponse(
    (res) => res.url().includes("/api/operate/run") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run Today (Updates -> Quality -> Step -> Report)" }).click();
  const runOperateBody = await (await runOperateRes).json();
  const operateJob = await waitForJob(request, apiBase, String(runOperateBody?.data?.job_id), 240_000);
  const operateResult = (operateJob.result_json ?? {}) as Record<string, unknown>;
  const operateSummary = (operateResult.summary ?? {}) as Record<string, unknown>;
  const reportPayload = (operateSummary.daily_report ?? {}) as Record<string, unknown>;
  const reportId = Number(reportPayload.id ?? 0);
  expect(reportId > 0).toBeTruthy();

  const pdfRes = await request.get(`${apiBase}/api/reports/daily/${reportId}/export.pdf`);
  expect(pdfRes.ok()).toBeTruthy();
  expect(pdfRes.headers()["content-type"] ?? "").toContain("application/pdf");
  expect((await pdfRes.body()).byteLength).toBeGreaterThan(1000);

  await expect(page.getByRole("heading", { name: "Latest operate run" })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.getByText("No operate run summary yet")).toHaveCount(0);
});
