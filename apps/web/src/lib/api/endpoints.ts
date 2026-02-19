import { apiFetch, buildApiUrl } from "@/src/lib/api/client";
import type {
  ApiBacktest,
  ApiBacktestComparison,
  ApiBacktestRunRow,
  ApiPaperOrder,
  ApiPaperPosition,
  ApiPaperSignalPreview,
  ApiPaperState,
  ApiPolicyEnsemble,
  ApiPolicyHealthSnapshot,
  ApiPolicy,
  ApiDailyReport,
  ApiDataCoverage,
  ApiDataQualityReport,
  ApiDataUpdateRun,
  ApiProviderUpdateRun,
  ApiOperateEvent,
  ApiOperateHealth,
  ApiOperateRunSummary,
  ApiOperateStatus,
  ApiAutoEvalRun,
  ApiMappingImportRun,
  ApiPolicySwitchEvent,
  ApiMonthlyReport,
  ApiUpstoxMappingStatus,
  ApiUpstoxAuthUrl,
  ApiUpstoxTokenStatus,
  ApiResearchCandidate,
  ApiResearchRun,
  ApiPolicyEvaluation,
  ApiPolicyShadowRun,
  ApiReplayRun,
  ApiTrade,
  ApiUniverseBundle,
  ApiWalkForwardRun,
  Job,
} from "@/src/lib/api/types";

export type JobStart = { job_id: string; status: string };

export const atlasApi = {
  health: () => apiFetch<{ status: string }>("/api/health"),
  universe: () => apiFetch<{ symbols: string[] }>("/api/universe"),
  universes: () => apiFetch<ApiUniverseBundle[]>("/api/universes"),
  strategyTemplates: () => apiFetch<Array<Record<string, unknown>>>("/api/strategies/templates"),
  strategies: () => apiFetch<Array<Record<string, unknown>>>("/api/strategies"),
  dataStatus: () => apiFetch<Array<Record<string, unknown>>>("/api/data/status"),
  runDataQuality: (payload: { bundle_id: number; timeframe?: string }) =>
    apiFetch<JobStart>("/api/data/quality/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runDataUpdates: (payload: { bundle_id: number; timeframe?: string; max_files_per_run?: number }) =>
    apiFetch<JobStart>("/api/data/updates/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runProviderUpdates: (payload: {
    bundle_id: number;
    timeframe?: string;
    provider_kind?: string;
    max_symbols_per_run?: number;
    max_calls_per_run?: number;
    start?: string;
    end?: string;
  }) =>
    apiFetch<JobStart>("/api/data/provider-updates/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  dataUpdatesLatest: (bundleId: number, timeframe = "1d") =>
    apiFetch<ApiDataUpdateRun>(
      `/api/data/updates/latest?bundle_id=${bundleId}&timeframe=${encodeURIComponent(timeframe)}`,
    ),
  dataUpdatesHistory: (params?: { bundle_id?: number; timeframe?: string; days?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (params?.days) search.set("days", String(params.days));
    return apiFetch<ApiDataUpdateRun[]>(
      `/api/data/updates/history${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  providerUpdatesLatest: (bundleId?: number, timeframe?: string) => {
    const search = new URLSearchParams();
    if (typeof bundleId === "number") search.set("bundle_id", String(bundleId));
    if (timeframe) search.set("timeframe", timeframe);
    return apiFetch<ApiProviderUpdateRun>(
      `/api/data/provider-updates/latest${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  providerUpdatesHistory: (params?: { bundle_id?: number; timeframe?: string; days?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (params?.days) search.set("days", String(params.days));
    return apiFetch<ApiProviderUpdateRun[]>(
      `/api/data/provider-updates/history${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  upstoxMappingStatus: (params?: { bundle_id?: number; timeframe?: string; sample_limit?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (typeof params?.sample_limit === "number") {
      search.set("sample_limit", String(params.sample_limit));
    }
    return apiFetch<ApiUpstoxMappingStatus>(
      `/api/providers/upstox/mapping/status${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  upstoxMappingMissing: (params?: { bundle_id?: number; timeframe?: string; limit?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (typeof params?.limit === "number") search.set("limit", String(params.limit));
    return apiFetch<{ symbols: string[]; count: number }>(
      `/api/providers/upstox/mapping/missing${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  importUpstoxMapping: (payload: { path: string; mode?: "UPSERT" | "REPLACE"; bundle_id?: number }) => {
    const search = new URLSearchParams();
    if (typeof payload.bundle_id === "number") search.set("bundle_id", String(payload.bundle_id));
    return apiFetch<{ run: ApiMappingImportRun; status: ApiUpstoxMappingStatus }>(
      `/api/providers/upstox/mapping/import${search.toString() ? `?${search.toString()}` : ""}`,
      {
        method: "POST",
        body: JSON.stringify({
          path: payload.path,
          mode: payload.mode ?? "UPSERT",
        }),
      },
    );
  },
  upstoxAuthUrl: (params?: { redirect_uri?: string; state?: string }) => {
    const search = new URLSearchParams();
    if (params?.redirect_uri) search.set("redirect_uri", params.redirect_uri);
    if (params?.state) search.set("state", params.state);
    return apiFetch<ApiUpstoxAuthUrl>(
      `/api/providers/upstox/auth-url${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  upstoxTokenExchange: (payload: {
    code: string;
    state: string;
    redirect_uri?: string;
    persist_token?: boolean;
  }) =>
    apiFetch<{
      token_masked?: string | null;
      connected: boolean;
      expires_at?: string | null;
      last_verified_at?: string | null;
      token_source?: string | null;
      verification?: Record<string, unknown>;
      persisted_paths?: string[];
      note?: string;
    }>("/api/providers/upstox/token/exchange", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  upstoxTokenStatus: () => apiFetch<ApiUpstoxTokenStatus>("/api/providers/upstox/token/status"),
  upstoxTokenVerify: () =>
    apiFetch<{
      token_configured: boolean;
      token_masked?: string | null;
      expires_at?: string | null;
      is_expired?: boolean;
      expires_soon?: boolean;
      last_verified_at?: string | null;
      verification?: Record<string, unknown>;
    }>("/api/providers/upstox/token/verify"),
  upstoxDisconnect: () =>
    apiFetch<{ disconnected: boolean }>("/api/providers/upstox/disconnect", {
      method: "POST",
      body: JSON.stringify({}),
    }),
  dataCoverage: (bundleId: number, timeframe = "1d", topN = 50) =>
    apiFetch<ApiDataCoverage>(
      `/api/data/coverage?bundle_id=${bundleId}&timeframe=${encodeURIComponent(timeframe)}&top_n=${topN}`,
    ),
  dataQualityLatest: (bundleId: number, timeframe = "1d") =>
    apiFetch<ApiDataQualityReport>(
      `/api/data/quality/latest?bundle_id=${bundleId}&timeframe=${encodeURIComponent(timeframe)}`,
    ),
  dataQualityHistory: (params?: { bundle_id?: number; timeframe?: string; days?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (params?.days) search.set("days", String(params.days));
    return apiFetch<ApiDataQualityReport[]>(
      `/api/data/quality/history${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  regimeCurrent: (symbol?: string) =>
    apiFetch<{ symbol: string; timeframe: string; regime: string }>(
      `/api/regime/current${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ""}`,
    ),
  jobs: (limit = 20) => apiFetch<Job[]>(`/api/jobs?limit=${limit}`),
  job: (jobId: string) => apiFetch<Job>(`/api/jobs/${jobId}`),

  importData: (formData: FormData) =>
    apiFetch<JobStart>("/api/data/import", {
      method: "POST",
      body: formData,
    }),

  runBacktest: (payload: Record<string, unknown>) =>
    apiFetch<JobStart>("/api/backtests/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  backtests: (params?: {
    template?: string;
    timeframe?: string;
    start_date?: string;
    end_date?: string;
    sort_by?: string;
    sort_dir?: "asc" | "desc";
    page?: number;
    page_size?: number;
  }) => {
    const search = new URLSearchParams();
    if (params?.template) search.set("template", params.template);
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    if (params?.start_date) search.set("start_date", params.start_date);
    if (params?.end_date) search.set("end_date", params.end_date);
    if (params?.sort_by) search.set("sort_by", params.sort_by);
    if (params?.sort_dir) search.set("sort_dir", params.sort_dir);
    if (params?.page) search.set("page", String(params.page));
    if (params?.page_size) search.set("page_size", String(params.page_size));
    return apiFetch<ApiBacktestRunRow[]>(
      `/api/backtests${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  compareBacktests: (ids: number[], maxPoints = 1000) =>
    apiFetch<ApiBacktestComparison[]>(
      `/api/backtests/compare?ids=${encodeURIComponent(ids.join(","))}&max_points=${maxPoints}`,
    ),
  backtestById: (id: number) => apiFetch<ApiBacktest>(`/api/backtests/${id}`),
  backtestEquity: (id: number) =>
    apiFetch<Array<{ datetime: string; equity: number }>>(`/api/backtests/${id}/equity`),
  backtestTrades: (id: number, page = 1, pageSize = 50) =>
    apiFetch<ApiTrade[]>(`/api/backtests/${id}/trades?page=${page}&page_size=${pageSize}`),
  backtestTradesExportUrl: (id: number) => buildApiUrl(`/api/backtests/${id}/trades/export.csv`),
  backtestSummaryExportUrl: (id: number) => buildApiUrl(`/api/backtests/${id}/summary/export.json`),

  runWalkForward: (payload: Record<string, unknown>) =>
    apiFetch<JobStart>("/api/walkforward/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  walkForwardById: (id: number) => apiFetch<ApiWalkForwardRun>(`/api/walkforward/${id}`),
  walkForwardFolds: (id: number, page = 1, pageSize = 20) =>
    apiFetch<Array<Record<string, unknown>>>(
      `/api/walkforward/${id}/folds?page=${page}&page_size=${pageSize}`,
    ),

  promoteStrategy: (payload: Record<string, unknown>) =>
    apiFetch<{ strategy_id: number; status: string }>("/api/strategies/promote", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  runResearch: (payload: Record<string, unknown>) =>
    apiFetch<JobStart>("/api/research/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  researchRuns: (page = 1, pageSize = 20) =>
    apiFetch<ApiResearchRun[]>(`/api/research/runs?page=${page}&page_size=${pageSize}`),
  researchRunById: (id: number) => apiFetch<ApiResearchRun>(`/api/research/runs/${id}`),
  researchCandidates: (runId: number, page = 1, pageSize = 25) =>
    apiFetch<ApiResearchCandidate[]>(
      `/api/research/runs/${runId}/candidates?page=${page}&page_size=${pageSize}`,
    ),
  createPolicy: (payload: { research_run_id: number; name: string }) =>
    apiFetch<ApiPolicy>("/api/policies", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  policies: (page = 1, pageSize = 50) =>
    apiFetch<ApiPolicy[]>(`/api/policies?page=${page}&page_size=${pageSize}`),
  policyById: (id: number) => apiFetch<ApiPolicy>(`/api/policies/${id}`),
  ensembles: (params?: { page?: number; page_size?: number; bundle_id?: number }) => {
    const search = new URLSearchParams();
    if (params?.page) search.set("page", String(params.page));
    if (params?.page_size) search.set("page_size", String(params.page_size));
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    return apiFetch<ApiPolicyEnsemble[]>(
      `/api/ensembles${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  ensembleById: (id: number) => apiFetch<ApiPolicyEnsemble>(`/api/ensembles/${id}`),
  createEnsemble: (payload: { name: string; bundle_id: number; is_active?: boolean }) =>
    apiFetch<ApiPolicyEnsemble>("/api/ensembles", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  upsertEnsembleMembers: (
    id: number,
    payload: { members: Array<{ policy_id: number; weight: number; enabled?: boolean }> },
  ) =>
    apiFetch<ApiPolicyEnsemble>(`/api/ensembles/${id}/members`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  putEnsembleRegimeWeights: (
    id: number,
    payload: Record<string, Record<string, number>>,
  ) =>
    apiFetch<ApiPolicyEnsemble>(`/api/ensembles/${id}/regime-weights`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  setActiveEnsemble: (id: number) =>
    apiFetch<{ status: string; ensemble_id: number; ensemble_name: string }>(
      `/api/ensembles/${id}/set-active`,
      {
        method: "POST",
      },
    ),
  policyHealth: (id: number, windowDays: 20 | 60, refresh = true) =>
    apiFetch<ApiPolicyHealthSnapshot>(
      `/api/policies/${id}/health?window_days=${windowDays}&refresh=${String(refresh)}`,
    ),
  policiesHealth: () => apiFetch<ApiPolicyHealthSnapshot[]>("/api/policies/health"),
  promotePolicyToPaper: (id: number) =>
    apiFetch<{ policy_id: number; status: string; paper_mode: string; active_policy_id: number }>(
      `/api/policies/${id}/promote-to-paper`,
      {
        method: "POST",
      },
    ),

  paperState: () =>
    apiFetch<{ state: ApiPaperState; positions: ApiPaperPosition[]; orders: ApiPaperOrder[] }>(
      "/api/paper/state",
    ),
  paperPositions: () => apiFetch<ApiPaperPosition[]>("/api/paper/positions"),
  paperOrders: () => apiFetch<ApiPaperOrder[]>("/api/paper/orders"),
  paperRunStep: (payload: Record<string, unknown>) =>
    apiFetch<JobStart>("/api/paper/run-step", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  paperSignalsPreview: (payload: Record<string, unknown>) =>
    apiFetch<ApiPaperSignalPreview>("/api/paper/signals/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  operateStatus: () => apiFetch<ApiOperateStatus>("/api/operate/status"),
  operateEvents: (params?: {
    since?: string;
    severity?: "INFO" | "WARN" | "ERROR";
    category?: "DATA" | "EXECUTION" | "POLICY" | "SYSTEM";
    limit?: number;
  }) => {
    const search = new URLSearchParams();
    if (params?.since) search.set("since", params.since);
    if (params?.severity) search.set("severity", params.severity);
    if (params?.category) search.set("category", params.category);
    if (params?.limit) search.set("limit", String(params.limit));
    return apiFetch<ApiOperateEvent[]>(
      `/api/operate/events${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  operateHealth: (params?: { bundle_id?: number; timeframe?: string }) => {
    const search = new URLSearchParams();
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (params?.timeframe) search.set("timeframe", params.timeframe);
    return apiFetch<ApiOperateHealth>(
      `/api/operate/health${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  operateRun: (payload?: {
    bundle_id?: number;
    timeframe?: string;
    regime?: string;
    policy_id?: number;
    include_data_updates?: boolean;
    date?: string;
    asof?: string;
    seed?: number;
  }) =>
    apiFetch<JobStart & { summary?: ApiOperateRunSummary }>("/api/operate/run", {
      method: "POST",
      body: JSON.stringify(payload ?? {}),
    }),
  operateAutoEvalRun: (payload?: {
    bundle_id?: number;
    active_policy_id?: number;
    active_ensemble_id?: number;
    challenger_policy_ids?: number[];
    challenger_ensemble_ids?: number[];
    timeframe?: string;
    lookback_trading_days?: number;
    min_trades?: number;
    asof_date?: string;
    seed?: number;
    auto_switch?: boolean;
  }) =>
    apiFetch<JobStart>("/api/operate/auto-eval/run", {
      method: "POST",
      body: JSON.stringify(payload ?? {}),
    }),
  operateAutoEvalHistory: (
    page = 1,
    pageSize = 20,
    params?: { bundle_id?: number; policy_id?: number },
  ) => {
    const search = new URLSearchParams();
    search.set("page", String(page));
    search.set("page_size", String(pageSize));
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (typeof params?.policy_id === "number") search.set("policy_id", String(params.policy_id));
    return apiFetch<ApiAutoEvalRun[]>(`/api/operate/auto-eval/history?${search.toString()}`);
  },
  operateAutoEvalById: (id: number) => apiFetch<ApiAutoEvalRun>(`/api/operate/auto-eval/${id}`),
  operatePolicySwitches: (limit = 10) =>
    apiFetch<ApiPolicySwitchEvent[]>(`/api/operate/policy-switches?limit=${limit}`),
  generateDailyReport: (payload: { date?: string; bundle_id?: number; policy_id?: number }) =>
    apiFetch<JobStart>("/api/reports/daily/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  dailyReports: (params?: { date?: string; bundle_id?: number; policy_id?: number }) => {
    const search = new URLSearchParams();
    if (params?.date) search.set("date", params.date);
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (typeof params?.policy_id === "number") search.set("policy_id", String(params.policy_id));
    return apiFetch<ApiDailyReport[]>(
      `/api/reports/daily${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  dailyReportById: (id: number) => apiFetch<ApiDailyReport>(`/api/reports/daily/${id}`),
  dailyReportExportJson: (id: number) =>
    apiFetch<Record<string, unknown>>(`/api/reports/daily/${id}/export.json`),
  dailyReportExportJsonUrl: (id: number) => buildApiUrl(`/api/reports/daily/${id}/export.json`),
  dailyReportExportCsvUrl: (id: number) => buildApiUrl(`/api/reports/daily/${id}/export.csv`),
  dailyReportExportPdfUrl: (id: number) => buildApiUrl(`/api/reports/daily/${id}/export.pdf`),
  generateMonthlyReport: (payload: { month?: string; bundle_id?: number; policy_id?: number }) =>
    apiFetch<JobStart>("/api/reports/monthly/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  monthlyReports: (params?: { month?: string; bundle_id?: number; policy_id?: number }) => {
    const search = new URLSearchParams();
    if (params?.month) search.set("month", params.month);
    if (typeof params?.bundle_id === "number") search.set("bundle_id", String(params.bundle_id));
    if (typeof params?.policy_id === "number") search.set("policy_id", String(params.policy_id));
    return apiFetch<ApiMonthlyReport[]>(
      `/api/reports/monthly${search.toString() ? `?${search.toString()}` : ""}`,
    );
  },
  monthlyReportById: (id: number) => apiFetch<ApiMonthlyReport>(`/api/reports/monthly/${id}`),
  monthlyReportExportJsonUrl: (id: number) =>
    buildApiUrl(`/api/reports/monthly/${id}/export.json`),
  monthlyReportExportPdfUrl: (id: number) =>
    buildApiUrl(`/api/reports/monthly/${id}/export.pdf`),

  runEvaluation: (payload: {
    bundle_id: number;
    champion_policy_id: number;
    challenger_policy_ids?: number[];
    regime?: string;
    window_days?: number;
    start_date?: string;
    end_date?: string;
    seed?: number;
  }) =>
    apiFetch<JobStart>("/api/evaluations/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  evaluations: (page = 1, pageSize = 20) =>
    apiFetch<ApiPolicyEvaluation[]>(`/api/evaluations?page=${page}&page_size=${pageSize}`),
  evaluationById: (id: number) => apiFetch<ApiPolicyEvaluation>(`/api/evaluations/${id}`),
  evaluationDetails: (id: number) =>
    apiFetch<ApiPolicyShadowRun[]>(`/api/evaluations/${id}/details`),
  setActivePolicy: (id: number) =>
    apiFetch<{ policy_id: number; status: string; paper_mode: string; active_policy_id: number }>(
      `/api/policies/${id}/set-active`,
      {
        method: "POST",
      },
    ),

  runReplay: (payload: {
    bundle_id: number;
    policy_id: number;
    regime?: string;
    start_date?: string;
    end_date?: string;
    seed?: number;
    window_days?: number;
  }) =>
    apiFetch<JobStart>("/api/replay/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  replayRuns: (page = 1, pageSize = 20) =>
    apiFetch<ApiReplayRun[]>(`/api/replay/runs?page=${page}&page_size=${pageSize}`),
  replayRunById: (id: number) => apiFetch<ApiReplayRun>(`/api/replay/runs/${id}`),
  replayExportJson: (id: number) =>
    apiFetch<Record<string, unknown>>(`/api/replay/runs/${id}/export.json`),
  replayExportJsonUrl: (id: number) => buildApiUrl(`/api/replay/runs/${id}/export.json`),

  settings: () => apiFetch<Record<string, unknown>>("/api/settings"),
  updateSettings: (payload: Record<string, unknown>) =>
    apiFetch<{ settings: Record<string, unknown> }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
