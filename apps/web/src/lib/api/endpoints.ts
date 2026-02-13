import { apiFetch, buildApiUrl } from "@/src/lib/api/client";
import type {
  ApiBacktest,
  ApiBacktestComparison,
  ApiBacktestRunRow,
  ApiPaperOrder,
  ApiPaperPosition,
  ApiPaperSignalPreview,
  ApiPaperState,
  ApiPolicyHealthSnapshot,
  ApiPolicy,
  ApiDailyReport,
  ApiDataQualityReport,
  ApiOperateEvent,
  ApiOperateHealth,
  ApiOperateStatus,
  ApiMonthlyReport,
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
