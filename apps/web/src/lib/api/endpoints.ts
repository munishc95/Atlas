import { apiFetch, buildApiUrl } from "@/src/lib/api/client";
import type {
  ApiBacktest,
  ApiBacktestComparison,
  ApiBacktestRunRow,
  ApiPaperOrder,
  ApiPaperPosition,
  ApiPaperSignalPreview,
  ApiPaperState,
  ApiPolicy,
  ApiResearchCandidate,
  ApiResearchRun,
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

  settings: () => apiFetch<Record<string, unknown>>("/api/settings"),
  updateSettings: (payload: Record<string, unknown>) =>
    apiFetch<{ settings: Record<string, unknown> }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
