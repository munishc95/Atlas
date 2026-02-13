import { apiFetch, buildApiUrl } from "@/src/lib/api/client";
import type {
  ApiBacktest,
  ApiBacktestComparison,
  ApiBacktestRunRow,
  ApiPaperOrder,
  ApiPaperPosition,
  ApiPaperState,
  ApiTrade,
  ApiWalkForwardRun,
  Job,
} from "@/src/lib/api/types";

export type JobStart = { job_id: string; status: string };

export const atlasApi = {
  health: () => apiFetch<{ status: string }>("/api/health"),
  universe: () => apiFetch<{ symbols: string[] }>("/api/universe"),
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
    return apiFetch<ApiBacktestRunRow[]>(`/api/backtests${search.toString() ? `?${search.toString()}` : ""}`);
  },
  compareBacktests: (ids: number[], maxPoints = 1000) =>
    apiFetch<ApiBacktestComparison[]>(
      `/api/backtests/compare?ids=${encodeURIComponent(ids.join(","))}&max_points=${maxPoints}`,
    ),
  backtestById: (id: number) => apiFetch<ApiBacktest>(`/api/backtests/${id}`),
  backtestEquity: (id: number) => apiFetch<Array<{ datetime: string; equity: number }>>(`/api/backtests/${id}/equity`),
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
    apiFetch<Array<Record<string, unknown>>>(`/api/walkforward/${id}/folds?page=${page}&page_size=${pageSize}`),

  promoteStrategy: (payload: Record<string, unknown>) =>
    apiFetch<{ strategy_id: number; status: string }>("/api/strategies/promote", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

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

  settings: () => apiFetch<Record<string, unknown>>("/api/settings"),
  updateSettings: (payload: Record<string, unknown>) =>
    apiFetch<{ settings: Record<string, unknown> }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
