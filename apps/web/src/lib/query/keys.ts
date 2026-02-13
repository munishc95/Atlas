export const qk = {
  universe: ["universe"] as const,
  universes: ["universes"] as const,
  strategies: ["strategies"] as const,
  strategyTemplates: ["strategyTemplates"] as const,
  dataStatus: ["dataStatus"] as const,
  regimeCurrent: (symbol?: string) => ["regimeCurrent", symbol ?? "default"] as const,
  jobs: (limit = 20) => ["jobs", limit] as const,
  job: (jobId: string | null) => ["job", jobId] as const,
  backtest: (id: number | null) => ["backtest", id] as const,
  backtestRuns: (filters: string) => ["backtestRuns", filters] as const,
  backtestCompare: (ids: number[]) => ["backtestCompare", ...ids] as const,
  backtestEquity: (id: number | null) => ["backtestEquity", id] as const,
  backtestTrades: (id: number | null, page: number, pageSize: number) =>
    ["backtestTrades", id, page, pageSize] as const,
  walkForward: (id: number | null) => ["walkForward", id] as const,
  walkForwardFolds: (id: number | null, page: number, pageSize: number) =>
    ["walkForwardFolds", id, page, pageSize] as const,
  researchRuns: (page: number, pageSize: number) => ["researchRuns", page, pageSize] as const,
  researchRun: (id: number | null) => ["researchRun", id] as const,
  researchCandidates: (id: number | null, page: number, pageSize: number) =>
    ["researchCandidates", id, page, pageSize] as const,
  policies: (page: number, pageSize: number) => ["policies", page, pageSize] as const,
  policy: (id: number | null) => ["policy", id] as const,
  policyHealth: (id: number | null, windowDays: number) =>
    ["policyHealth", id, windowDays] as const,
  policiesHealth: ["policiesHealth"] as const,
  paperState: ["paperState"] as const,
  paperPositions: ["paperPositions"] as const,
  paperOrders: ["paperOrders"] as const,
  paperSignalsPreview: (datasetId: number | null, regime: string) =>
    ["paperSignalsPreview", datasetId, regime] as const,
  operateStatus: ["operateStatus"] as const,
  dailyReports: (date?: string, bundleId?: number | null, policyId?: number | null) =>
    ["dailyReports", date ?? "latest", bundleId ?? "all", policyId ?? "all"] as const,
  dailyReport: (id: number | null) => ["dailyReport", id] as const,
  settings: ["settings"] as const,
};
