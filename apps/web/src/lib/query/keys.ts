export const qk = {
  universe: ["universe"] as const,
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
  paperState: ["paperState"] as const,
  paperPositions: ["paperPositions"] as const,
  paperOrders: ["paperOrders"] as const,
  settings: ["settings"] as const,
};
