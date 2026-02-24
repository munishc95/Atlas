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
  ensembles: (page = 1, pageSize = 20, bundleId?: number | null) =>
    ["ensembles", page, pageSize, bundleId ?? "all"] as const,
  ensemble: (id: number | null) => ["ensemble", id] as const,
  policyHealth: (id: number | null, windowDays: number) =>
    ["policyHealth", id, windowDays] as const,
  policiesHealth: ["policiesHealth"] as const,
  paperState: ["paperState"] as const,
  paperPositions: ["paperPositions"] as const,
  paperOrders: ["paperOrders"] as const,
  paperSignalsPreview: (datasetId: number | null, regime: string) =>
    ["paperSignalsPreview", datasetId, regime] as const,
  operateStatus: ["operateStatus"] as const,
  operateHealth: (bundleId?: number | null, timeframe?: string | null) =>
    ["operateHealth", bundleId ?? "active", timeframe ?? "active"] as const,
  operateEvents: (severity?: string | null, category?: string | null, limit = 20) =>
    ["operateEvents", severity ?? "all", category ?? "all", limit] as const,
  operateAutoEvalHistory: (
    page = 1,
    pageSize = 20,
    bundleId?: number | null,
    policyId?: number | null,
  ) => ["operateAutoEvalHistory", page, pageSize, bundleId ?? "all", policyId ?? "all"] as const,
  operateAutoEval: (id: number | null) => ["operateAutoEval", id] as const,
  operatePolicySwitches: (limit = 10) => ["operatePolicySwitches", limit] as const,
  dataQualityLatest: (bundleId?: number | null, timeframe = "1d") =>
    ["dataQualityLatest", bundleId ?? "none", timeframe] as const,
  dataQualityHistory: (bundleId?: number | null, timeframe?: string | null, days = 7) =>
    ["dataQualityHistory", bundleId ?? "all", timeframe ?? "all", days] as const,
  dataUpdatesLatest: (bundleId?: number | null, timeframe = "1d") =>
    ["dataUpdatesLatest", bundleId ?? "none", timeframe] as const,
  dataUpdatesHistory: (bundleId?: number | null, timeframe?: string | null, days = 7) =>
    ["dataUpdatesHistory", bundleId ?? "all", timeframe ?? "all", days] as const,
  providerUpdatesLatest: (bundleId?: number | null, timeframe?: string | null) =>
    ["providerUpdatesLatest", bundleId ?? "all", timeframe ?? "all"] as const,
  providerUpdatesHistory: (bundleId?: number | null, timeframe?: string | null, days = 7) =>
    ["providerUpdatesHistory", bundleId ?? "all", timeframe ?? "all", days] as const,
  upstoxMappingStatus: (bundleId?: number | null, timeframe?: string | null, sampleLimit = 20) =>
    ["upstoxMappingStatus", bundleId ?? "all", timeframe ?? "all", sampleLimit] as const,
  upstoxMappingMissing: (bundleId?: number | null, timeframe?: string | null, limit = 50) =>
    ["upstoxMappingMissing", bundleId ?? "all", timeframe ?? "all", limit] as const,
  upstoxTokenStatus: ["upstoxTokenStatus"] as const,
  upstoxTokenRequestLatest: ["upstoxTokenRequestLatest"] as const,
  upstoxTokenRequestHistory: (page = 1, pageSize = 10) =>
    ["upstoxTokenRequestHistory", page, pageSize] as const,
  upstoxNotifierStatus: ["upstoxNotifierStatus"] as const,
  upstoxNotifierEvents: (limit = 20, offset = 0) =>
    ["upstoxNotifierEvents", limit, offset] as const,
  upstoxNotifierPing: (pingId: string | null) => ["upstoxNotifierPing", pingId ?? "none"] as const,
  dataCoverage: (bundleId?: number | null, timeframe = "1d", topN = 50) =>
    ["dataCoverage", bundleId ?? "none", timeframe, topN] as const,
  dataProvenance: (
    bundleId?: number | null,
    timeframe = "1d",
    symbol?: string | null,
    from?: string | null,
    to?: string | null,
    limit = 500,
  ) =>
    [
      "dataProvenance",
      bundleId ?? "none",
      timeframe,
      symbol ?? "all",
      from ?? "none",
      to ?? "none",
      limit,
    ] as const,
  providersStatus: ["providersStatus"] as const,
  providersStatusTrend: (bundleId?: number | null, timeframe = "1d", days = 30) =>
    ["providersStatusTrend", bundleId ?? "none", timeframe, days] as const,
  confidenceGateLatest: (bundleId?: number | null, timeframe?: string | null) =>
    ["confidenceGateLatest", bundleId ?? "all", timeframe ?? "all"] as const,
  confidenceGateHistory: (bundleId?: number | null, timeframe?: string | null, limit = 60) =>
    ["confidenceGateHistory", bundleId ?? "all", timeframe ?? "all", limit] as const,
  dailyReports: (date?: string, bundleId?: number | null, policyId?: number | null) =>
    ["dailyReports", date ?? "latest", bundleId ?? "all", policyId ?? "all"] as const,
  dailyReport: (id: number | null) => ["dailyReport", id] as const,
  monthlyReports: (month?: string, bundleId?: number | null, policyId?: number | null) =>
    ["monthlyReports", month ?? "latest", bundleId ?? "all", policyId ?? "all"] as const,
  monthlyReport: (id: number | null) => ["monthlyReport", id] as const,
  evaluations: (page: number, pageSize: number) => ["evaluations", page, pageSize] as const,
  evaluation: (id: number | null) => ["evaluation", id] as const,
  evaluationDetails: (id: number | null) => ["evaluationDetails", id] as const,
  replayRuns: (page: number, pageSize: number) => ["replayRuns", page, pageSize] as const,
  replayRun: (id: number | null) => ["replayRun", id] as const,
  settings: ["settings"] as const,
};
