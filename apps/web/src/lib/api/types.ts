export type ErrorBody = {
  code: string;
  message: string;
  details?: unknown;
};

export type ErrorEnvelope = {
  error: ErrorBody;
};

export type DataEnvelope<T> = {
  data: T;
  meta?: Record<string, unknown>;
};

export type JobStatus = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "DONE";

export type Job = {
  id: string;
  type: string;
  status: JobStatus;
  progress: number;
  started_at: string | null;
  ended_at: string | null;
  logs_json: string[];
  result_json?: Record<string, unknown> | null;
};

export type ApiBacktestRun = {
  backtest_id: number;
  symbol: string;
  timeframe: string;
  metrics: Record<string, number>;
  trade_count: number;
};

export type ApiBacktest = {
  id: number;
  symbol: string;
  timeframe: string;
  metrics_json: Record<string, number>;
  config_json: Record<string, unknown>;
  created_at: string;
};

export type ApiBacktestRunRow = {
  id: number;
  strategy_template: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  created_at: string;
  metrics: Record<string, number>;
  status: string;
};

export type ApiBacktestComparison = {
  id: number;
  label: string;
  metrics: Record<string, number>;
  equity_curve: Array<{ datetime: string; equity: number }>;
};

export type ApiTrade = {
  id: number;
  symbol: string;
  entry_dt: string;
  exit_dt: string;
  qty: number;
  entry_px: number;
  exit_px: number;
  pnl: number;
  r_multiple: number;
  reason: string;
};

export type ApiWalkForwardRun = {
  id: number;
  summary_json: Record<string, unknown>;
  config_json: Record<string, unknown>;
};

export type ApiPaperPosition = {
  id: number;
  symbol: string;
  side: "BUY" | "SELL";
  instrument_kind: string;
  lot_size: number;
  must_exit_by_eod: boolean;
  qty: number;
  avg_price: number;
  stop_price?: number | null;
  target_price?: number | null;
  opened_at: string;
};

export type ApiPaperOrder = {
  id: number;
  symbol: string;
  side: "BUY" | "SELL";
  instrument_kind: string;
  lot_size: number;
  qty: number;
  fill_price?: number | null;
  status: string;
  reason?: string | null;
  created_at: string;
};

export type ApiPaperSignal = {
  symbol: string;
  side: "BUY" | "SELL";
  template: string;
  timeframe?: string;
  instrument_kind?: string;
  lot_size?: number;
  price: number;
  stop_distance: number;
  target_price?: number | null;
  signal_strength: number;
  adv: number;
  vol_scale: number;
  explanation?: string;
  correlations?: Record<string, number>;
};

export type ApiPaperSignalPreview = {
  regime: string;
  policy_mode: string;
  policy_selection_reason: string;
  signals_source: string;
  generated_signals_count: number;
  selected_signals_count: number;
  bundle_id?: number;
  dataset_id?: number;
  timeframes?: string[];
  symbol_scope?: string;
  scan_truncated?: boolean;
  scanned_symbols?: number;
  evaluated_candidates?: number;
  total_symbols?: number;
  signals: ApiPaperSignal[];
  skipped_signals?: Array<Record<string, unknown>>;
};

export type ApiPaperState = {
  id: number;
  equity: number;
  cash: number;
  peak_equity: number;
  drawdown: number;
  kill_switch_active: boolean;
  cooldown_days_left: number;
  settings_json: Record<string, unknown>;
};

export type ApiResearchRun = {
  id: number;
  created_at: string;
  bundle_id?: number | null;
  dataset_id?: number | null;
  timeframes_json: string[];
  config_json: Record<string, unknown>;
  status: string;
  summary_json: Record<string, unknown>;
};

export type ApiResearchCandidate = {
  id: number;
  run_id: number;
  symbol: string;
  timeframe: string;
  strategy_key: string;
  best_params_json: Record<string, unknown>;
  oos_metrics_json: Record<string, number>;
  stress_metrics_json: Record<string, number>;
  param_dispersion: number;
  fold_variance: number;
  stress_pass_rate: number;
  score: number;
  rank: number;
  accepted: boolean;
  explanations_json: string[];
};

export type ApiPolicy = {
  id: number;
  name: string;
  created_at: string;
  definition_json: Record<string, unknown>;
  promoted_from_research_run_id?: number | null;
};

export type ApiUniverseBundle = {
  id: number;
  name: string;
  provider: string;
  description?: string | null;
  symbols: string[];
  supported_timeframes: string[];
  created_at: string;
};
