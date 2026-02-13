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
  qty_lots: number;
  margin_reserved: number;
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
  qty_lots: number;
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
  policy_status?: string;
  health_status?: string;
  health_reasons?: string[];
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

export type ApiPolicyHealthSnapshot = {
  id: number;
  policy_id: number;
  asof_date: string;
  window_days: number;
  metrics_json: Record<string, number>;
  status: string;
  reasons_json: string[];
  created_at: string;
};

export type ApiDailyReport = {
  id: number;
  date: string;
  bundle_id?: number | null;
  policy_id?: number | null;
  content_json: Record<string, unknown>;
  created_at: string;
};

export type ApiMonthlyReport = {
  id: number;
  month: string;
  bundle_id?: number | null;
  policy_id?: number | null;
  content_json: Record<string, unknown>;
  created_at: string;
};

export type ApiPolicyEvaluation = {
  id: number;
  created_at: string;
  bundle_id: number;
  regime?: string | null;
  window_start: string;
  window_end: string;
  champion_policy_id: number;
  challenger_policy_ids_json: number[];
  status: string;
  summary_json: Record<string, unknown>;
  notes?: string | null;
};

export type ApiPolicyShadowRun = {
  id: number;
  evaluation_id: number;
  policy_id: number;
  asof_date: string;
  run_summary_json: Record<string, unknown>;
  created_at: string;
};

export type ApiReplayRun = {
  id: number;
  created_at: string;
  bundle_id: number;
  policy_id: number;
  regime?: string | null;
  start_date: string;
  end_date: string;
  seed: number;
  status: string;
  summary_json: Record<string, unknown>;
};

export type ApiOperateStatus = {
  mode?: "NORMAL" | "SAFE MODE";
  mode_reason?: string | null;
  active_policy_id?: number | null;
  active_policy_name?: string | null;
  active_bundle_id?: number | null;
  current_regime?: string | null;
  last_run_step_at?: string | null;
  latest_run?: Record<string, unknown> | null;
  latest_data_quality?: ApiDataQualityReport | null;
  recent_event_counts_24h?: Record<string, number>;
  health_short?: ApiPolicyHealthSnapshot | null;
  health_long?: ApiPolicyHealthSnapshot | null;
  paper_state: Record<string, unknown>;
};

export type ApiDataQualityIssue = {
  severity: "OK" | "WARN" | "FAIL";
  code: string;
  message: string;
  symbol?: string;
  details?: Record<string, unknown>;
};

export type ApiDataQualityReport = {
  id: number;
  bundle_id?: number | null;
  timeframe: string;
  status: "OK" | "WARN" | "FAIL";
  issues_json: ApiDataQualityIssue[];
  last_bar_ts?: string | null;
  coverage_pct: number;
  checked_symbols: number;
  total_symbols: number;
  created_at: string;
};

export type ApiOperateEvent = {
  id: number;
  ts: string;
  severity: "INFO" | "WARN" | "ERROR";
  category: "DATA" | "EXECUTION" | "POLICY" | "SYSTEM";
  message: string;
  details_json?: Record<string, unknown>;
  correlation_id?: string | null;
};

export type ApiOperateHealth = {
  mode: "NORMAL" | "SAFE MODE";
  mode_reason?: string | null;
  safe_mode_on_fail: boolean;
  safe_mode_action: string;
  active_bundle_id?: number | null;
  active_timeframe: string;
  latest_data_quality?: ApiDataQualityReport | null;
  latest_paper_run_id?: number | null;
  last_run_step_at?: string | null;
  recent_event_counts_24h: Record<string, number>;
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
