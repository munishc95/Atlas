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
  ensemble?: ApiPolicyEnsemble | null;
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

export type ApiPolicyEnsembleMember = {
  id: number | null;
  ensemble_id: number;
  policy_id: number;
  policy_name?: string | null;
  weight: number;
  enabled: boolean;
  created_at: string;
};

export type ApiPolicyEnsemble = {
  id: number;
  name: string;
  bundle_id: number;
  is_active: boolean;
  created_at: string;
  members: ApiPolicyEnsembleMember[];
  regime_weights?: Record<string, Record<string, number>>;
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
  safe_mode_on_fail?: boolean;
  safe_mode_action?: string;
  operate_mode?: string;
  calendar_segment?: string;
  calendar_today_ist?: string;
  calendar_is_trading_day_today?: boolean;
  calendar_session_today?: {
    open_time?: string | null;
    close_time?: string | null;
    is_special?: boolean;
    label?: string | null;
    is_trading_day?: boolean;
    holiday_name?: string | null;
  } | null;
  calendar_next_trading_day?: string | null;
  calendar_previous_trading_day?: string | null;
  auto_run_enabled?: boolean;
  auto_run_time_ist?: string;
  auto_run_include_data_updates?: boolean;
  last_auto_run_date?: string | null;
  next_scheduled_run_ist?: string | null;
  auto_eval_enabled?: boolean;
  auto_eval_frequency?: string;
  auto_eval_day_of_week?: number;
  auto_eval_time_ist?: string;
  last_auto_eval_date?: string | null;
  next_auto_eval_run_ist?: string | null;
  active_policy_id?: number | null;
  active_policy_name?: string | null;
  active_ensemble_id?: number | null;
  active_ensemble_name?: string | null;
  active_ensemble?: ApiPolicyEnsemble | null;
  active_bundle_id?: number | null;
  current_regime?: string | null;
  no_trade?: Record<string, unknown>;
  no_trade_triggered?: boolean;
  no_trade_reasons?: string[];
  confidence_gate?: Record<string, unknown>;
  latest_confidence_gate?: ApiConfidenceGateSnapshot | null;
  ensemble_weights_source?: string | null;
  ensemble_regime_used?: string | null;
  last_run_step_at?: string | null;
  latest_run?: Record<string, unknown> | null;
  latest_data_quality?: ApiDataQualityReport | null;
  latest_data_update?: ApiDataUpdateRun | null;
  latest_provider_update?: ApiProviderUpdateRun | null;
  upstox_token_status?: ApiUpstoxTokenStatus | null;
  upstox_token_request_latest?: ApiUpstoxTokenRequestRun | null;
  upstox_notifier_health?: ApiUpstoxNotifierHealth | null;
  upstox_auto_renew_enabled?: boolean;
  upstox_auto_renew_time_ist?: string;
  upstox_auto_renew_if_expires_within_hours?: number;
  upstox_auto_renew_lead_hours_before_open?: number;
  operate_provider_stage_on_token_invalid?: string;
  operate_last_upstox_auto_renew_date?: string | null;
  next_upstox_auto_renew_ist?: string | null;
  upstox_token_expires_within_hours?: number | null;
  provider_stage_status?: string | null;
  recent_event_counts_24h?: Record<string, number>;
  fast_mode_enabled?: boolean;
  last_job_durations?: Record<
    string,
    { duration_seconds: number; status: string; ts: string }
  >;
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
  coverage_by_source_json?: Record<string, number>;
  low_confidence_days_count?: number;
  low_confidence_symbols_count?: number;
  created_at: string;
};

export type ApiDataUpdateRun = {
  id: number;
  bundle_id?: number | null;
  timeframe: string;
  status: string;
  inbox_path: string;
  scanned_files: number;
  processed_files: number;
  skipped_files: number;
  rows_ingested: number;
  symbols_affected_json: string[];
  warnings_json: Array<Record<string, unknown>>;
  errors_json: Array<Record<string, unknown>>;
  created_at: string;
  ended_at?: string | null;
};

export type ApiProviderUpdateRun = {
  id: number;
  bundle_id?: number | null;
  timeframe: string;
  provider_kind: string;
  provider_mode?: string;
  provider_priority_json?: string[];
  status: string;
  symbols_attempted: number;
  symbols_succeeded: number;
  symbols_failed: number;
  bars_added: number;
  repaired_days_used: number;
  missing_days_detected: number;
  backfill_truncated: boolean;
  api_calls: number;
  coverage_before_pct?: number;
  coverage_after_pct?: number;
  by_provider_count_json?: Record<string, number>;
  confidence_distribution_json?: Record<string, number>;
  continuity_met?: boolean;
  duration_seconds: number;
  warnings_json: Array<Record<string, unknown>>;
  errors_json: Array<Record<string, unknown>>;
  created_at: string;
  ended_at?: string | null;
};

export type ApiDataProvenanceEntry = {
  id?: number | null;
  symbol: string;
  bar_date: string;
  source_provider: string;
  source_run_kind?: string;
  source_run_id?: string | null;
  confidence_score: number;
  reason?: string | null;
  metadata_json?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
};

export type ApiDataProvenance = {
  bundle_id: number;
  timeframe: string;
  symbol?: string | null;
  from?: string | null;
  to?: string | null;
  entries: ApiDataProvenanceEntry[];
  latest_day_summary?: {
    latest_day?: string | null;
    coverage_by_source_provider?: Record<string, number>;
    low_confidence_days_count?: number;
    low_confidence_symbols_count?: number;
    latest_day_all_low_confidence?: boolean;
    confidence_gate_latest?: ApiConfidenceGateSnapshot | null;
  };
};

export type ApiConfidenceGateSnapshot = {
  id?: number | null;
  created_at?: string | null;
  bundle_id?: number | null;
  timeframe: string;
  trading_date: string;
  decision: "PASS" | "SHADOW_ONLY" | "BLOCK_ENTRIES" | string;
  reasons: string[];
  avg_confidence: number;
  pct_low_confidence: number;
  provider_mix: Record<string, number>;
  threshold_used?: Record<string, unknown>;
};

export type ApiProviderStatusTrendDay = {
  trading_date: string;
  provider_counts: Record<string, number>;
  provider_mix: Record<string, number>;
  dominant_provider?: string | null;
  avg_confidence: number;
  pct_low_confidence: number;
  symbols: number;
};

export type ApiProviderStatusTrend = {
  bundle_id: number;
  timeframe: string;
  days: number;
  trend: ApiProviderStatusTrendDay[];
};

export type ApiProviderStatusItem = {
  provider: string;
  last_run_at?: string | null;
  last_status?: string;
  run_id?: number;
  timeframe?: string;
  bundle_id?: number | null;
  bars_added?: number;
  rows_ingested?: number;
  enabled?: boolean;
  token?: {
    connected: boolean;
    is_expired: boolean;
    expires_at?: string | null;
    last_verified_at?: string | null;
  };
  notes?: string[];
};

export type ApiProvidersStatus = {
  providers: ApiProviderStatusItem[];
  upstox_token_status?: Record<string, unknown>;
};

export type ApiUpstoxMappingStatus = {
  provider: string;
  bundle_id?: number | null;
  timeframe: string;
  mapped_count: number;
  missing_count: number;
  total_symbols: number;
  sample_missing_symbols: string[];
  last_import_at?: string | null;
  last_import_id?: number | null;
};

export type ApiUpstoxAuthUrl = {
  auth_url: string;
  state: string;
  redirect_uri: string;
  state_expires_at?: string | null;
  client_id_hint?: string | null;
};

export type ApiUpstoxTokenStatus = {
  provider_kind: string;
  connected: boolean;
  token_masked?: string | null;
  token_source?: string | null;
  issued_at?: string | null;
  expires_at?: string | null;
  is_expired: boolean;
  expires_soon: boolean;
  last_verified_at?: string | null;
  user_id?: string | null;
  auto_renew?: {
    enabled: boolean;
    time_ist?: string;
    if_expires_within_hours?: number;
    lead_hours_before_open?: number;
    only_when_provider_enabled?: boolean;
    last_run_date?: string | null;
    next_scheduled_run_ist?: string | null;
    expires_within_hours?: number | null;
  } | null;
  token_request_latest?: ApiUpstoxTokenRequestRun | null;
  notifier_health?: ApiUpstoxNotifierHealth | null;
};

export type ApiUpstoxTokenRequestRun = {
  id: string;
  provider_kind: string;
  status: "PENDING" | "APPROVED" | "REJECTED" | "EXPIRED" | "ERROR";
  status_legacy?: "REQUESTED" | "APPROVED" | "REJECTED" | "EXPIRED" | "FAILED";
  requested_at?: string | null;
  authorization_expiry?: string | null;
  approved_at?: string | null;
  resolved_at?: string | null;
  resolution_reason?: string | null;
  notifier_url?: string | null;
  client_id?: string | null;
  user_id?: string | null;
  correlation_nonce?: string | null;
  last_error?: Record<string, unknown> | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ApiUpstoxNotifierHealth = {
  last_notifier_received_at?: string | null;
  status: "OK" | "NEVER_RECEIVED" | "STALE" | "FAILING";
  pending_request?: {
    id: string;
    status: "PENDING" | "APPROVED" | "REJECTED" | "EXPIRED" | "ERROR";
    requested_at?: string | null;
    authorization_expiry?: string | null;
    minutes_waiting?: number;
  } | null;
  pending_no_callback?: boolean;
  pending_threshold_minutes?: number;
};

export type ApiUpstoxNotifierStatus = {
  recommended_notifier_url: string;
  legacy_notifier_url: string;
  legacy_route_security?: "less_secure" | string;
  secret_configured: boolean;
  webhook_health: ApiUpstoxNotifierHealth;
  last_request_run?: ApiUpstoxTokenRequestRun | null;
  suggested_actions: string[];
};

export type ApiUpstoxNotifierPing = {
  id?: string;
  ping_id: string;
  status: "SENT" | "RECEIVED" | "EXPIRED" | "UNKNOWN";
  source?: string;
  client_id?: string | null;
  notes?: string | null;
  created_at?: string | null;
  received_at?: string | null;
  expires_at?: string | null;
  ping_url?: string;
  ok?: boolean;
};

export type ApiUpstoxNotifierEvent = {
  id: string;
  received_at?: string | null;
  client_id?: string | null;
  user_id?: string | null;
  message_type?: string | null;
  issued_at?: string | null;
  expires_at?: string | null;
  payload_digest?: string | null;
  raw_payload_json?: Record<string, unknown>;
  headers_json?: Record<string, unknown>;
  correlated_request_run_id?: string | null;
  correlated_request_status?: "PENDING" | "APPROVED" | "REJECTED" | "EXPIRED" | "ERROR";
  correlated_resolution_reason?: string | null;
  created_at?: string | null;
};

export type ApiMappingImportRun = {
  id: number;
  provider: string;
  mode: string;
  source_path: string;
  file_hash: string;
  status: string;
  mapped_count: number;
  missing_count: number;
  inserted_count: number;
  updated_count: number;
  removed_count: number;
  warnings_json: Array<Record<string, unknown>>;
  errors_json: Array<Record<string, unknown>>;
  duration_seconds: number;
  created_at: string;
};

export type ApiDataCoverage = {
  bundle_id: number;
  timeframe: string;
  coverage_pct: number;
  missing_pct: number;
  status: "OK" | "WARN" | "FAIL";
  expected_latest_trading_day: string;
  missing_symbols: string[];
  stale_symbols: Array<{
    symbol: string;
    missing_trading_days: number;
    last_bar_day_ist?: string | null;
  }>;
  inactive_symbols: string[];
  last_bar_by_symbol: Array<{
    symbol: string;
    last_bar_ts?: string | null;
    missing_trading_days: number;
  }>;
  checked_symbols: number;
  total_symbols: number;
  last_bar_ts?: string | null;
  thresholds: {
    warn_pct: number;
    fail_pct: number;
    inactive_after_missing_days: number;
  };
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
  operate_mode?: string;
  calendar_segment?: string;
  calendar_today_ist?: string;
  calendar_is_trading_day_today?: boolean;
  calendar_session_today?: {
    open_time?: string | null;
    close_time?: string | null;
    is_special?: boolean;
    label?: string | null;
    is_trading_day?: boolean;
    holiday_name?: string | null;
  } | null;
  calendar_next_trading_day?: string | null;
  calendar_previous_trading_day?: string | null;
  auto_run_enabled?: boolean;
  auto_run_time_ist?: string;
  auto_run_include_data_updates?: boolean;
  last_auto_run_date?: string | null;
  next_scheduled_run_ist?: string | null;
  auto_eval_enabled?: boolean;
  auto_eval_frequency?: string;
  auto_eval_day_of_week?: number;
  auto_eval_time_ist?: string;
  last_auto_eval_date?: string | null;
  next_auto_eval_run_ist?: string | null;
  active_bundle_id?: number | null;
  active_timeframe: string;
  latest_data_quality?: ApiDataQualityReport | null;
  latest_data_update?: ApiDataUpdateRun | null;
  latest_provider_update?: ApiProviderUpdateRun | null;
  upstox_token_status?: ApiUpstoxTokenStatus | null;
  upstox_token_request_latest?: ApiUpstoxTokenRequestRun | null;
  upstox_notifier_health?: ApiUpstoxNotifierHealth | null;
  upstox_auto_renew_enabled?: boolean;
  upstox_auto_renew_time_ist?: string;
  upstox_auto_renew_if_expires_within_hours?: number;
  upstox_auto_renew_lead_hours_before_open?: number;
  operate_provider_stage_on_token_invalid?: string;
  operate_last_upstox_auto_renew_date?: string | null;
  next_upstox_auto_renew_ist?: string | null;
  upstox_token_expires_within_hours?: number | null;
  provider_stage_status?: string | null;
  latest_paper_run_id?: number | null;
  current_regime?: string | null;
  no_trade?: Record<string, unknown>;
  no_trade_triggered?: boolean;
  no_trade_reasons?: string[];
  latest_confidence_gate?: ApiConfidenceGateSnapshot | null;
  ensemble_weights_source?: string | null;
  ensemble_regime_used?: string | null;
  last_run_step_at?: string | null;
  recent_event_counts_24h: Record<string, number>;
  fast_mode_enabled?: boolean;
  last_job_durations?: Record<
    string,
    { duration_seconds: number; status: string; ts: string }
  >;
};

export type ApiOperateRunSummary = {
  bundle_id?: number | null;
  timeframe?: string;
  policy_id?: number | null;
  regime?: string;
  provider_stage_status?: string;
  mode?: "NORMAL" | "SAFE" | "SHADOW";
  quality_status?: string;
  update_status?: string;
  step_order?: string[];
  data_updates?: Record<string, unknown>;
  data_quality?: Record<string, unknown>;
  paper?: Record<string, unknown>;
  daily_report?: {
    status?: string;
    id?: number;
    date?: string;
  };
  durations_seconds?: Record<string, number>;
  steps?: Array<Record<string, unknown>>;
};

export type ApiAutoEvalRun = {
  id: number;
  ts: string;
  bundle_id: number;
  active_policy_id?: number | null;
  active_ensemble_id?: number | null;
  recommended_action: "KEEP" | "SWITCH" | "SHADOW_ONLY";
  recommended_entity_type?: "policy" | "ensemble" | null;
  recommended_policy_id?: number | null;
  recommended_ensemble_id?: number | null;
  reasons_json: string[];
  score_table_json: Record<string, unknown>;
  lookback_days: number;
  digest: string;
  status: string;
  auto_switch_attempted: boolean;
  auto_switch_applied: boolean;
  details_json: Record<string, unknown>;
};

export type ApiPolicySwitchEvent = {
  id: number;
  ts: string;
  from_policy_id: number;
  to_policy_id: number;
  reason: string;
  auto_eval_id?: number | null;
  cooldown_state_json: Record<string, unknown>;
  mode: string;
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
