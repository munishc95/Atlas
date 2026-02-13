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
  qty: number;
  avg_price: number;
  stop_price?: number | null;
  target_price?: number | null;
  opened_at: string;
};

export type ApiPaperOrder = {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  fill_price?: number | null;
  status: string;
  reason?: string | null;
  created_at: string;
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
