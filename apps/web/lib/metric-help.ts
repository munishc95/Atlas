export type MetricHelpKey = "calmar" | "max_drawdown" | "cvar_95" | "sharpe" | "sortino";

export const metricHelpText: Record<MetricHelpKey, { title: string; body: string }> = {
  calmar: {
    title: "Calmar",
    body: "Return divided by worst drawdown. Higher means better risk-adjusted compounding.",
  },
  max_drawdown: {
    title: "Max Drawdown",
    body: "Largest peak-to-trough equity drop. Lower magnitude is safer.",
  },
  cvar_95: {
    title: "CVaR 95%",
    body: "Average loss in the worst 5% outcomes. More negative means fatter downside tail.",
  },
  sharpe: {
    title: "Sharpe",
    body: "Return per unit of volatility. Higher is better if drawdowns are acceptable.",
  },
  sortino: {
    title: "Sortino",
    body: "Return per unit of downside volatility. Focuses on harmful volatility only.",
  },
};

export function asMetricHelpKey(label: string): MetricHelpKey | null {
  const normalized = label.trim().toLowerCase();
  if (normalized === "calmar") return "calmar";
  if (normalized === "max_drawdown") return "max_drawdown";
  if (normalized === "cvar_95") return "cvar_95";
  if (normalized === "sharpe") return "sharpe";
  if (normalized === "sortino") return "sortino";
  return null;
}
