import { MetricHelpTooltip } from "@/components/metric-help-tooltip";
import type { MetricHelpKey } from "@/lib/metric-help";

type TileProps = {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "success" | "warning";
  helpKey?: MetricHelpKey;
};

export function MetricTile({ label, value, hint, tone = "default", helpKey }: TileProps) {
  const toneClass =
    tone === "success" ? "text-success" : tone === "warning" ? "text-warning" : "text-ink";

  return (
    <article className="card p-4">
      <p className="inline-flex items-center text-xs uppercase tracking-[0.08em] text-muted">
        {label}
        {helpKey ? <MetricHelpTooltip metricKey={helpKey} /> : null}
      </p>
      <p className={`mt-2 text-2xl font-semibold ${toneClass}`}>{value}</p>
      {hint ? <p className="mt-1 text-xs text-muted">{hint}</p> : null}
    </article>
  );
}
