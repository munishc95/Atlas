import Link from "next/link";

import { metricHelpText } from "@/lib/metric-help";

export default function HelpPage() {
  return (
    <div className="space-y-5">
      <section className="card p-4">
        <h2 className="text-xl font-semibold">Metric Help</h2>
        <p className="mt-1 text-sm text-muted">
          Quick definitions for core backtest and walk-forward metrics shown in Atlas.
        </p>
      </section>

      {Object.entries(metricHelpText).map(([key, item]) => (
        <section key={key} id={key} className="card p-4">
          <h3 className="text-base font-semibold">{item.title}</h3>
          <p className="mt-1 text-sm text-muted">{item.body}</p>
        </section>
      ))}

      <section className="card p-4 text-sm text-muted">
        <p>
          Need deeper context? Use the strategy and walk-forward docs in `docs/` for implementation
          details.
        </p>
        <Link href="/strategy-lab" className="mt-2 inline-block text-accent underline">
          Back to Strategy Lab
        </Link>
      </section>
    </div>
  );
}
