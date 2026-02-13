"use client";

import Link from "next/link";

import { metricHelpText, type MetricHelpKey } from "@/lib/metric-help";

export function MetricHelpTooltip({ metricKey }: { metricKey: MetricHelpKey }) {
  const copy = metricHelpText[metricKey];

  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-label={`Explain ${copy.title}`}
        className="focus-ring ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] text-muted"
      >
        ?
      </button>
      <span className="pointer-events-none absolute left-0 top-5 z-20 hidden w-60 rounded-xl border border-border bg-panel p-2 text-xs text-muted shadow-lg group-hover:block group-focus-within:block">
        <span className="block font-semibold text-ink">{copy.title}</span>
        <span className="mt-1 block">{copy.body}</span>
        <Link
          href={`/help#${metricKey}`}
          className="pointer-events-auto mt-2 inline-block text-accent underline"
        >
          Learn more
        </Link>
      </span>
    </span>
  );
}
