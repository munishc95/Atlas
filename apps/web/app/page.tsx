"use client";

import { useEffect, useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { EquityChart } from "@/components/equity-chart";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { MetricTile } from "@/components/metric-tile";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

export default function DashboardPage() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const regimeQuery = useQuery({
    queryKey: qk.regimeCurrent(),
    queryFn: async () => (await atlasApi.regimeCurrent()).data,
  });

  const paperStateQuery = useQuery({
    queryKey: qk.paperState,
    queryFn: async () => (await atlasApi.paperState()).data,
  });

  const strategiesQuery = useQuery({
    queryKey: qk.strategies,
    queryFn: async () => (await atlasApi.strategies()).data,
  });

  const jobsQuery = useQuery({
    queryKey: qk.jobs(20),
    queryFn: async () => (await atlasApi.jobs(20)).data,
    refetchInterval: 3_000,
  });

  const latestBacktestId = useMemo(() => {
    const jobs = jobsQuery.data ?? [];
    const backtestJob = jobs.find(
      (job) => job.type === "backtest" && ["SUCCEEDED", "DONE"].includes(job.status),
    );
    const result = backtestJob?.result_json as { backtest_id?: number } | undefined;
    return result?.backtest_id ?? null;
  }, [jobsQuery.data]);

  const equityQuery = useQuery({
    queryKey: qk.backtestEquity(latestBacktestId),
    queryFn: async () => (await atlasApi.backtestEquity(latestBacktestId as number)).data,
    enabled: latestBacktestId !== null,
  });

  useEffect(() => {
    if (activeJobId) {
      return;
    }
    const running = (jobsQuery.data ?? []).find((job) => ["QUEUED", "RUNNING"].includes(job.status));
    if (running) {
      setActiveJobId(running.id);
    }
  }, [jobsQuery.data, activeJobId]);

  const equityPoints =
    equityQuery.data?.map((row) => ({ time: row.datetime.slice(0, 10), value: row.equity })) ?? [];

  const positions = paperStateQuery.data?.positions ?? [];
  const state = paperStateQuery.data?.state;

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Active Job" />

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile
          label="Portfolio equity"
          value={state ? `INR ${state.equity.toLocaleString("en-IN")}` : "-"}
          hint="Paper account"
          tone="success"
        />
        <MetricTile
          label="Cash"
          value={state ? `INR ${state.cash.toLocaleString("en-IN")}` : "-"}
          hint={`Risk used: ${state ? `${Math.max(0, (1 - state.cash / state.equity) * 100).toFixed(2)}%` : "-"}`}
        />
        <MetricTile label="Open positions" value={`${positions.length}`} hint="Max 3 positions" />
        <MetricTile
          label="Current regime"
          value={regimeQuery.data?.regime ?? "-"}
          hint="Allocator active"
          tone="success"
        />
      </section>

      <section className="card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Portfolio equity curve</h2>
          <span className="badge bg-success/15 text-success">Paper</span>
        </div>

        {equityQuery.isLoading ? (
          <LoadingState label="Loading equity series" />
        ) : equityPoints.length === 0 ? (
          <EmptyState title="No equity curve yet" action="Run a backtest in Strategy Lab to populate this chart." />
        ) : (
          <EquityChart points={equityPoints} />
        )}
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="card p-4 lg:col-span-2">
          <h2 className="text-lg font-semibold">Active strategies</h2>
          {strategiesQuery.isLoading ? (
            <LoadingState label="Loading strategies" />
          ) : strategiesQuery.isError ? (
            <ErrorState
              title="Could not load strategies"
              action="Retry in a few seconds."
              onRetry={() => void strategiesQuery.refetch()}
            />
          ) : (strategiesQuery.data ?? []).length === 0 ? (
            <EmptyState title="No promoted strategy" action="Run walk-forward and promote a strategy." />
          ) : (
            <div className="mt-3 overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-3 py-2 font-medium">Template</th>
                    <th className="px-3 py-2 font-medium">Status</th>
                    <th className="px-3 py-2 font-medium">Promoted At</th>
                  </tr>
                </thead>
                <tbody>
                  {strategiesQuery.data?.slice(0, 8).map((strategy) => (
                    <tr key={String(strategy.id)} className="border-t border-border">
                      <td className="px-3 py-2">{String(strategy.template ?? "-")}</td>
                      <td className="px-3 py-2 text-success">Enabled</td>
                      <td className="px-3 py-2">{String(strategy.promoted_at ?? "-")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <article className="card p-4">
          <h2 className="text-lg font-semibold">Recent jobs</h2>
          {jobsQuery.isLoading ? (
            <LoadingState label="Loading jobs" />
          ) : jobsQuery.isError ? (
            <ErrorState
              title="Could not load jobs"
              action="Check API connectivity."
              onRetry={() => void jobsQuery.refetch()}
            />
          ) : (jobsQuery.data ?? []).length === 0 ? (
            <EmptyState title="No jobs yet" action="Run an import or backtest job." />
          ) : (
            <ul className="mt-3 space-y-2 text-sm text-muted">
              {jobsQuery.data?.slice(0, 6).map((job) => (
                <li key={job.id} className="rounded-lg border border-border px-3 py-2">
                  <button
                    type="button"
                    className="w-full text-left"
                    onClick={() => setActiveJobId(job.id)}
                  >
                    <span className="font-medium text-ink">{job.type}</span>
                    <span className="ml-2 text-xs">{job.status}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </article>
      </section>

      {(regimeQuery.isError || paperStateQuery.isError) && (
        <ErrorState
          title="Dashboard data unavailable"
          action="Confirm API is running on NEXT_PUBLIC_API_BASE_URL and retry."
          onRetry={() => {
            void regimeQuery.refetch();
            void paperStateQuery.refetch();
          }}
        />
      )}
    </div>
  );
}
