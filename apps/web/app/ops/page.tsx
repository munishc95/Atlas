"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useJobStream } from "@/src/hooks/useJobStream";
import { atlasApi } from "@/src/lib/api/endpoints";
import type { ApiOperateEvent } from "@/src/lib/api/types";
import { qk } from "@/src/lib/query/keys";

function badgeTone(status: string): string {
  const token = status.toUpperCase();
  if (token.includes("ERROR") || token.includes("FAIL") || token.includes("SAFE")) {
    return "bg-danger/15 text-danger";
  }
  if (token.includes("WARN")) {
    return "bg-warning/15 text-warning";
  }
  return "bg-success/15 text-success";
}

export default function OpsPage() {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<ApiOperateEvent | null>(null);

  const statusQuery = useQuery({
    queryKey: qk.operateStatus,
    queryFn: async () => (await atlasApi.operateStatus()).data,
    refetchInterval: 8_000,
  });
  const healthQuery = useQuery({
    queryKey: qk.operateHealth(null, null),
    queryFn: async () => (await atlasApi.operateHealth()).data,
    refetchInterval: 8_000,
  });
  const eventsQuery = useQuery({
    queryKey: qk.operateEvents(null, null, 20),
    queryFn: async () => (await atlasApi.operateEvents({ limit: 20 })).data,
    refetchInterval: 8_000,
  });

  const activeBundleId =
    (statusQuery.data?.active_bundle_id as number | null | undefined) ??
    (healthQuery.data?.active_bundle_id as number | null | undefined) ??
    null;
  const activePolicyId = (statusQuery.data?.active_policy_id as number | null | undefined) ?? null;
  const activeTimeframe = String(healthQuery.data?.active_timeframe ?? "1d");

  const runQualityMutation = useMutation({
    mutationFn: async () => {
      if (!activeBundleId) {
        throw new Error("Active bundle is required before running data quality.");
      }
      return (
        await atlasApi.runDataQuality({
          bundle_id: activeBundleId,
          timeframe: activeTimeframe,
        })
      ).data;
    },
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Data quality job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue data quality job");
    },
  });

  const dailyReportMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.generateDailyReport({
          date: new Date().toISOString().slice(0, 10),
          bundle_id: activeBundleId ?? undefined,
          policy_id: activePolicyId ?? undefined,
        })
      ).data,
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Daily report job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue daily report");
    },
  });

  const replayMutation = useMutation({
    mutationFn: async () => {
      if (!activeBundleId || !activePolicyId) {
        throw new Error("Active bundle and policy are required for replay.");
      }
      const end = new Date();
      const start = new Date();
      start.setDate(end.getDate() - 19);
      return (
        await atlasApi.runReplay({
          bundle_id: activeBundleId,
          policy_id: activePolicyId,
          start_date: start.toISOString().slice(0, 10),
          end_date: end.toISOString().slice(0, 10),
          seed: 7,
        })
      ).data;
    },
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Replay job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue replay");
    },
  });

  const stream = useJobStream(jobId);
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
      queryClient.invalidateQueries({ queryKey: qk.operateEvents(null, null, 20) });
      if (activeBundleId) {
        queryClient.invalidateQueries({ queryKey: qk.dataQualityLatest(activeBundleId, activeTimeframe) });
        queryClient.invalidateQueries({ queryKey: qk.dataQualityHistory(activeBundleId, activeTimeframe, 7) });
      }
      toast.success("Ops action complete");
      return;
    }
    if (stream.status === "FAILED") {
      toast.error("Ops action failed");
    }
  }, [
    activeBundleId,
    activeTimeframe,
    queryClient,
    stream.isTerminal,
    stream.status,
  ]);

  const mode = String(healthQuery.data?.mode ?? statusQuery.data?.mode ?? "NORMAL");
  const latestQuality = healthQuery.data?.latest_data_quality ?? statusQuery.data?.latest_data_quality ?? null;
  const eventCounts = healthQuery.data?.recent_event_counts_24h ?? {};
  const events = useMemo(() => eventsQuery.data ?? [], [eventsQuery.data]);

  return (
    <div className="space-y-5">
      <JobDrawer jobId={jobId} onClose={() => setJobId(null)} title="Ops Job" />

      <section className="card p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold">Operate Mode</h2>
            <p className="mt-1 text-sm text-muted">Operational trust, guardrails, and explainable safety controls.</p>
          </div>
          <span className={`badge ${badgeTone(mode)}`}>{mode}</span>
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Active bundle: {activeBundleId ?? "-"}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Active policy: {activePolicyId ?? "-"}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Timeframe: {activeTimeframe}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Last run-step: {healthQuery.data?.last_run_step_at ?? statusQuery.data?.last_run_step_at ?? "-"}
          </p>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <article className="card p-4">
          <h3 className="text-base font-semibold">Data quality</h3>
          {healthQuery.isLoading ? (
            <LoadingState label="Loading health summary" />
          ) : healthQuery.isError ? (
            <ErrorState
              title="Could not load health summary"
              action="Retry to fetch latest guardrail status."
              onRetry={() => void healthQuery.refetch()}
            />
          ) : !latestQuality ? (
            <EmptyState title="No quality report yet" action="Run data quality from quick actions." />
          ) : (
            <div className="mt-3 space-y-2 text-sm">
              <p>
                <span className="text-muted">Status:</span>{" "}
                <span className={`badge ${badgeTone(String(latestQuality.status))}`}>{latestQuality.status}</span>
              </p>
              <p>
                <span className="text-muted">Coverage:</span> {Number(latestQuality.coverage_pct ?? 0).toFixed(2)}%
              </p>
              <p>
                <span className="text-muted">Checked symbols:</span> {latestQuality.checked_symbols} / {latestQuality.total_symbols}
              </p>
              <p>
                <span className="text-muted">Last bar:</span> {latestQuality.last_bar_ts ?? "-"}
              </p>
            </div>
          )}
        </article>

        <article className="card p-4">
          <h3 className="text-base font-semibold">Quick actions</h3>
          <p className="mt-1 text-xs text-muted">Run quality checks, generate reports, and replay policy deterministically.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => runQualityMutation.mutate()}
              disabled={runQualityMutation.isPending || !activeBundleId}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            >
              {runQualityMutation.isPending ? "Queuing..." : "Run Data Quality"}
            </button>
            <button
              type="button"
              onClick={() => dailyReportMutation.mutate()}
              disabled={dailyReportMutation.isPending}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            >
              {dailyReportMutation.isPending ? "Queuing..." : "Generate Daily Report"}
            </button>
            <button
              type="button"
              onClick={() => replayMutation.mutate()}
              disabled={replayMutation.isPending || !activeBundleId || !activePolicyId}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            >
              {replayMutation.isPending ? "Queuing..." : "Run Replay"}
            </button>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            <p className="rounded-lg border border-border px-2 py-2">INFO: {eventCounts.INFO ?? 0}</p>
            <p className="rounded-lg border border-border px-2 py-2">WARN: {eventCounts.WARN ?? 0}</p>
            <p className="rounded-lg border border-border px-2 py-2">ERROR: {eventCounts.ERROR ?? 0}</p>
          </div>
        </article>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Recent operate events</h3>
        {eventsQuery.isLoading ? (
          <LoadingState label="Loading operate events" />
        ) : eventsQuery.isError ? (
          <ErrorState
            title="Could not load events"
            action="Retry to fetch recent operate events."
            onRetry={() => void eventsQuery.refetch()}
          />
        ) : events.length === 0 ? (
          <EmptyState title="No events yet" action="Run a paper step or data quality check." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Severity</th>
                  <th className="px-3 py-2">Category</th>
                  <th className="px-3 py-2">Message</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => (
                  <tr key={event.id} className="border-t border-border">
                    <td className="px-3 py-2">{event.ts}</td>
                    <td className="px-3 py-2">
                      <span className={`badge ${badgeTone(event.severity)}`}>{event.severity}</span>
                    </td>
                    <td className="px-3 py-2">{event.category}</td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        className="focus-ring rounded-md border border-border px-2 py-1 text-left"
                        onClick={() => setSelectedEvent(event)}
                      >
                        {event.message}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <DetailsDrawer
        open={Boolean(selectedEvent)}
        onClose={() => setSelectedEvent(null)}
        title={`Operate Event ${selectedEvent?.id ?? ""}`}
      >
        {selectedEvent ? (
          <div className="space-y-2 text-sm">
            <p><span className="text-muted">Timestamp:</span> {selectedEvent.ts}</p>
            <p><span className="text-muted">Severity:</span> {selectedEvent.severity}</p>
            <p><span className="text-muted">Category:</span> {selectedEvent.category}</p>
            <p><span className="text-muted">Correlation:</span> {selectedEvent.correlation_id ?? "-"}</p>
            <p><span className="text-muted">Message:</span> {selectedEvent.message}</p>
            <pre className="max-h-[280px] overflow-auto rounded-xl border border-border bg-surface p-3 text-xs text-muted">
{JSON.stringify(selectedEvent.details_json ?? {}, null, 2)}
            </pre>
          </div>
        ) : null}
      </DetailsDrawer>

      {(statusQuery.isError || healthQuery.isError) && (
        <ErrorState
          title="Ops data unavailable"
          action="Confirm API and retry."
          onRetry={() => {
            void statusQuery.refetch();
            void healthQuery.refetch();
            void eventsQuery.refetch();
          }}
        />
      )}
    </div>
  );
}
