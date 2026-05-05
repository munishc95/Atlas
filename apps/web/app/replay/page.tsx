"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useJobStream } from "@/src/hooks/useJobStream";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

function todayIso(): string {
  const now = new Date();
  return now.toISOString().slice(0, 10);
}

function priorIso(days: number): string {
  const now = new Date();
  now.setDate(now.getDate() - days);
  return now.toISOString().slice(0, 10);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
}

function pct(value: unknown): string {
  const number = typeof value === "number" ? value : Number(value ?? 0);
  return `${(number * 100).toFixed(1)}%`;
}

function pctPoints(value: unknown): string {
  const number = typeof value === "number" ? value : Number(value ?? 0);
  return `${number.toFixed(2)}%`;
}

export default function ReplayPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedReplayId, setSelectedReplayId] = useState<number | null>(null);
  const [bundleId, setBundleId] = useState("");
  const [policyId, setPolicyId] = useState("");
  const [startDate, setStartDate] = useState(priorIso(90));
  const [endDate, setEndDate] = useState(todayIso());
  const [seed, setSeed] = useState("7");
  const [includeSignalAudit, setIncludeSignalAudit] = useState(true);
  const [auditMinUpsidePct, setAuditMinUpsidePct] = useState("2");
  const [auditMaxSignals, setAuditMaxSignals] = useState("25");

  const universesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });
  const policiesQuery = useQuery({
    queryKey: qk.policies(1, 200),
    queryFn: async () => (await atlasApi.policies(1, 200)).data,
  });
  const runsQuery = useQuery({
    queryKey: qk.replayRuns(1, 30),
    queryFn: async () => (await atlasApi.replayRuns(1, 30)).data,
    refetchInterval: 10_000,
  });
  const replayDetailQuery = useQuery({
    queryKey: qk.replayRun(selectedReplayId),
    queryFn: async () => (await atlasApi.replayRunById(selectedReplayId as number)).data,
    enabled: selectedReplayId !== null,
  });

  useEffect(() => {
    const bundles = universesQuery.data ?? [];
    const first = bundles[0];
    if (!bundleId && first?.id) {
      setBundleId(String(first.id));
    }
  }, [bundleId, universesQuery.data]);

  useEffect(() => {
    const policies = policiesQuery.data ?? [];
    const first = policies[0];
    if (!policyId && first?.id) {
      setPolicyId(String(first.id));
    }
  }, [policyId, policiesQuery.data]);

  const runMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runReplay({
          bundle_id: Number(bundleId),
          policy_id: Number(policyId),
          start_date: startDate,
          end_date: endDate,
          seed: Number(seed) || 7,
          include_signal_audit: includeSignalAudit,
          audit_offsets_days: [7, 14, 30],
          audit_min_upside_pct: Number(auditMinUpsidePct) || 2,
          audit_max_signals_per_checkpoint: Number(auditMaxSignals) || 25,
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast.success("Replay queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue replay");
    },
  });

  const stream = useJobStream(activeJobId);
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      toast.success("Replay completed");
      queryClient.invalidateQueries({ queryKey: qk.replayRuns(1, 30) });
    } else if (stream.status === "FAILED") {
      toast.error("Replay failed");
    }
  }, [queryClient, stream.isTerminal, stream.status]);

  const selectedRun =
    selectedReplayId === null
      ? null
      : (runsQuery.data ?? []).find((row) => row.id === selectedReplayId) ?? replayDetailQuery.data ?? null;

  const selectedFinalMetrics = useMemo(() => {
    const summary = (selectedRun?.summary_json ?? {}) as Record<string, unknown>;
    const final = (summary.final ?? {}) as Record<string, unknown>;
    return (final.metrics ?? {}) as Record<string, unknown>;
  }, [selectedRun?.summary_json]);
  const selectedSignalAudit = useMemo(() => {
    const summary = asRecord(selectedRun?.summary_json);
    return asRecord(summary.signal_audit);
  }, [selectedRun?.summary_json]);
  const selectedAuditSummary = useMemo(
    () => asRecord(selectedSignalAudit.summary),
    [selectedSignalAudit],
  );
  const selectedAuditCheckpoints = useMemo(
    () => asArray(selectedSignalAudit.checkpoints),
    [selectedSignalAudit],
  );

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Replay Job Progress" />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Replay</h2>
        <p className="mt-1 text-sm text-muted">
          Deterministic policy replay for reproducibility and audit-ready exports.
        </p>
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <label className="text-xs text-muted">
            Bundle
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={bundleId}
              onChange={(event) => setBundleId(event.target.value)}
            >
              {(universesQuery.data ?? []).map((bundle) => (
                <option key={bundle.id} value={bundle.id}>
                  {bundle.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted">
            Policy
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={policyId}
              onChange={(event) => setPolicyId(event.target.value)}
            >
              {(policiesQuery.data ?? []).map((policy) => (
                <option key={policy.id} value={policy.id}>
                  {policy.id} - {policy.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted">
            Start date
            <input
              type="date"
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={startDate}
              onChange={(event) => setStartDate(event.target.value)}
            />
          </label>
          <label className="text-xs text-muted">
            End date
            <input
              type="date"
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={endDate}
              onChange={(event) => setEndDate(event.target.value)}
            />
          </label>
          <label className="text-xs text-muted">
            Seed
            <input
              type="number"
              min={1}
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              className="focus-ring h-4 w-4 rounded border-border"
              checked={includeSignalAudit}
              onChange={(event) => setIncludeSignalAudit(event.target.checked)}
            />
            Signal audit
          </label>
          <label className="text-xs text-muted">
            Upside threshold %
            <input
              type="number"
              min={0.1}
              step={0.1}
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={auditMinUpsidePct}
              onChange={(event) => setAuditMinUpsidePct(event.target.value)}
            />
          </label>
          <label className="text-xs text-muted">
            Max audit signals
            <input
              type="number"
              min={1}
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={auditMaxSignals}
              onChange={(event) => setAuditMaxSignals(event.target.value)}
            />
          </label>
        </div>
        <div className="mt-3 flex items-center gap-2">
          <button
            type="button"
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            onClick={() => runMutation.mutate()}
            disabled={runMutation.isPending}
          >
            {runMutation.isPending ? "Queuing..." : "Run Replay"}
          </button>
          {stream.status !== "IDLE" ? (
            <p className="text-xs text-muted">
              Live status: {stream.status} ({stream.progress}%)
            </p>
          ) : null}
        </div>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Replay runs</h3>
        {runsQuery.isLoading ? (
          <LoadingState label="Loading replay runs" />
        ) : runsQuery.isError ? (
          <ErrorState
            title="Could not load replay runs"
            action="Retry after API connectivity is restored."
            onRetry={() => void runsQuery.refetch()}
          />
        ) : (runsQuery.data ?? []).length === 0 ? (
          <EmptyState title="No replay runs yet" action="Run your first deterministic replay." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Created</th>
                  <th className="px-3 py-2">Bundle</th>
                  <th className="px-3 py-2">Policy</th>
                  <th className="px-3 py-2">Window</th>
                  <th className="px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {(runsQuery.data ?? []).map((row) => (
                  <tr key={row.id} className="border-t border-border">
                    <td className="px-3 py-2">{row.created_at}</td>
                    <td className="px-3 py-2">{row.bundle_id}</td>
                    <td className="px-3 py-2">{row.policy_id}</td>
                    <td className="px-3 py-2">
                      {row.start_date} to {row.end_date}
                    </td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                        onClick={() => setSelectedReplayId(row.id)}
                      >
                        View
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
        open={selectedRun !== null}
        onClose={() => setSelectedReplayId(null)}
        title={`Replay #${selectedRun?.id ?? ""}`}
      >
        {!selectedRun ? (
          <EmptyState title="No replay selected" action="Select a replay row to inspect details." />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              <span className="text-muted">Digest:</span>{" "}
              {String((selectedRun.summary_json?.digest as string | undefined) ?? "-")}
            </p>
            <p>
              <span className="text-muted">Engine version:</span>{" "}
              {String((selectedRun.summary_json?.engine_version as string | undefined) ?? "-")}
            </p>
            <p>
              <span className="text-muted">Period return:</span>{" "}
              {String(selectedFinalMetrics.period_return ?? "-")}
            </p>
            <p>
              <span className="text-muted">Max drawdown:</span>{" "}
              {String(selectedFinalMetrics.max_drawdown ?? "-")}
            </p>
            <p>
              <span className="text-muted">Calmar:</span> {String(selectedFinalMetrics.calmar ?? "-")}
            </p>
            {selectedSignalAudit.enabled ? (
              <div className="space-y-3 border-t border-border pt-3">
                <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
                  <div>
                    <p className="text-muted">Eligible</p>
                    <p className="text-base font-semibold">{String(selectedAuditSummary.eligible_count ?? 0)}</p>
                  </div>
                  <div>
                    <p className="text-muted">Upside hit</p>
                    <p className="text-base font-semibold">{pct(selectedAuditSummary.worked_rate)}</p>
                  </div>
                  <div>
                    <p className="text-muted">Positive close</p>
                    <p className="text-base font-semibold">{pct(selectedAuditSummary.positive_close_rate)}</p>
                  </div>
                  <div>
                    <p className="text-muted">Stop hit</p>
                    <p className="text-base font-semibold">{pct(selectedAuditSummary.stop_hit_rate)}</p>
                  </div>
                </div>
                <div className="overflow-hidden rounded-xl border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-surface text-left text-muted">
                      <tr>
                        <th className="px-2 py-1.5">Checkpoint</th>
                        <th className="px-2 py-1.5">Signals</th>
                        <th className="px-2 py-1.5">Upside</th>
                        <th className="px-2 py-1.5">Avg return</th>
                        <th className="px-2 py-1.5">Avg drawdown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedAuditCheckpoints.map((checkpoint) => {
                        const summary = asRecord(checkpoint.summary);
                        return (
                          <tr key={String(checkpoint.asof_date)} className="border-t border-border">
                            <td className="px-2 py-1.5">{String(checkpoint.asof_date ?? "-")}</td>
                            <td className="px-2 py-1.5">{String(summary.eligible_count ?? 0)}</td>
                            <td className="px-2 py-1.5">{pct(summary.worked_rate)}</td>
                            <td className="px-2 py-1.5">{pctPoints(summary.avg_close_return_pct)}</td>
                            <td className="px-2 py-1.5">{pctPoints(summary.avg_max_adverse_pct)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <a
                href={atlasApi.replayExportJsonUrl(selectedRun.id)}
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
              >
                Export JSON
              </a>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}
