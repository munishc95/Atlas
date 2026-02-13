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

function toNumber(value: string): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export default function EvaluationsPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedEvaluationId, setSelectedEvaluationId] = useState<number | null>(null);
  const [bundleId, setBundleId] = useState("");
  const [championPolicyId, setChampionPolicyId] = useState("");
  const [challengerPolicyIdsText, setChallengerPolicyIdsText] = useState("");
  const [windowDays, setWindowDays] = useState("20");
  const [seed, setSeed] = useState("7");

  const policiesQuery = useQuery({
    queryKey: qk.policies(1, 200),
    queryFn: async () => (await atlasApi.policies(1, 200)).data,
  });
  const universesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });
  const evaluationsQuery = useQuery({
    queryKey: qk.evaluations(1, 25),
    queryFn: async () => (await atlasApi.evaluations(1, 25)).data,
    refetchInterval: 10_000,
  });
  const detailsQuery = useQuery({
    queryKey: qk.evaluationDetails(selectedEvaluationId),
    queryFn: async () => (await atlasApi.evaluationDetails(selectedEvaluationId as number)).data,
    enabled: selectedEvaluationId !== null,
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
    if (!championPolicyId && first?.id) {
      setChampionPolicyId(String(first.id));
    }
  }, [championPolicyId, policiesQuery.data]);

  const runMutation = useMutation({
    mutationFn: async () => {
      const parsedBundle = toNumber(bundleId);
      const parsedChampion = toNumber(championPolicyId);
      if (parsedBundle === null || parsedChampion === null) {
        throw new Error("Bundle and champion policy are required.");
      }
      const challengerIds = challengerPolicyIdsText
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => Number(item))
        .filter((item) => Number.isFinite(item) && item > 0);
      return (
        await atlasApi.runEvaluation({
          bundle_id: parsedBundle,
          champion_policy_id: parsedChampion,
          challenger_policy_ids: challengerIds.length > 0 ? challengerIds : undefined,
          window_days: Number(windowDays) || 20,
          seed: Number(seed) || 7,
        })
      ).data;
    },
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast.success("Evaluation queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue evaluation");
    },
  });

  const setActiveMutation = useMutation({
    mutationFn: async (policyId: number) => (await atlasApi.setActivePolicy(policyId)).data,
    onSuccess: () => {
      toast.success("Active policy updated");
      queryClient.invalidateQueries({ queryKey: qk.settings });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.paperState });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not set active policy");
    },
  });

  const stream = useJobStream(activeJobId);
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      toast.success("Evaluation completed");
      queryClient.invalidateQueries({ queryKey: qk.evaluations(1, 25) });
    } else if (stream.status === "FAILED") {
      toast.error("Evaluation failed");
    }
  }, [queryClient, stream.isTerminal, stream.status]);

  const selectedEvaluation =
    selectedEvaluationId === null
      ? null
      : (evaluationsQuery.data ?? []).find((row) => row.id === selectedEvaluationId) ?? null;
  const selectedSummary = (selectedEvaluation?.summary_json ?? {}) as Record<string, unknown>;
  const selectedDecision = (selectedSummary.decision ?? {}) as Record<string, unknown>;
  const recommendedPolicyId = useMemo(() => {
    const value = Number(selectedDecision.recommended_policy_id);
    return Number.isFinite(value) && value > 0 ? value : null;
  }, [selectedDecision.recommended_policy_id]);

  return (
    <div className="space-y-5">
      <JobDrawer
        jobId={activeJobId}
        onClose={() => setActiveJobId(null)}
        title="Evaluation Job Progress"
      />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Champion-Challenger Evaluations</h2>
        <p className="mt-1 text-sm text-muted">
          Shadow-evaluate challengers against the active champion without mutating live paper state.
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
            Champion policy
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={championPolicyId}
              onChange={(event) => setChampionPolicyId(event.target.value)}
            >
              {(policiesQuery.data ?? []).map((policy) => (
                <option key={policy.id} value={policy.id}>
                  {policy.id} - {policy.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted">
            Challenger IDs (csv)
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={challengerPolicyIdsText}
              onChange={(event) => setChallengerPolicyIdsText(event.target.value)}
              placeholder="2,3,4"
            />
          </label>
          <label className="text-xs text-muted">
            Window days
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              type="number"
              min={5}
              value={windowDays}
              onChange={(event) => setWindowDays(event.target.value)}
            />
          </label>
          <label className="text-xs text-muted">
            Seed
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              type="number"
              min={1}
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
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
            {runMutation.isPending ? "Queuing..." : "Run Evaluation"}
          </button>
          {stream.status !== "IDLE" ? (
            <p className="text-xs text-muted">
              Live status: {stream.status} ({stream.progress}%)
            </p>
          ) : null}
        </div>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Recent evaluations</h3>
        {evaluationsQuery.isLoading ? (
          <LoadingState label="Loading evaluations" />
        ) : evaluationsQuery.isError ? (
          <ErrorState
            title="Could not load evaluations"
            action="Retry after API connectivity is restored."
            onRetry={() => void evaluationsQuery.refetch()}
          />
        ) : (evaluationsQuery.data ?? []).length === 0 ? (
          <EmptyState title="No evaluations yet" action="Run your first champion-challenger comparison." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Created</th>
                  <th className="px-3 py-2">Bundle</th>
                  <th className="px-3 py-2">Champion</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {(evaluationsQuery.data ?? []).map((row) => (
                  <tr key={row.id} className="border-t border-border">
                    <td className="px-3 py-2">{row.created_at}</td>
                    <td className="px-3 py-2">{row.bundle_id}</td>
                    <td className="px-3 py-2">{row.champion_policy_id}</td>
                    <td className="px-3 py-2">{row.status}</td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                        onClick={() => setSelectedEvaluationId(row.id)}
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
        open={selectedEvaluation !== null}
        onClose={() => setSelectedEvaluationId(null)}
        title={`Evaluation #${selectedEvaluation?.id ?? ""}`}
      >
        {!selectedEvaluation ? (
          <EmptyState title="No evaluation selected" action="Select a row to inspect details." />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              <span className="text-muted">Decision:</span>{" "}
              {String(selectedDecision.recommendation ?? "-")}
            </p>
            <p>
              <span className="text-muted">Recommended policy:</span>{" "}
              {String(selectedDecision.recommended_policy_id ?? "-")}
            </p>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Reasons
              </p>
              <ul className="space-y-1 text-xs">
                {((selectedDecision.reasons as string[] | undefined) ?? []).map((reason, index) => (
                  <li key={`reason-${index}`}>{reason}</li>
                ))}
              </ul>
            </div>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Shadow runs
              </p>
              {detailsQuery.isLoading ? (
                <LoadingState label="Loading shadow run details" />
              ) : detailsQuery.isError ? (
                <ErrorState
                  title="Could not load shadow details"
                  action="Retry to fetch candidate breakdown."
                  onRetry={() => void detailsQuery.refetch()}
                />
              ) : (
                <ul className="space-y-1 text-xs">
                  {(detailsQuery.data ?? []).map((item) => {
                    const metrics = (item.run_summary_json?.metrics ?? {}) as Record<string, unknown>;
                    return (
                      <li key={item.id}>
                        Policy {item.policy_id}: return {String(metrics.period_return ?? "-")}, maxDD{" "}
                        {String(metrics.max_drawdown ?? "-")}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
                disabled={recommendedPolicyId === null || setActiveMutation.isPending}
                onClick={() => {
                  if (recommendedPolicyId !== null) {
                    setActiveMutation.mutate(recommendedPolicyId);
                  }
                }}
              >
                {setActiveMutation.isPending ? "Applying..." : "Apply as Active Policy"}
              </button>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}
