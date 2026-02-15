"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import type { ApiPolicyEnsemble } from "@/src/lib/api/types";
import { qk } from "@/src/lib/query/keys";

type MemberDraft = {
  policy_id: number;
  weight: number;
  enabled: boolean;
};

function normalizeAllocation(rows: MemberDraft[]): Array<{ policy_id: number; weight: number }> {
  const active = rows
    .filter((row) => row.enabled && row.weight > 0)
    .map((row) => ({ policy_id: row.policy_id, weight: row.weight }));
  const total = active.reduce((acc, row) => acc + row.weight, 0);
  if (total <= 0) {
    return [];
  }
  return active.map((row) => ({ ...row, weight: row.weight / total }));
}

export default function EnsemblesPage() {
  const queryClient = useQueryClient();
  const [bundleId, setBundleId] = useState<number | null>(null);
  const [ensembleName, setEnsembleName] = useState("Balanced Ensemble");
  const [selectedEnsemble, setSelectedEnsemble] = useState<ApiPolicyEnsemble | null>(null);
  const [draft, setDraft] = useState<Record<number, MemberDraft>>({});

  const bundlesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });
  const policiesQuery = useQuery({
    queryKey: qk.policies(1, 100),
    queryFn: async () => (await atlasApi.policies(1, 100)).data,
  });
  const operateStatusQuery = useQuery({
    queryKey: qk.operateStatus,
    queryFn: async () => (await atlasApi.operateStatus()).data,
    refetchInterval: 10_000,
  });
  const ensemblesQuery = useQuery({
    queryKey: qk.ensembles(1, 100, bundleId),
    queryFn: async () =>
      (
        await atlasApi.ensembles({
          page: 1,
          page_size: 100,
          bundle_id: bundleId ?? undefined,
        })
      ).data,
  });

  useEffect(() => {
    if (bundleId !== null) {
      return;
    }
    const firstBundle = bundlesQuery.data?.[0];
    if (firstBundle?.id) {
      setBundleId(Number(firstBundle.id));
    }
  }, [bundleId, bundlesQuery.data]);

  useEffect(() => {
    if (!selectedEnsemble) {
      return;
    }
    const nextDraft: Record<number, MemberDraft> = {};
    const current = new Map(
      (selectedEnsemble.members ?? []).map((row) => [
        row.policy_id,
        { policy_id: row.policy_id, weight: Number(row.weight), enabled: Boolean(row.enabled) },
      ]),
    );
    for (const policy of policiesQuery.data ?? []) {
      const row = current.get(policy.id);
      nextDraft[policy.id] = {
        policy_id: policy.id,
        weight: Number(row?.weight ?? 0),
        enabled: Boolean(row?.enabled ?? false),
      };
    }
    setDraft(nextDraft);
  }, [policiesQuery.data, selectedEnsemble]);

  const createEnsembleMutation = useMutation({
    mutationFn: async () => {
      if (!bundleId) {
        throw new Error("Select a bundle first.");
      }
      return (
        await atlasApi.createEnsemble({
          name: ensembleName.trim(),
          bundle_id: bundleId,
          is_active: false,
        })
      ).data;
    },
    onSuccess: () => {
      toast.success("Ensemble created");
      queryClient.invalidateQueries({ queryKey: qk.ensembles(1, 100, bundleId) });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not create ensemble");
    },
  });

  const setActiveMutation = useMutation({
    mutationFn: async (ensembleId: number) => (await atlasApi.setActiveEnsemble(ensembleId)).data,
    onSuccess: () => {
      toast.success("Active ensemble updated");
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.ensembles(1, 100, bundleId) });
      queryClient.invalidateQueries({ queryKey: qk.paperState });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not set active ensemble");
    },
  });

  const saveMembersMutation = useMutation({
    mutationFn: async (ensembleId: number) => {
      const members = Object.values(draft)
        .filter((row) => row.enabled || row.weight > 0)
        .map((row) => ({
          policy_id: row.policy_id,
          weight: Number(row.weight),
          enabled: Boolean(row.enabled),
        }));
      return (await atlasApi.upsertEnsembleMembers(ensembleId, { members })).data;
    },
    onSuccess: async (payload) => {
      toast.success("Ensemble members updated");
      const refreshed = await atlasApi.ensembleById(Number(payload.id));
      setSelectedEnsemble(refreshed.data);
      queryClient.invalidateQueries({ queryKey: qk.ensembles(1, 100, bundleId) });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not update members");
    },
  });

  const activeEnsemble = operateStatusQuery.data?.active_ensemble ?? null;
  const latestRunSummary = (operateStatusQuery.data?.latest_run as Record<string, unknown> | null)
    ?.summary_json as Record<string, unknown> | undefined;
  const latestSelectedCounts =
    (latestRunSummary?.ensemble_selected_counts_by_policy as Record<string, number> | undefined) ?? {};
  const latestRiskBudget =
    (latestRunSummary?.ensemble_risk_budget_by_policy as Record<string, number> | undefined) ?? {};
  const draftRows = useMemo(() => Object.values(draft), [draft]);
  const draftAllocation = useMemo(() => normalizeAllocation(draftRows), [draftRows]);

  return (
    <div className="space-y-5">
      <section className="card p-4">
        <h2 className="text-xl font-semibold">Policy Ensembles</h2>
        <p className="mt-1 text-sm text-muted">
          Blend multiple policies with deterministic weights and global portfolio caps.
        </p>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <label className="text-sm">
            Bundle
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={bundleId ?? ""}
              onChange={(event) =>
                setBundleId(event.target.value ? Number(event.target.value) : null)
              }
            >
              {(bundlesQuery.data ?? []).map((bundle) => (
                <option key={bundle.id} value={bundle.id}>
                  {bundle.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm md:col-span-2">
            Ensemble name
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={ensembleName}
              onChange={(event) => setEnsembleName(event.target.value)}
              placeholder="Balanced Ensemble"
            />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => createEnsembleMutation.mutate()}
            disabled={createEnsembleMutation.isPending || !bundleId || !ensembleName.trim()}
            className="focus-ring rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
          >
            {createEnsembleMutation.isPending ? "Creating..." : "Create Ensemble"}
          </button>
          <p className="rounded-xl border border-border px-3 py-2 text-xs text-muted">
            Active ensemble: {activeEnsemble?.name ?? "-"}
          </p>
        </div>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Last Run Allocation</h3>
        {!activeEnsemble ? (
          <EmptyState title="No active ensemble" action="Set one ensemble active to use blended policy mode." />
        ) : (
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <span className="text-muted">Ensemble:</span> {activeEnsemble.name} (#{activeEnsemble.id})
            </p>
            <div className="overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-3 py-2">Policy</th>
                    <th className="px-3 py-2">Selected</th>
                    <th className="px-3 py-2">Risk Budget</th>
                  </tr>
                </thead>
                <tbody>
                  {(activeEnsemble.members ?? []).length === 0 ? (
                    <tr className="border-t border-border">
                      <td className="px-3 py-3 text-xs text-muted" colSpan={3}>
                        No members configured yet.
                      </td>
                    </tr>
                  ) : (
                    (activeEnsemble.members ?? []).map((member) => (
                      <tr key={`${member.ensemble_id}-${member.policy_id}`} className="border-t border-border">
                        <td className="px-3 py-2">
                          {member.policy_name ?? `Policy ${member.policy_id}`} (#{member.policy_id})
                        </td>
                        <td className="px-3 py-2">{Number(latestSelectedCounts[String(member.policy_id)] ?? 0)}</td>
                        <td className="px-3 py-2">
                          {Number(latestRiskBudget[String(member.policy_id)] ?? 0).toFixed(6)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Ensemble Library</h3>
        {ensemblesQuery.isLoading ? (
          <LoadingState label="Loading ensembles" />
        ) : ensemblesQuery.isError ? (
          <ErrorState
            title="Could not load ensembles"
            action="Retry to fetch available ensembles."
            onRetry={() => void ensemblesQuery.refetch()}
          />
        ) : (ensemblesQuery.data ?? []).length === 0 ? (
          <EmptyState title="No ensembles yet" action="Create one from the setup section above." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2">Bundle</th>
                  <th className="px-3 py-2">Members</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(ensemblesQuery.data ?? []).map((ensemble) => (
                  <tr key={ensemble.id} className="border-t border-border">
                    <td className="px-3 py-2">{ensemble.name}</td>
                    <td className="px-3 py-2">#{ensemble.bundle_id}</td>
                    <td className="px-3 py-2">{ensemble.members.length}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`badge ${
                          ensemble.is_active ? "bg-success/15 text-success" : "bg-surface text-muted"
                        }`}
                      >
                        {ensemble.is_active ? "Active" : "Inactive"}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          onClick={() => setSelectedEnsemble(ensemble)}
                        >
                          Manage
                        </button>
                        <button
                          type="button"
                          className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          onClick={() => setActiveMutation.mutate(ensemble.id)}
                          disabled={setActiveMutation.isPending}
                        >
                          Set Active
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <DetailsDrawer
        open={Boolean(selectedEnsemble)}
        onClose={() => setSelectedEnsemble(null)}
        title={`Ensemble ${selectedEnsemble?.name ?? ""}`}
      >
        {!selectedEnsemble ? null : (
          <div className="space-y-3 text-sm">
            <p className="text-muted">Configure enabled policies and relative weights.</p>
            {(policiesQuery.data ?? []).length === 0 ? (
              <EmptyState title="No policies yet" action="Create policies in Auto Research first." />
            ) : (
              <div className="overflow-hidden rounded-xl border border-border">
                <table className="w-full text-sm">
                  <thead className="bg-surface text-left text-muted">
                    <tr>
                      <th className="px-3 py-2">Policy</th>
                      <th className="px-3 py-2">Enabled</th>
                      <th className="px-3 py-2">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(policiesQuery.data ?? []).map((policy) => {
                      const row = draft[policy.id] ?? {
                        policy_id: policy.id,
                        weight: 0,
                        enabled: false,
                      };
                      return (
                        <tr key={policy.id} className="border-t border-border">
                          <td className="px-3 py-2">{policy.name}</td>
                          <td className="px-3 py-2">
                            <input
                              type="checkbox"
                              checked={row.enabled}
                              onChange={(event) =>
                                setDraft((prev) => ({
                                  ...prev,
                                  [policy.id]: {
                                    ...row,
                                    enabled: event.target.checked,
                                  },
                                }))
                              }
                            />
                          </td>
                          <td className="px-3 py-2">
                            <input
                              type="number"
                              min={0}
                              step={0.01}
                              value={row.weight}
                              onChange={(event) =>
                                setDraft((prev) => ({
                                  ...prev,
                                  [policy.id]: {
                                    ...row,
                                    weight: Number(event.target.value || 0),
                                  },
                                }))
                              }
                              className="focus-ring w-28 rounded-md border border-border px-2 py-1 text-xs"
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <div className="rounded-xl border border-border p-3 text-xs text-muted">
              <p className="mb-2 font-semibold">Expected Allocation</p>
              {draftAllocation.length === 0 ? (
                <p>No enabled positive-weight members.</p>
              ) : (
                <ul className="space-y-1">
                  {draftAllocation.map((row) => {
                    const policy = (policiesQuery.data ?? []).find((item) => item.id === row.policy_id);
                    return (
                      <li key={row.policy_id}>
                        {policy?.name ?? `Policy ${row.policy_id}`}: {(row.weight * 100).toFixed(1)}%
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
                onClick={() => saveMembersMutation.mutate(selectedEnsemble.id)}
                disabled={saveMembersMutation.isPending}
              >
                {saveMembersMutation.isPending ? "Saving..." : "Save Members"}
              </button>
              <button
                type="button"
                className="focus-ring rounded-xl bg-accent px-3 py-2 text-sm font-semibold text-white"
                onClick={() => setActiveMutation.mutate(selectedEnsemble.id)}
                disabled={setActiveMutation.isPending}
              >
                Set Active
              </button>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}
