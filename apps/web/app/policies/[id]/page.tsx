"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

function asNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export default function PolicyDetailPage() {
  const params = useParams<{ id: string }>();
  const policyId = asNumber(params.id);

  const policyQuery = useQuery({
    queryKey: qk.policy(policyId),
    queryFn: async () => (await atlasApi.policyById(policyId as number)).data,
    enabled: policyId !== null,
  });
  const health20Query = useQuery({
    queryKey: qk.policyHealth(policyId, 20),
    queryFn: async () => (await atlasApi.policyHealth(policyId as number, 20, true)).data,
    enabled: policyId !== null,
  });
  const health60Query = useQuery({
    queryKey: qk.policyHealth(policyId, 60),
    queryFn: async () => (await atlasApi.policyHealth(policyId as number, 60, true)).data,
    enabled: policyId !== null,
  });

  if (policyId === null) {
    return <EmptyState title="Invalid policy id" action="Open this page from a valid policy link." />;
  }

  if (policyQuery.isLoading || health20Query.isLoading || health60Query.isLoading) {
    return <LoadingState label="Loading policy health" />;
  }

  if (policyQuery.isError || health20Query.isError || health60Query.isError) {
    return (
      <ErrorState
        title="Could not load policy health"
        action="Retry after API connectivity is restored."
        onRetry={() => {
          void policyQuery.refetch();
          void health20Query.refetch();
          void health60Query.refetch();
        }}
      />
    );
  }

  const policy = policyQuery.data;
  if (!policy) {
    return <EmptyState title="Policy not found" action="Create a policy from Auto Research first." />;
  }

  const baseline = (policy.definition_json?.baseline ?? {}) as Record<string, unknown>;
  const health20 = health20Query.data;
  const health60 = health60Query.data;

  return (
    <div className="space-y-5">
      <section className="card p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-xl font-semibold">{policy.name}</h2>
            <p className="mt-1 text-sm text-muted">Policy health timeline and drift diagnostics.</p>
          </div>
          <Link
            href="/reports"
            className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
          >
            Open reports
          </Link>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="card p-4 lg:col-span-1">
          <h3 className="text-base font-semibold">Baseline</h3>
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <span className="text-muted">OOS score:</span> {String(baseline.oos_score ?? "-")}
            </p>
            <p>
              <span className="text-muted">MaxDD:</span> {String(baseline.max_drawdown ?? "-")}
            </p>
            <p>
              <span className="text-muted">Win rate:</span> {String(baseline.win_rate ?? "-")}
            </p>
            <p>
              <span className="text-muted">Period return:</span> {String(baseline.period_return ?? "-")}
            </p>
          </div>
        </article>

        <article className="card p-4 lg:col-span-2">
          <h3 className="text-base font-semibold">Health snapshots</h3>
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Window</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Return</th>
                  <th className="px-3 py-2">MaxDD</th>
                  <th className="px-3 py-2">Cost ratio</th>
                </tr>
              </thead>
              <tbody>
                {[health20, health60].map((snapshot) => {
                  if (!snapshot) {
                    return null;
                  }
                  return (
                    <tr key={`${snapshot.window_days}-${snapshot.id}`} className="border-t border-border">
                      <td className="px-3 py-2">{snapshot.window_days}d</td>
                      <td className="px-3 py-2">{snapshot.status}</td>
                      <td className="px-3 py-2">
                        {String(snapshot.metrics_json?.period_return ?? "-")}
                      </td>
                      <td className="px-3 py-2">
                        {String(snapshot.metrics_json?.max_drawdown ?? "-")}
                      </td>
                      <td className="px-3 py-2">{String(snapshot.metrics_json?.cost_ratio ?? "-")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">20d reasons</p>
              {(health20?.reasons_json ?? []).length === 0 ? (
                <p className="text-xs text-muted">No drift reasons.</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {(health20?.reasons_json ?? []).map((reason, index) => (
                    <li key={`r20-${index}`}>{reason}</li>
                  ))}
                </ul>
              )}
            </div>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">60d reasons</p>
              {(health60?.reasons_json ?? []).length === 0 ? (
                <p className="text-xs text-muted">No drift reasons.</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {(health60?.reasons_json ?? []).map((reason, index) => (
                    <li key={`r60-${index}`}>{reason}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </article>
      </section>
    </div>
  );
}
