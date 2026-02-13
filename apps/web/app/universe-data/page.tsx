"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import { qk } from "@/src/lib/query/keys";

export default function UniverseDataPage() {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState("NIFTY500");
  const [timeframe, setTimeframe] = useState("1d");
  const [bundleId, setBundleId] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const universeQuery = useQuery({
    queryKey: qk.universe,
    queryFn: async () => (await atlasApi.universe()).data,
  });

  const dataStatusQuery = useQuery({
    queryKey: qk.dataStatus,
    queryFn: async () => (await atlasApi.dataStatus()).data,
    refetchInterval: 10_000,
  });
  const bundlesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });

  useEffect(() => {
    if (bundleId !== null) {
      return;
    }
    const firstBundle = Number((bundlesQuery.data ?? [])[0]?.id);
    if (Number.isFinite(firstBundle) && firstBundle > 0) {
      setBundleId(firstBundle);
    }
  }, [bundleId, bundlesQuery.data]);

  const importMutation = useMutation({
    mutationFn: async () => {
      if (!file) {
        throw new Error("Please choose a CSV or Parquet file");
      }
      const formData = new FormData();
      formData.append("symbol", symbol);
      formData.append("timeframe", timeframe);
      formData.append("provider", "csv");
      if (bundleId) {
        formData.append("bundle_id", String(bundleId));
      }
      formData.append("file", file);
      return (await atlasApi.importData(formData)).data;
    },
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      toast.success("Import job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue import");
    },
  });

  const stream = useJobStream(activeJobId);

  useEffect(() => {
    if (!activeJobId || !stream.isTerminal) {
      return;
    }

    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      queryClient.invalidateQueries({ queryKey: qk.universe });
      queryClient.invalidateQueries({ queryKey: qk.dataStatus });
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Import completed");
    }
    if (stream.status === "FAILED") {
      toast.error("Import failed");
    }
  }, [activeJobId, queryClient, stream.isTerminal, stream.status]);

  const symbols = universeQuery.data?.symbols ?? [];
  const pagedRows = useMemo(
    () => (dataStatusQuery.data ?? []).slice(0, 50),
    [dataStatusQuery.data],
  );

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Import Job" />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Universe & Data</h2>
        <p className="mt-1 text-sm text-muted">
          NIFTY 500 universe management, liquidity filters, and data health.
        </p>

        {dataStatusQuery.isLoading ? (
          <LoadingState label="Loading data status" />
        ) : dataStatusQuery.isError ? (
          <ErrorState
            title="Could not load dataset status"
            action="Check API and retry."
            onRetry={() => void dataStatusQuery.refetch()}
          />
        ) : pagedRows.length === 0 ? (
          <EmptyState
            title="No data imported"
            action="Use the import wizard below."
            cta="Refresh"
            onCta={() => void dataStatusQuery.refetch()}
          />
        ) : (
          <div className="mt-4 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Bundle</th>
                  <th className="px-3 py-2 font-medium">Symbol</th>
                  <th className="px-3 py-2 font-medium">Timeframe</th>
                  <th className="px-3 py-2 font-medium">Start</th>
                  <th className="px-3 py-2 font-medium">End</th>
                  <th className="px-3 py-2 font-medium">Last update</th>
                </tr>
              </thead>
              <tbody>
                {pagedRows.map((row) => (
                  <tr key={String(row.id)} className="border-t border-border">
                    <td className="px-3 py-2">{String(row.bundle_name ?? "-")}</td>
                    <td className="px-3 py-2">{String(row.symbol)}</td>
                    <td className="px-3 py-2">{String(row.timeframe)}</td>
                    <td className="px-3 py-2">{String(row.start_date)}</td>
                    <td className="px-3 py-2">{String(row.end_date)}</td>
                    <td className="px-3 py-2">{String(row.last_updated)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="card p-4 lg:col-span-2">
          <h3 className="text-base font-semibold">Import wizard</h3>
          <p className="mt-1 text-sm text-muted">
            Upload CSV/Parquet and register the dataset for backtests.
          </p>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <input
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              placeholder="Symbol"
            />
            <select
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
            >
              <option value="1d">1d</option>
              <option value="4h_ish">4h_ish</option>
              <option value="4h_ish_resampled">4h_ish_resampled</option>
            </select>
            <select
              value={bundleId ?? ""}
              onChange={(event) =>
                setBundleId(event.target.value ? Number(event.target.value) : null)
              }
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
            >
              {(bundlesQuery.data ?? []).map((bundle) => (
                <option key={bundle.id} value={bundle.id}>
                  {bundle.name}
                </option>
              ))}
            </select>
            <input
              type="file"
              accept=".csv,.parquet"
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                setFile(event.target.files?.[0] ?? null)
              }
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2 md:col-span-2"
            />
            <button
              type="button"
              onClick={() => importMutation.mutate()}
              disabled={importMutation.isPending || !file}
              className="focus-ring rounded-xl bg-accent px-4 py-2 text-white disabled:opacity-60 md:col-span-2"
            >
              {importMutation.isPending ? "Queuing..." : "Import File"}
            </button>
          </div>
        </article>

        <article className="card p-4">
          <h3 className="text-base font-semibold">Universe coverage</h3>
          {universeQuery.isLoading ? (
            <LoadingState label="Loading symbols" />
          ) : universeQuery.isError ? (
            <ErrorState
              title="Could not load universe"
              action="Retry after API reconnect."
              onRetry={() => void universeQuery.refetch()}
            />
          ) : symbols.length === 0 ? (
            <EmptyState title="No symbols" action="Import at least one symbol." />
          ) : (
            <div className="mt-3 rounded-xl border border-border px-3 py-2 text-sm">
              Loaded symbols: <span className="font-semibold">{symbols.length}</span>
            </div>
          )}
        </article>
      </section>
    </div>
  );
}
