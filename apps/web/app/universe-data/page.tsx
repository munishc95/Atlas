"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import { ApiClientError } from "@/src/lib/api/client";
import { qk } from "@/src/lib/query/keys";

export default function UniverseDataPage() {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState("NIFTY500");
  const [timeframe, setTimeframe] = useState("1d");
  const [instrumentKind, setInstrumentKind] = useState("EQUITY_CASH");
  const [underlying, setUnderlying] = useState("");
  const [lotSize, setLotSize] = useState("1");
  const [bundleId, setBundleId] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJobKind, setActiveJobKind] = useState<"import" | "updates" | null>(null);
  const [showCoverageDrawer, setShowCoverageDrawer] = useState(false);

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
  const coverageTimeframe = timeframe === "4h_ish_resampled" ? "4h_ish" : timeframe;
  const dataUpdatesLatestQuery = useQuery({
    queryKey: qk.dataUpdatesLatest(bundleId, coverageTimeframe),
    enabled: bundleId !== null,
    queryFn: async () => {
      if (!bundleId) {
        return null;
      }
      try {
        return (await atlasApi.dataUpdatesLatest(bundleId, coverageTimeframe)).data;
      } catch (error) {
        if (error instanceof ApiClientError && error.status === 404) {
          return null;
        }
        throw error;
      }
    },
    refetchInterval: 10_000,
  });
  const dataCoverageQuery = useQuery({
    queryKey: qk.dataCoverage(bundleId, coverageTimeframe, 100),
    enabled: bundleId !== null,
    queryFn: async () => {
      if (!bundleId) {
        return null;
      }
      return (await atlasApi.dataCoverage(bundleId, coverageTimeframe, 100)).data;
    },
    refetchInterval: 15_000,
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
      formData.append("instrument_kind", instrumentKind);
      if (underlying.trim()) {
        formData.append("underlying", underlying.trim().toUpperCase());
      }
      if (instrumentKind !== "EQUITY_CASH") {
        formData.append("lot_size", lotSize);
      }
      if (bundleId) {
        formData.append("bundle_id", String(bundleId));
      }
      formData.append("file", file);
      return (await atlasApi.importData(formData)).data;
    },
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      setActiveJobKind("import");
      toast.success("Import job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue import");
    },
  });
  const updatesMutation = useMutation({
    mutationFn: async () => {
      if (!bundleId) {
        throw new Error("Choose a bundle before running updates.");
      }
      return (
        await atlasApi.runDataUpdates({
          bundle_id: bundleId,
          timeframe: coverageTimeframe,
        })
      ).data;
    },
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      setActiveJobKind("updates");
      toast.success("Data update job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue data update job");
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
      if (bundleId) {
        queryClient.invalidateQueries({ queryKey: qk.dataUpdatesLatest(bundleId, coverageTimeframe) });
        queryClient.invalidateQueries({ queryKey: qk.dataCoverage(bundleId, coverageTimeframe, 100) });
      }
      toast.success(activeJobKind === "updates" ? "Data updates completed" : "Import completed");
    }
    if (stream.status === "FAILED") {
      toast.error(activeJobKind === "updates" ? "Data updates failed" : "Import failed");
    }
  }, [activeJobId, activeJobKind, bundleId, coverageTimeframe, queryClient, stream.isTerminal, stream.status]);

  const symbols = universeQuery.data?.symbols ?? [];
  const pagedRows = useMemo(
    () => (dataStatusQuery.data ?? []).slice(0, 50),
    [dataStatusQuery.data],
  );

  return (
    <div className="space-y-5">
      <JobDrawer
        jobId={activeJobId}
        onClose={() => {
          setActiveJobId(null);
          setActiveJobKind(null);
        }}
        title={activeJobKind === "updates" ? "Data Updates Job" : "Import Job"}
      />

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
                  <th className="px-3 py-2 font-medium">Instrument</th>
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
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-xs ${
                          String(row.instrument_kind ?? "EQUITY_CASH").includes("FUT")
                            ? "bg-warning/15 text-warning"
                            : "bg-accent/10 text-accent"
                        }`}
                      >
                        {String(row.instrument_kind ?? "EQUITY_CASH")}
                      </span>
                    </td>
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
              value={instrumentKind}
              onChange={(event) => setInstrumentKind(event.target.value)}
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
            >
              <option value="EQUITY_CASH">EQUITY_CASH</option>
              <option value="STOCK_FUT">STOCK_FUT</option>
              <option value="INDEX_FUT">INDEX_FUT</option>
            </select>
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
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
              value={underlying}
              onChange={(event) => setUnderlying(event.target.value.toUpperCase())}
              placeholder="Underlying (optional)"
            />
            <input
              className="focus-ring rounded-xl border border-border bg-panel px-3 py-2"
              value={lotSize}
              onChange={(event) => setLotSize(event.target.value)}
              placeholder="Lot size"
              disabled={instrumentKind === "EQUITY_CASH"}
            />
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

        <article className="card p-4 space-y-4">
          <div>
            <h3 className="text-base font-semibold">Universe coverage</h3>
            <p className="mt-1 text-xs text-muted">
              Drop updates into <code className="rounded bg-surface px-1 py-0.5">data/inbox/&lt;bundle&gt;/&lt;timeframe&gt;/</code>
            </p>
          </div>
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
            <div className="space-y-2 rounded-xl border border-border px-3 py-2 text-sm">
              <p>
                Loaded symbols: <span className="font-semibold">{symbols.length}</span>
              </p>
              <p>
                Coverage:{" "}
                <span className="font-semibold">
                  {Number(dataCoverageQuery.data?.coverage_pct ?? 0).toFixed(2)}%
                </span>
              </p>
              <p>
                Missing latest: <span className="font-semibold">{dataCoverageQuery.data?.missing_symbols.length ?? 0}</span>
              </p>
              <button
                type="button"
                className="focus-ring rounded-lg border border-border px-2 py-1 text-xs text-muted"
                onClick={() => setShowCoverageDrawer(true)}
              >
                View missing/stale symbols
              </button>
            </div>
          )}
          <div className="rounded-xl border border-border px-3 py-2 text-xs text-muted">
            <p>
              Last update run:{" "}
              <span className="font-semibold text-foreground">
                {dataUpdatesLatestQuery.data?.status ?? "Not run yet"}
              </span>
            </p>
            <p>Rows ingested: {dataUpdatesLatestQuery.data?.rows_ingested ?? 0}</p>
            <p>Processed files: {dataUpdatesLatestQuery.data?.processed_files ?? 0}</p>
            <button
              type="button"
              onClick={() => updatesMutation.mutate()}
              disabled={updatesMutation.isPending || !bundleId}
              className="focus-ring mt-2 rounded-lg border border-border px-2 py-1 text-xs text-muted disabled:opacity-60"
            >
              {updatesMutation.isPending ? "Queuing..." : "Run Data Updates"}
            </button>
          </div>
        </article>
      </section>

      <DetailsDrawer
        open={showCoverageDrawer}
        onClose={() => setShowCoverageDrawer(false)}
        title="Coverage Details"
      >
        {dataCoverageQuery.isLoading ? (
          <LoadingState label="Loading coverage details" />
        ) : dataCoverageQuery.isError ? (
          <ErrorState
            title="Could not load coverage details"
            action="Retry coverage query."
            onRetry={() => void dataCoverageQuery.refetch()}
          />
        ) : !dataCoverageQuery.data ? (
          <EmptyState title="No coverage data" action="Run data updates or quality checks." />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              Expected latest trading day:{" "}
              <span className="font-semibold">{dataCoverageQuery.data.expected_latest_trading_day}</span>
            </p>
            <p>
              Inactive symbols: <span className="font-semibold">{dataCoverageQuery.data.inactive_symbols.length}</span>
            </p>
            <div>
              <p className="mb-1 font-medium">Missing symbols</p>
              <div className="max-h-40 overflow-auto rounded-lg border border-border p-2 text-xs text-muted">
                {(dataCoverageQuery.data.missing_symbols ?? []).slice(0, 120).join(", ") || "None"}
              </div>
            </div>
            <div>
              <p className="mb-1 font-medium">Top stale symbols</p>
              <div className="max-h-48 overflow-auto rounded-lg border border-border p-2 text-xs text-muted">
                {(dataCoverageQuery.data.stale_symbols ?? []).slice(0, 40).map((row) => (
                  <p key={String(row.symbol)}>
                    {String(row.symbol)} - {Number(row.missing_trading_days)} missing day(s)
                  </p>
                ))}
              </div>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}
