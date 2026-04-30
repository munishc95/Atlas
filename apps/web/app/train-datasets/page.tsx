"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useJobStream } from "@/src/hooks/useJobStream";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function priorIso(days: number): string {
  const next = new Date();
  next.setDate(next.getDate() - days);
  return next.toISOString().slice(0, 10);
}

export default function TrainDatasetsPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [name, setName] = useState(`dataset-${todayIso()}`);
  const [bundleId, setBundleId] = useState("");
  const [timeframe, setTimeframe] = useState("1d");
  const [startDate, setStartDate] = useState(priorIso(365));
  const [endDate, setEndDate] = useState(todayIso());
  const [adjustmentMode, setAdjustmentMode] = useState<"RAW" | "ADJUSTED">("ADJUSTED");
  const [membershipMode, setMembershipMode] = useState<"CURRENT" | "HISTORICAL">("HISTORICAL");
  const [features, setFeatures] = useState({
    return_1d: true,
    return_5d: true,
    atr_14: true,
    rsi_14: true,
    ema_20: true,
  });
  const [labels, setLabels] = useState({
    future_return_5d: true,
  });

  const bundlesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });
  const datasetsQuery = useQuery({
    queryKey: qk.trainDatasets(100),
    queryFn: async () => (await atlasApi.trainDatasets(100)).data,
    refetchInterval: 10_000,
  });
  const selectedDatasetQuery = useQuery({
    queryKey: qk.trainDataset(selectedDatasetId),
    enabled: selectedDatasetId !== null,
    queryFn: async () => (await atlasApi.trainDatasetById(selectedDatasetId as number)).data,
  });
  const latestRunQuery = useQuery({
    queryKey: qk.trainDatasetLatestRun(selectedDatasetId),
    enabled: selectedDatasetId !== null,
    queryFn: async () => {
      try {
        return (await atlasApi.trainDatasetLatestRun(selectedDatasetId as number)).data;
      } catch {
        return null;
      }
    },
    refetchInterval: 10_000,
  });
  const downloadInfoQuery = useQuery({
    queryKey: qk.trainDatasetDownloadInfo(selectedDatasetId),
    enabled: selectedDatasetId !== null,
    queryFn: async () => {
      try {
        return (await atlasApi.trainDatasetDownloadInfo(selectedDatasetId as number)).data;
      } catch {
        return null;
      }
    },
    refetchInterval: 10_000,
  });

  useEffect(() => {
    const firstBundle = bundlesQuery.data?.[0];
    if (!bundleId && firstBundle?.id) {
      setBundleId(String(firstBundle.id));
    }
  }, [bundleId, bundlesQuery.data]);

  useEffect(() => {
    if (selectedDatasetId !== null) {
      return;
    }
    const firstDataset = datasetsQuery.data?.[0];
    if (firstDataset?.id) {
      setSelectedDatasetId(firstDataset.id);
    }
  }, [datasetsQuery.data, selectedDatasetId]);

  const createMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.createTrainDataset({
          name,
          bundle_id: Number(bundleId),
          timeframe,
          start_date: startDate,
          end_date: endDate,
          adjustment_mode: adjustmentMode,
          membership_mode: membershipMode,
          feature_config_json: features,
          label_config_json: labels,
        })
      ).data,
    onSuccess: (payload) => {
      queryClient.invalidateQueries({ queryKey: qk.trainDatasets(100) });
      setSelectedDatasetId(payload.id);
      toast.success("Train dataset created");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not create train dataset");
    },
  });

  const buildMutation = useMutation({
    mutationFn: async () => {
      if (!selectedDatasetId) {
        throw new Error("Select a train dataset before building.");
      }
      return (await atlasApi.buildTrainDataset(selectedDatasetId, { force: true })).data;
    },
    onSuccess: (payload) => {
      setActiveJobId(payload.job_id);
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Train dataset build queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue train dataset build");
    },
  });

  const stream = useJobStream(activeJobId);
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    queryClient.invalidateQueries({ queryKey: qk.trainDatasets(100) });
    if (selectedDatasetId !== null) {
      queryClient.invalidateQueries({ queryKey: qk.trainDataset(selectedDatasetId) });
      queryClient.invalidateQueries({ queryKey: qk.trainDatasetLatestRun(selectedDatasetId) });
      queryClient.invalidateQueries({ queryKey: qk.trainDatasetDownloadInfo(selectedDatasetId) });
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      toast.success("Train dataset build completed");
    } else if (stream.status === "FAILED") {
      toast.error("Train dataset build failed");
    }
  }, [queryClient, selectedDatasetId, stream.isTerminal, stream.status]);

  const selectedDataset = selectedDatasetQuery.data ?? null;
  const latestRun = latestRunQuery.data ?? null;
  const downloadInfo = downloadInfoQuery.data ?? null;
  const featureSummary = useMemo(
    () =>
      Object.entries(features)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key)
        .join(", "),
    [features],
  );

  if (bundlesQuery.isLoading || datasetsQuery.isLoading) {
    return <LoadingState label="Loading train datasets" />;
  }

  if (bundlesQuery.isError || datasetsQuery.isError) {
    return (
      <ErrorState
        title="Could not load train dataset workspace"
        action="Retry after API connectivity is restored."
        onRetry={() => {
          void bundlesQuery.refetch();
          void datasetsQuery.refetch();
        }}
      />
    );
  }

  return (
    <div className="space-y-5">
      <JobDrawer
        jobId={activeJobId}
        onClose={() => setActiveJobId(null)}
        title="Train Dataset Build"
      />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Train Datasets</h2>
        <p className="mt-1 text-sm text-muted">
          Materialize bounded, local parquet datasets for research and ML workflows.
        </p>
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <label className="text-xs text-muted">
            Dataset name
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label className="text-xs text-muted">
            Bundle
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={bundleId}
              onChange={(event) => setBundleId(event.target.value)}
            >
              {(bundlesQuery.data ?? []).map((bundle) => (
                <option key={bundle.id} value={bundle.id}>
                  {bundle.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted">
            Timeframe
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
            >
              <option value="1d">1d</option>
            </select>
          </label>
          <label className="text-xs text-muted">
            Adjustment mode
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={adjustmentMode}
              onChange={(event) => setAdjustmentMode(event.target.value as "RAW" | "ADJUSTED")}
            >
              <option value="RAW">RAW</option>
              <option value="ADJUSTED">ADJUSTED</option>
            </select>
          </label>
          <label className="text-xs text-muted">
            Membership mode
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={membershipMode}
              onChange={(event) => setMembershipMode(event.target.value as "CURRENT" | "HISTORICAL")}
            >
              <option value="CURRENT">CURRENT</option>
              <option value="HISTORICAL">HISTORICAL</option>
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
          <div className="rounded-xl border border-border px-3 py-2 text-sm text-muted">
            <p className="text-xs uppercase tracking-[0.08em]">Features</p>
            <p className="mt-1">{featureSummary || "None selected"}</p>
          </div>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-3 xl:grid-cols-6">
          {Object.entries(features).map(([key, enabled]) => (
            <label
              key={key}
              className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 text-sm"
            >
              <input
                type="checkbox"
                checked={enabled}
                onChange={(event) =>
                  setFeatures((current) => ({
                    ...current,
                    [key]: event.target.checked,
                  }))
                }
              />
              <span>{key}</span>
            </label>
          ))}
          {Object.entries(labels).map(([key, enabled]) => (
            <label
              key={key}
              className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 text-sm"
            >
              <input
                type="checkbox"
                checked={enabled}
                onChange={(event) =>
                  setLabels((current) => ({
                    ...current,
                    [key]: event.target.checked,
                  }))
                }
              />
              <span>{key}</span>
            </label>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            onClick={() => createMutation.mutate()}
            disabled={createMutation.isPending || !bundleId}
          >
            {createMutation.isPending ? "Creating..." : "Create Dataset"}
          </button>
          <button
            type="button"
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            onClick={() => buildMutation.mutate()}
            disabled={buildMutation.isPending || selectedDatasetId === null}
          >
            {buildMutation.isPending ? "Queuing..." : "Build Dataset"}
          </button>
          {stream.status !== "IDLE" ? (
            <p className="text-xs text-muted">
              Build status: {stream.status} ({stream.progress}%)
            </p>
          ) : null}
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1.1fr,0.9fr]">
        <div className="card p-4">
          <h3 className="text-base font-semibold">Registry</h3>
          {(datasetsQuery.data ?? []).length === 0 ? (
            <EmptyState title="No train datasets yet" action="Create a bounded dataset to get started." />
          ) : (
            <div className="mt-3 overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-3 py-2">Name</th>
                    <th className="px-3 py-2">Bundle</th>
                    <th className="px-3 py-2">Mode</th>
                    <th className="px-3 py-2">Rows</th>
                    <th className="px-3 py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {(datasetsQuery.data ?? []).map((row) => (
                    <tr key={row.id} className="border-t border-border">
                      <td className="px-3 py-2">{row.name}</td>
                      <td className="px-3 py-2">{row.bundle_id}</td>
                      <td className="px-3 py-2">
                        {row.adjustment_mode} / {row.membership_mode}
                      </td>
                      <td className="px-3 py-2">{row.row_count}</td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          onClick={() => setSelectedDatasetId(row.id)}
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
        </div>

        <div className="card p-4">
          <h3 className="text-base font-semibold">Latest build</h3>
          {!selectedDataset ? (
            <EmptyState title="No dataset selected" action="Select a dataset to inspect build output." />
          ) : (
            <div className="mt-3 space-y-3 text-sm">
              <div className="rounded-xl border border-border px-3 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-muted">Dataset</p>
                <p className="mt-1 font-medium">{selectedDataset.name}</p>
                <p className="mt-1 text-muted">
                  {selectedDataset.start_date} to {selectedDataset.end_date} -{" "}
                  {selectedDataset.adjustment_mode} - {selectedDataset.membership_mode}
                </p>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-xl border border-border px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.08em] text-muted">Status</p>
                  <p className="mt-1 font-medium">{latestRun?.status ?? selectedDataset.status}</p>
                </div>
                <div className="rounded-xl border border-border px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.08em] text-muted">Rows</p>
                  <p className="mt-1 font-medium">{latestRun?.row_count ?? selectedDataset.row_count}</p>
                </div>
              </div>
              <div className="rounded-xl border border-border px-3 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-muted">Output path</p>
                <p className="mt-1 break-all font-mono text-xs text-muted">
                  {downloadInfo?.latest_run?.output_path ?? latestRun?.output_path ?? "Not built yet"}
                </p>
              </div>
              <div className="rounded-xl border border-border px-3 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-muted">Warnings</p>
                {latestRun?.warnings_json?.length ? (
                  <ul className="mt-2 space-y-2 text-xs text-muted">
                    {latestRun.warnings_json.slice(0, 5).map((warning, index) => (
                      <li key={`${warning.code ?? "warning"}-${index}`}>
                        {String(warning.code ?? "warning")}: {String(warning.message ?? "")}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-1 text-sm text-muted">No warnings recorded.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
