"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useJobStream } from "@/src/hooks/useJobStream";
import { atlasApi } from "@/src/lib/api/endpoints";
import type {
  ApiAutoEvalRun,
  ApiOperateEvent,
  ApiOperateRunSummary,
  ApiPolicySwitchEvent,
} from "@/src/lib/api/types";
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
  const [selectedSwitch, setSelectedSwitch] = useState<ApiPolicySwitchEvent | null>(null);
  const [lastOperateSummary, setLastOperateSummary] = useState<ApiOperateRunSummary | null>(null);

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
  const activeBundleId =
    (statusQuery.data?.active_bundle_id as number | null | undefined) ??
    (healthQuery.data?.active_bundle_id as number | null | undefined) ??
    null;
  const activePolicyId = (statusQuery.data?.active_policy_id as number | null | undefined) ?? null;
  const activeTimeframe = String(healthQuery.data?.active_timeframe ?? "1d");
  const eventsQuery = useQuery({
    queryKey: qk.operateEvents(null, null, 20),
    queryFn: async () => (await atlasApi.operateEvents({ limit: 20 })).data,
    refetchInterval: 8_000,
  });
  const autoEvalHistoryQuery = useQuery({
    queryKey: qk.operateAutoEvalHistory(1, 10, activeBundleId, activePolicyId),
    queryFn: async () =>
      (
        await atlasApi.operateAutoEvalHistory(1, 10, {
          bundle_id: activeBundleId ?? undefined,
          policy_id: activePolicyId ?? undefined,
        })
      ).data,
    refetchInterval: 8_000,
  });
  const switchHistoryQuery = useQuery({
    queryKey: qk.operatePolicySwitches(10),
    queryFn: async () => (await atlasApi.operatePolicySwitches(10)).data,
    refetchInterval: 8_000,
  });

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
  const operateRunMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.operateRun({
          bundle_id: activeBundleId ?? undefined,
          timeframe: activeTimeframe,
          policy_id: activePolicyId ?? undefined,
          date: new Date().toISOString().slice(0, 10),
        })
      ).data,
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      setLastOperateSummary(null);
      toast.success("Operate run queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue operate run");
    },
  });
  const autoEvalMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.operateAutoEvalRun({
          bundle_id: activeBundleId ?? undefined,
          active_policy_id: activePolicyId ?? undefined,
          timeframe: activeTimeframe,
          asof_date: new Date().toISOString().slice(0, 10),
          seed: 7,
        })
      ).data,
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Auto-evaluation queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue auto-evaluation");
    },
  });
  const setActivePolicyMutation = useMutation({
    mutationFn: async (policyId: number) => (await atlasApi.setActivePolicy(policyId)).data,
    onSuccess: () => {
      toast.success("Active policy switched");
      queryClient.invalidateQueries({ queryKey: qk.settings });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.paperState });
      queryClient.invalidateQueries({ queryKey: qk.operatePolicySwitches(10) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not switch active policy");
    },
  });
  const runUpdatesMutation = useMutation({
    mutationFn: async () => {
      if (!activeBundleId) {
        throw new Error("Active bundle is required before running data updates.");
      }
      return (
        await atlasApi.runDataUpdates({
          bundle_id: activeBundleId,
          timeframe: activeTimeframe,
        })
      ).data;
    },
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Data update job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue data update job");
    },
  });
  const runProviderUpdatesMutation = useMutation({
    mutationFn: async () => {
      if (!activeBundleId) {
        throw new Error("Active bundle is required before running provider updates.");
      }
      return (
        await atlasApi.runProviderUpdates({
          bundle_id: activeBundleId,
          timeframe: activeTimeframe,
        })
      ).data;
    },
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      toast.success("Provider update job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue provider update job");
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
  const streamSummary = stream.result?.summary as ApiOperateRunSummary | undefined;
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      const maybeSummary = streamSummary;
      if (maybeSummary && typeof maybeSummary === "object") {
        setLastOperateSummary(maybeSummary);
      }
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
      queryClient.invalidateQueries({ queryKey: qk.operateEvents(null, null, 20) });
      queryClient.invalidateQueries({ queryKey: qk.operateAutoEvalHistory(1, 10, activeBundleId, activePolicyId) });
      queryClient.invalidateQueries({ queryKey: qk.operatePolicySwitches(10) });
      if (activeBundleId) {
        queryClient.invalidateQueries({ queryKey: qk.dataQualityLatest(activeBundleId, activeTimeframe) });
        queryClient.invalidateQueries({ queryKey: qk.dataQualityHistory(activeBundleId, activeTimeframe, 7) });
        queryClient.invalidateQueries({ queryKey: ["dataUpdatesLatest"] });
        queryClient.invalidateQueries({ queryKey: qk.providerUpdatesLatest(activeBundleId, activeTimeframe) });
        queryClient.invalidateQueries({ queryKey: ["dataCoverage"] });
      }
      toast.success("Ops action complete");
      return;
    }
    if (stream.status === "FAILED") {
      toast.error("Ops action failed");
    }
  }, [
    activeBundleId,
    activePolicyId,
    activeTimeframe,
    queryClient,
    stream.isTerminal,
    streamSummary,
    stream.status,
  ]);

  const mode = String(healthQuery.data?.mode ?? statusQuery.data?.mode ?? "NORMAL");
  const latestQuality = healthQuery.data?.latest_data_quality ?? statusQuery.data?.latest_data_quality ?? null;
  const latestUpdate = healthQuery.data?.latest_data_update ?? statusQuery.data?.latest_data_update ?? null;
  const latestProviderUpdate =
    healthQuery.data?.latest_provider_update ?? statusQuery.data?.latest_provider_update ?? null;
  const eventCounts = healthQuery.data?.recent_event_counts_24h ?? {};
  const autoRunEnabled = Boolean(healthQuery.data?.auto_run_enabled ?? statusQuery.data?.auto_run_enabled);
  const autoRunTimeIst = String(
    healthQuery.data?.auto_run_time_ist ?? statusQuery.data?.auto_run_time_ist ?? "15:35",
  );
  const autoRunIncludesUpdates = Boolean(
    healthQuery.data?.auto_run_include_data_updates ??
      statusQuery.data?.auto_run_include_data_updates,
  );
  const nextScheduledRun = String(
    healthQuery.data?.next_scheduled_run_ist ?? statusQuery.data?.next_scheduled_run_ist ?? "-",
  );
  const autoEvalEnabled = Boolean(
    healthQuery.data?.auto_eval_enabled ?? statusQuery.data?.auto_eval_enabled,
  );
  const autoEvalFrequency = String(
    healthQuery.data?.auto_eval_frequency ?? statusQuery.data?.auto_eval_frequency ?? "WEEKLY",
  );
  const autoEvalTimeIst = String(
    healthQuery.data?.auto_eval_time_ist ?? statusQuery.data?.auto_eval_time_ist ?? "16:00",
  );
  const nextAutoEvalRun = String(
    healthQuery.data?.next_auto_eval_run_ist ?? statusQuery.data?.next_auto_eval_run_ist ?? "-",
  );
  const paperStateSettings =
    ((statusQuery.data?.paper_state as Record<string, unknown> | undefined)?.settings_json as
      | Record<string, unknown>
      | undefined) ?? {};
  const autoEvalAutoSwitch = Boolean(paperStateSettings.operate_auto_eval_auto_switch ?? false);
  const calendarSegment = String(
    healthQuery.data?.calendar_segment ?? statusQuery.data?.calendar_segment ?? "EQUITIES",
  );
  const tradingDayToday = Boolean(
    healthQuery.data?.calendar_is_trading_day_today ?? statusQuery.data?.calendar_is_trading_day_today,
  );
  const calendarSession =
    healthQuery.data?.calendar_session_today ?? statusQuery.data?.calendar_session_today ?? null;
  const nextTradingDay = String(
    healthQuery.data?.calendar_next_trading_day ?? statusQuery.data?.calendar_next_trading_day ?? "-",
  );
  const fastModeEnabled = Boolean(
    healthQuery.data?.fast_mode_enabled ?? statusQuery.data?.fast_mode_enabled ?? false,
  );
  const lastJobDurations = (healthQuery.data?.last_job_durations ??
    statusQuery.data?.last_job_durations ??
    {}) as Record<string, { duration_seconds?: number; status?: string; ts?: string }>;
  const events = useMemo(() => eventsQuery.data ?? [], [eventsQuery.data]);
  const autoEvalRuns = useMemo(
    () => (autoEvalHistoryQuery.data ?? []) as ApiAutoEvalRun[],
    [autoEvalHistoryQuery.data],
  );
  const latestAutoEval = autoEvalRuns[0] ?? null;
  const switchHistory = useMemo(
    () => (switchHistoryQuery.data ?? []) as ApiPolicySwitchEvent[],
    [switchHistoryQuery.data],
  );
  const recommendedPolicyId =
    latestAutoEval?.recommended_action === "SWITCH" && typeof latestAutoEval.recommended_policy_id === "number"
      ? latestAutoEval.recommended_policy_id
      : null;
  const canApplyRecommendedSwitch =
    !autoEvalAutoSwitch &&
    recommendedPolicyId !== null &&
    Number(recommendedPolicyId) > 0 &&
    Number(activePolicyId ?? 0) !== Number(recommendedPolicyId);

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
        <p className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
          Fast mode:{" "}
          <span className={`badge ${fastModeEnabled ? "bg-warning/15 text-warning" : "bg-success/15 text-success"}`}>
            {fastModeEnabled ? "Enabled" : "Disabled"}
          </span>
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Auto-run: {autoRunEnabled ? "Enabled" : "Disabled"} ({autoRunTimeIst} IST)
            {autoRunEnabled ? ` - updates ${autoRunIncludesUpdates ? "on" : "off"}` : ""}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm lg:col-span-2">
            Next scheduled run: {nextScheduledRun}
          </p>
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Auto-eval: {autoEvalEnabled ? "Enabled" : "Disabled"} ({autoEvalFrequency} @ {autoEvalTimeIst} IST)
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm lg:col-span-2">
            Next evaluation: {nextAutoEvalRun}
          </p>
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Calendar: {calendarSegment}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Today: {tradingDayToday ? "Trading day" : "Holiday/Closed"}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Next trading day: {nextTradingDay}
          </p>
        </div>
        <p className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
          Session:{" "}
          {calendarSession?.open_time && calendarSession?.close_time
            ? `${calendarSession.open_time} - ${calendarSession.close_time} IST`
            : "Closed"}
          {calendarSession?.is_special ? " (Special session)" : ""}
          {calendarSession?.label ? ` - ${calendarSession.label}` : ""}
          {calendarSession?.holiday_name ? ` - ${calendarSession.holiday_name}` : ""}
        </p>
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
              <p>
                <span className="text-muted">Last update run:</span>{" "}
                {latestUpdate ? (
                  <span className={`badge ${badgeTone(String(latestUpdate.status))}`}>
                    {latestUpdate.status}
                  </span>
                ) : (
                  "-"
                )}
              </p>
              {latestUpdate ? (
                <p>
                  <span className="text-muted">Rows ingested:</span> {latestUpdate.rows_ingested}
                </p>
              ) : null}
            </div>
          )}
        </article>

        <article className="card p-4">
          <h3 className="text-base font-semibold">Quick actions</h3>
          <p className="mt-1 text-xs text-muted">Run quality checks, generate reports, and replay policy deterministically.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => operateRunMutation.mutate()}
              disabled={operateRunMutation.isPending}
              className="focus-ring rounded-xl bg-accent px-3 py-2 text-sm font-semibold text-white"
            >
              {operateRunMutation.isPending
                ? "Queuing..."
                : "Run Today (Updates -> Quality -> Step -> Report)"}
            </button>
            <button
              type="button"
              onClick={() => runProviderUpdatesMutation.mutate()}
              disabled={runProviderUpdatesMutation.isPending || !activeBundleId}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            >
              {runProviderUpdatesMutation.isPending ? "Queuing..." : "Run Provider Update"}
            </button>
            <button
              type="button"
              onClick={() => runUpdatesMutation.mutate()}
              disabled={runUpdatesMutation.isPending || !activeBundleId}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            >
              {runUpdatesMutation.isPending ? "Queuing..." : "Run Data Updates"}
            </button>
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
          <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
            {(["data_updates", "data_quality", "paper_step", "daily_report", "auto_eval"] as const).map(
              (jobKind) => {
                const row = lastJobDurations[jobKind] ?? {};
                const duration = Number(row.duration_seconds ?? 0);
                return (
                  <p key={jobKind} className="rounded-lg border border-border px-2 py-2">
                    {jobKind}: {duration > 0 ? `${duration.toFixed(2)}s` : "-"}
                  </p>
                );
              },
            )}
          </div>
          <div className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
            <p>
              Provider update run:{" "}
              <span className={`badge ${badgeTone(String(latestProviderUpdate?.status ?? "N/A"))}`}>
                {latestProviderUpdate?.status ?? "Not run"}
              </span>
            </p>
            <p>API calls: {Number(latestProviderUpdate?.api_calls ?? 0)}</p>
            <p>Duration: {Number(latestProviderUpdate?.duration_seconds ?? 0).toFixed(2)}s</p>
          </div>
        </article>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Learning</h3>
        <p className="mt-1 text-xs text-muted">
          Closed-loop policy evaluation with cooldown and switch-rate safety gates.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => autoEvalMutation.mutate()}
            disabled={autoEvalMutation.isPending || !activeBundleId || !activePolicyId}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
          >
            {autoEvalMutation.isPending ? "Queuing..." : "Run Evaluation Now"}
          </button>
          <button
            type="button"
            onClick={() => {
              if (recommendedPolicyId !== null) {
                setActivePolicyMutation.mutate(recommendedPolicyId);
              }
            }}
            disabled={!canApplyRecommendedSwitch || setActivePolicyMutation.isPending}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted disabled:opacity-50"
          >
            {setActivePolicyMutation.isPending ? "Applying..." : "Apply Recommended Switch"}
          </button>
          <p className="rounded-xl border border-border px-3 py-2 text-xs text-muted">
            Auto-switch: {autoEvalAutoSwitch ? "Enabled" : "Manual confirmation required"}
          </p>
        </div>
        {autoEvalHistoryQuery.isLoading ? (
          <div className="mt-3">
            <LoadingState label="Loading auto-evaluation history" />
          </div>
        ) : autoEvalHistoryQuery.isError ? (
          <div className="mt-3">
            <ErrorState
              title="Could not load auto-evaluations"
              action="Retry to fetch the latest recommendation."
              onRetry={() => void autoEvalHistoryQuery.refetch()}
            />
          </div>
        ) : !latestAutoEval ? (
          <div className="mt-3">
            <EmptyState title="No evaluations yet" action="Run evaluation to generate a recommendation." />
          </div>
        ) : (
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <span className="text-muted">Latest recommendation:</span>{" "}
              <span className={`badge ${badgeTone(latestAutoEval.recommended_action)}`}>
                {latestAutoEval.recommended_action}
              </span>
            </p>
            <p>
              <span className="text-muted">Recommended policy:</span>{" "}
              {latestAutoEval.recommended_policy_id ?? "-"}
            </p>
            <p>
              <span className="text-muted">Digest:</span> {latestAutoEval.digest}
            </p>
            <ul className="space-y-1 text-xs text-muted">
              {(latestAutoEval.reasons_json ?? []).slice(0, 4).map((reason, index) => (
                <li key={`auto-eval-reason-${index}`}>{reason}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-4 overflow-hidden rounded-xl border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">From</th>
                <th className="px-3 py-2">To</th>
                <th className="px-3 py-2">Mode</th>
                <th className="px-3 py-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {switchHistory.length === 0 ? (
                <tr className="border-t border-border">
                  <td className="px-3 py-3 text-xs text-muted" colSpan={5}>
                    No policy switches recorded yet.
                  </td>
                </tr>
              ) : (
                switchHistory.map((row) => (
                  <tr key={row.id} className="border-t border-border">
                    <td className="px-3 py-2">{row.ts}</td>
                    <td className="px-3 py-2">{row.from_policy_id}</td>
                    <td className="px-3 py-2">{row.to_policy_id}</td>
                    <td className="px-3 py-2">{row.mode}</td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                        onClick={() => setSelectedSwitch(row)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Latest operate run</h3>
        {!lastOperateSummary ? (
          <EmptyState title="No operate run summary yet" action="Run the one-button operate flow above." />
        ) : (
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <span className="text-muted">Mode:</span>{" "}
              <span className={`badge ${badgeTone(String(lastOperateSummary.mode ?? "NORMAL"))}`}>
                {String(lastOperateSummary.mode ?? "NORMAL")}
              </span>
            </p>
            <p>
              <span className="text-muted">Data updates:</span> {String(lastOperateSummary.update_status ?? "-")}
            </p>
            <p>
              <span className="text-muted">Data quality:</span> {String(lastOperateSummary.quality_status ?? "-")}
            </p>
            <p>
              <span className="text-muted">Paper:</span>{" "}
              {String((lastOperateSummary.paper?.status as string | undefined) ?? "-")} / selected{" "}
              {Number(lastOperateSummary.paper?.selected_signals_count ?? 0)}
            </p>
            <p>
              <span className="text-muted">Report:</span>{" "}
              {lastOperateSummary.daily_report?.id ? (
                <>
                  #{lastOperateSummary.daily_report.id}{" "}
                  <a
                    href={atlasApi.dailyReportExportPdfUrl(Number(lastOperateSummary.daily_report.id))}
                    target="_blank"
                    rel="noreferrer"
                    className="text-accent underline underline-offset-2"
                  >
                    Download PDF
                  </a>
                </>
              ) : (
                "-"
              )}
            </p>
          </div>
        )}
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

      <DetailsDrawer
        open={Boolean(selectedSwitch)}
        onClose={() => setSelectedSwitch(null)}
        title={`Policy Switch ${selectedSwitch?.id ?? ""}`}
      >
        {selectedSwitch ? (
          <div className="space-y-2 text-sm">
            <p><span className="text-muted">Timestamp:</span> {selectedSwitch.ts}</p>
            <p><span className="text-muted">From policy:</span> {selectedSwitch.from_policy_id}</p>
            <p><span className="text-muted">To policy:</span> {selectedSwitch.to_policy_id}</p>
            <p><span className="text-muted">Mode:</span> {selectedSwitch.mode}</p>
            <p><span className="text-muted">Reason:</span> {selectedSwitch.reason}</p>
            <p><span className="text-muted">Auto-eval:</span> {selectedSwitch.auto_eval_id ?? "-"}</p>
            <pre className="max-h-[280px] overflow-auto rounded-xl border border-border bg-surface p-3 text-xs text-muted">
{JSON.stringify(selectedSwitch.cooldown_state_json ?? {}, null, 2)}
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
            void autoEvalHistoryQuery.refetch();
            void switchHistoryQuery.refetch();
          }}
        />
      )}
    </div>
  );
}
