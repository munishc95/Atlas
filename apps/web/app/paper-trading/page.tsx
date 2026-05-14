"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import type {
  ApiEffectiveTradingContext,
  ApiEventRiskEvent,
  ApiEventRiskRefresh,
  ApiForwardJournalSummary,
  ApiPaperSignal,
  ApiPaperSignalPreview,
} from "@/src/lib/api/types";
import { qk } from "@/src/lib/query/keys";

function isTypingElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || target.isContentEditable;
}

function qualityBadgeClass(status: string | undefined): string {
  const token = String(status ?? "PASS").toUpperCase();
  if (token === "FAIL") return "bg-danger/15 text-danger";
  if (token === "WARN") return "bg-warning/15 text-warning";
  return "bg-success/15 text-success";
}

function eventRiskBadgeClass(status: string | undefined): string {
  const token = String(status ?? "SKIPPED").toUpperCase();
  if (token === "FAILED" || token === "FAIL" || token === "BLOCK") {
    return "border-danger/30 bg-danger/10 text-danger";
  }
  if (token === "PARTIAL" || token === "WARN" || token === "HIGH") {
    return "border-warning/30 bg-warning/10 text-warning";
  }
  if (token === "SUCCEEDED" || token === "PASS" || token === "INFO" || token === "LOW") {
    return "border-success/30 bg-success/10 text-success";
  }
  return "border-border bg-surface text-muted";
}

function journalStatusClass(status: string | undefined): string {
  const token = String(status ?? "OPEN").toUpperCase();
  if (token === "STOP_HIT") return "bg-danger/15 text-danger";
  if (token === "EXPIRED") return "bg-muted/15 text-muted";
  if (token === "T1_HIT" || token === "T2_HIT") return "bg-success/15 text-success";
  return "bg-accent/10 text-accent";
}

function formatMoney(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `INR ${value.toLocaleString("en-IN", {
    maximumFractionDigits: value >= 100 ? 0 : 2,
    minimumFractionDigits: value >= 100 ? 0 : 2,
  })}`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toLocaleString("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asText(value: unknown): string {
  return typeof value === "string" && value.trim() ? value : "-";
}

function optionalText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatRecordNumber(row: Record<string, unknown>, key: string, digits = 2): string {
  return formatNumber(asNumber(row[key]), digits);
}

function sourceLabel(source: string | undefined): string {
  const token = String(source ?? "").toUpperCase();
  if (token.includes("ANNOUNCEMENTS")) return "NSE announcement";
  if (token.includes("BOARD")) return "Board meeting";
  if (token.includes("CORPORATE_ACTION")) return "Corporate action";
  return source ? source.replaceAll("_", " ") : "Event risk";
}

function eventTypeLabel(eventType: string | undefined): string {
  return String(eventType ?? "EVENT").replaceAll("_", " ").toLowerCase();
}

function eventRiskStatusFromSeverity(severity: string | undefined): "PASS" | "WARN" | "FAIL" {
  const token = String(severity ?? "WARN").toUpperCase();
  if (["BLOCK", "FAIL", "CRITICAL"].includes(token)) return "FAIL";
  if (["WARN", "WARNING", "HIGH"].includes(token)) return "WARN";
  return "PASS";
}

function eventRiskDecisionText(event: ApiEventRiskEvent): string {
  const status = eventRiskStatusFromSeverity(event.severity);
  const verb = status === "FAIL" ? "Blocked" : status === "WARN" ? "Warned" : "Flagged";
  return `${verb}: ${eventTypeLabel(event.event_type)} on ${event.event_date}`;
}

function eventRiskEventsFromSignal(
  signal: ApiPaperSignal | Record<string, unknown>,
): ApiEventRiskEvent[] {
  const metrics = isRecord(signal.quality_metrics) ? signal.quality_metrics : null;
  const rawEvents = metrics?.event_risk_events;
  if (!Array.isArray(rawEvents)) {
    return [];
  }
  return rawEvents.filter(isRecord).map((event) => ({
    event_date: optionalText(event.event_date) ?? "-",
    scope: optionalText(event.scope) ?? "SYMBOL",
    symbol: optionalText(event.symbol),
    event_type: optionalText(event.event_type) ?? "EVENT",
    severity: optionalText(event.severity) ?? "WARN",
    title: optionalText(event.title) ?? "Event risk",
    source: optionalText(event.source) ?? "event_risk",
    blackout_before_days: asNumber(event.blackout_before_days) ?? 0,
    blackout_after_days: asNumber(event.blackout_after_days) ?? 0,
  }));
}

function EventRiskRefreshPanel({
  refresh,
  compact = false,
}: {
  refresh?: ApiEventRiskRefresh | Record<string, unknown> | null;
  compact?: boolean;
}) {
  if (!isRecord(refresh)) {
    return null;
  }
  const status = String(refresh.status ?? "SKIPPED").toUpperCase();
  const reason = optionalText(refresh.reason)?.replaceAll("_", " ");
  const eventCount = asNumber(refresh.event_count);
  const symbolCount = asNumber(refresh.symbols_with_events);
  const sourceErrors = isRecord(refresh.source_errors) ? Object.keys(refresh.source_errors) : [];
  const window =
    optionalText(refresh.start_date) && optionalText(refresh.end_date)
      ? `${optionalText(refresh.start_date)} to ${optionalText(refresh.end_date)}`
      : null;
  return (
    <div
      className={`rounded-xl border px-3 py-2 text-xs ${eventRiskBadgeClass(status)} ${
        compact ? "mt-2" : ""
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold">Event risk {status}</span>
        <span className="text-current/80">
          NSE announcements, board meetings, and corporate actions
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-current/80">
        {eventCount !== null ? <span>{eventCount} active rows</span> : null}
        {symbolCount !== null ? <span>{symbolCount} symbols</span> : null}
        {window ? <span>{window}</span> : null}
        {reason ? <span>{reason}</span> : null}
        {sourceErrors.length > 0 ? <span>{sourceErrors.length} source issue(s)</span> : null}
      </div>
    </div>
  );
}

function EventRiskEventsList({
  events,
  compact = false,
}: {
  events: ApiEventRiskEvent[];
  compact?: boolean;
}) {
  if (events.length === 0) {
    return null;
  }
  return (
    <div className={compact ? "mt-1 space-y-1" : "mt-2 space-y-2"}>
      {events.slice(0, compact ? 2 : 5).map((event, index) => {
        const status = eventRiskStatusFromSeverity(event.severity);
        return (
          <div
            key={`${event.event_date}-${event.event_type}-${index}`}
            className={`rounded-lg border px-2 py-1 ${eventRiskBadgeClass(status)}`}
          >
            <p className="font-medium">{eventRiskDecisionText(event)}</p>
            <p className="text-current/80">
              {sourceLabel(event.source)}: {event.title}
            </p>
          </div>
        );
      })}
    </div>
  );
}

export default function PaperTradingPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const [selectedPositionId, setSelectedPositionId] = useState<number | null>(null);
  const [selectedOrderId, setSelectedOrderId] = useState<number | null>(null);
  const [autopilotEnabled, setAutopilotEnabled] = useState(true);
  const [bundleId, setBundleId] = useState<number | null>(null);
  const [preview, setPreview] = useState<ApiPaperSignalPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [whyOpen, setWhyOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [latestDecision, setLatestDecision] = useState<Record<string, unknown> | null>(null);

  const paperStateQuery = useQuery({
    queryKey: qk.paperState,
    queryFn: async () => (await atlasApi.paperState()).data,
    refetchInterval: 5_000,
  });

  const regimeQuery = useQuery({
    queryKey: qk.regimeCurrent(),
    queryFn: async () => (await atlasApi.regimeCurrent()).data,
    refetchInterval: 15_000,
  });

  const strategiesQuery = useQuery({
    queryKey: qk.strategies,
    queryFn: async () => (await atlasApi.strategies()).data,
  });

  const policiesQuery = useQuery({
    queryKey: qk.policies(1, 50),
    queryFn: async () => (await atlasApi.policies(1, 50)).data,
  });
  const bundlesQuery = useQuery({
    queryKey: qk.universes,
    queryFn: async () => (await atlasApi.universes()).data,
  });
  const operateQuery = useQuery({
    queryKey: qk.operateStatus,
    queryFn: async () => (await atlasApi.operateStatus()).data,
    refetchInterval: 10_000,
  });
  const forwardJournalQuery = useQuery({
    queryKey: qk.forwardJournal(bundleId, "1d", null, 1, 50),
    queryFn: async () =>
      atlasApi.forwardJournal({
        bundle_id: bundleId ?? undefined,
        timeframe: "1d",
        page: 1,
        page_size: 50,
      }),
    enabled: bundleId !== null,
    refetchInterval: 30_000,
  });

  const state = paperStateQuery.data?.state;
  const paperMode = String(state?.settings_json?.paper_mode ?? "strategy");
  const allowedSides = Array.isArray(state?.settings_json?.allowed_sides)
    ? (state?.settings_json?.allowed_sides as string[]).join(" / ")
    : "BUY";
  const squareoffCutoff = String(state?.settings_json?.paper_short_squareoff_time ?? "15:20");

  useEffect(() => {
    setAutopilotEnabled(paperMode === "policy");
  }, [paperMode]);

  const runStepMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) =>
      (await atlasApi.paperRunStep(payload)).data,
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      toast.success("Paper step queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not run paper step");
    },
  });

  const previewMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) =>
      (await atlasApi.paperSignalsPreview(payload)).data,
    onSuccess: (payload) => {
      setPreview(payload);
      setPreviewOpen(true);
      toast.success(`Preview ready (${payload.generated_signals_count} candidates)`);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not preview signals");
    },
  });

  const captureJournalMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.captureForwardJournal({
          regime: regimeQuery.data?.regime ?? "TREND_UP",
          bundle_id: bundleId ?? undefined,
          timeframe: "1d",
          symbol_scope: "all",
          max_symbols_scan: 500,
          max_runtime_seconds: 60,
          max_entry_extension_pct: 1,
        })
      ).data,
    onSuccess: (payload) => {
      queryClient.invalidateQueries({ queryKey: qk.forwardJournal(bundleId, "1d", null, 1, 50) });
      toast.success(`Captured ${payload.captured_count} signals`);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not capture forward journal");
    },
  });

  const evaluateJournalMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.evaluateForwardJournal({
          bundle_id: bundleId ?? undefined,
          timeframe: "1d",
          horizon_bars: 5,
        })
      ).data,
    onSuccess: (payload) => {
      queryClient.invalidateQueries({ queryKey: qk.forwardJournal(bundleId, "1d", null, 1, 50) });
      toast.success(`Evaluated ${payload.evaluated_count} journal rows`);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not evaluate forward journal");
    },
  });

  const stream = useJobStream(activeJobId);
  const reportStream = useJobStream(reportJobId);

  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      queryClient.invalidateQueries({ queryKey: qk.paperState });
      queryClient.invalidateQueries({ queryKey: qk.paperPositions });
      queryClient.invalidateQueries({ queryKey: qk.paperOrders });
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      setLatestDecision((stream.result as Record<string, unknown> | null) ?? null);
      toast.success("Paper step complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Paper step failed");
    }
  }, [queryClient, stream.isTerminal, stream.result, stream.status]);

  useEffect(() => {
    if (!reportStream.isTerminal) {
      return;
    }
    if (reportStream.status === "SUCCEEDED" || reportStream.status === "DONE") {
      queryClient.invalidateQueries({ queryKey: qk.dailyReports(undefined, null, null) });
      toast.success("Daily report generated");
    }
    if (reportStream.status === "FAILED") {
      toast.error("Daily report generation failed");
    }
  }, [queryClient, reportStream.isTerminal, reportStream.status]);

  const positions = paperStateQuery.data?.positions ?? [];
  const orders = paperStateQuery.data?.orders ?? [];
  const selectedPosition = positions.find((position) => position.id === selectedPositionId) ?? null;
  const selectedOrder = orders.find((order) => order.id === selectedOrderId) ?? null;

  const promoted = useMemo(
    () => (strategiesQuery.data ?? []).filter((item) => Boolean(item.enabled)),
    [strategiesQuery.data],
  );
  const activePolicyId =
    typeof state?.settings_json?.active_policy_id === "number"
      ? (state.settings_json.active_policy_id as number)
      : null;
  const activePolicy =
    activePolicyId === null
      ? null
      : ((policiesQuery.data ?? []).find((policy) => policy.id === activePolicyId) ?? null);
  const healthStatus = operateQuery.data?.health_short?.status ?? "HEALTHY";
  const healthReasons = operateQuery.data?.health_short?.reasons_json ?? [];
  const operateMode = String(operateQuery.data?.mode ?? "NORMAL");
  const safeModeAction = String(operateQuery.data?.safe_mode_action ?? "none");
  const latestQuality = operateQuery.data?.latest_data_quality;

  useEffect(() => {
    if (bundleId !== null) {
      return;
    }
    const policyUniverse = (activePolicy?.definition_json as Record<string, unknown> | undefined)?.[
      "universe"
    ] as Record<string, unknown> | undefined;
    const policyBundle = Number(policyUniverse?.bundle_id);
    if (Number.isFinite(policyBundle) && policyBundle > 0) {
      setBundleId(policyBundle);
      return;
    }
    const firstBundle = Number((bundlesQuery.data ?? [])[0]?.id);
    if (Number.isFinite(firstBundle) && firstBundle > 0) {
      setBundleId(firstBundle);
    }
  }, [activePolicy, bundleId, bundlesQuery.data]);

  const runStep = useCallback((shadowOnly = false) => {
    const regime = regimeQuery.data?.regime ?? "TREND_UP";
    const useAutopilot = autopilotEnabled || paperMode === "policy";
    const fallbackSignals = useAutopilot
      ? []
      : [
          {
            symbol: "NIFTY500",
            side: "BUY",
            template: "trend_breakout",
            price: 1800,
            stop_distance: 40,
            target_price: 1880,
            signal_strength: 0.5,
            adv: 1_000_000_000,
            vol_scale: 0.01,
          },
        ];
    runStepMutation.mutate({
      regime,
      auto_generate_signals: useAutopilot,
      bundle_id: bundleId ?? undefined,
      signals: fallbackSignals,
      mark_prices: {},
      shadow_only: shadowOnly,
    });
  }, [autopilotEnabled, bundleId, paperMode, regimeQuery.data?.regime, runStepMutation]);

  const previewSignals = useCallback(() => {
    previewMutation.mutate({
      regime: regimeQuery.data?.regime ?? "TREND_UP",
      bundle_id: bundleId ?? undefined,
      max_symbols_scan: 50,
    });
  }, [bundleId, previewMutation, regimeQuery.data?.regime]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingElement(event.target)) {
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        if (!runStepMutation.isPending) {
          runStep(false);
        }
      }
      if (event.key.toLowerCase() === "p") {
        event.preventDefault();
        if (!previewMutation.isPending) {
          previewSignals();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [previewMutation.isPending, previewSignals, runStep, runStepMutation.isPending]);

  const skippedSignals =
    (latestDecision?.skipped_signals as Array<Record<string, unknown>> | undefined) ?? [];
  const selectedSignals =
    (latestDecision?.selected_signals as Array<Record<string, unknown>> | undefined) ?? [];
  const latestEventRiskRefresh = isRecord(latestDecision?.event_risk_refresh)
    ? (latestDecision.event_risk_refresh as ApiEventRiskRefresh)
    : null;
  const costSummary = (latestDecision?.cost_summary as Record<string, unknown> | undefined) ?? {};
  const riskScaled = Boolean(latestDecision?.risk_scaled);
  const reportId = Number(latestDecision?.report_id ?? 0);
  const effectiveContext =
    (latestDecision?.effective_context as ApiEffectiveTradingContext | undefined) ??
    (operateQuery.data?.effective_context as ApiEffectiveTradingContext | undefined) ??
    null;
  const forwardJournalRows = forwardJournalQuery.data?.data ?? [];
  const forwardJournalMeta = forwardJournalQuery.data?.meta ?? {};
  const forwardJournalSummary =
    (forwardJournalMeta.summary as ApiForwardJournalSummary | undefined) ?? null;
  const journalCounts = forwardJournalSummary?.counts ?? {};

  const generateReportMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.generateDailyReport({
          date: new Date().toISOString().slice(0, 10),
          bundle_id: bundleId ?? undefined,
          policy_id: activePolicyId ?? undefined,
        })
      ).data,
    onSuccess: (payload) => {
      setReportJobId(payload.job_id);
      toast.success("Daily report job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue daily report");
    },
  });

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Paper Step Job" />
      <JobDrawer
        jobId={reportJobId}
        onClose={() => setReportJobId(null)}
        title="Daily Report Job"
      />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Paper Trading</h2>
        <p className="mt-1 text-sm text-muted">
          Policy autopilot can generate and rank signals automatically before every paper step.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Equity: {state?.equity ?? "-"}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Cash: {state?.cash ?? "-"}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Drawdown: {state?.drawdown ?? "-"}
          </p>
        </div>
        <p className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
          Execution mode: {paperMode === "policy" ? "Policy mode" : "Single strategy mode"}
          {activePolicy ? ` (${activePolicy.name})` : ""}
        </p>
        {effectiveContext ? (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full border border-border px-2 py-1 text-muted">
              {effectiveContext.trading_date}
            </span>
            <span className="rounded-full border border-border px-2 py-1 text-muted">
              as-of {effectiveContext.data_asof_ist ?? "-"}
            </span>
            <span className="rounded-full border border-border px-2 py-1 text-muted">
              {String(effectiveContext.confidence_gate_decision ?? "PASS")}
            </span>
            <span className="rounded-full border border-border px-2 py-1 text-muted">
              scale {(Number(effectiveContext.confidence_risk_scale ?? 1) * 100).toFixed(1)}%
            </span>
            <button
              type="button"
              onClick={() => setContextOpen(true)}
              className="focus-ring rounded-full border border-border px-2 py-1 text-muted"
            >
              Context
            </button>
          </div>
        ) : null}
        <p className="mt-2 rounded-xl border border-border px-3 py-2 text-xs text-muted">
          Shorts: Allowed sides {allowedSides}. Short mode: Cash intraday (auto square-off{" "}
          {squareoffCutoff}) + Futures swing (if available).
        </p>
        <p
          className={`mt-2 rounded-xl border px-3 py-2 text-xs ${
            healthStatus === "DEGRADED"
              ? "border-danger/30 bg-danger/10 text-danger"
              : healthStatus === "WARNING"
                ? "border-warning/30 bg-warning/10 text-warning"
                : "border-success/30 bg-success/10 text-success"
          }`}
        >
          Policy health: {healthStatus}
          {healthReasons.length > 0 ? ` - ${healthReasons[0]}` : ""}
        </p>
        {operateMode === "SAFE MODE" && safeModeAction === "shadow_only" ? (
          <p className="mt-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            SAFE MODE - SHADOW. Atlas simulates full run-steps without mutating live positions/cash.
          </p>
        ) : operateMode === "SAFE MODE" ? (
          <p className="mt-2 rounded-xl border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            SAFE MODE active. New entries are blocked until data quality recovers.
          </p>
        ) : latestQuality?.status === "WARN" ? (
          <p className="mt-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            Data quality warning:{" "}
            {String(latestQuality.issues_json?.[0]?.message ?? "Check Ops page for details.")}
          </p>
        ) : null}
        <div className="mt-3">
          <label className="text-xs text-muted">
            Universe bundle
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={bundleId ?? ""}
              onChange={(event) =>
                setBundleId(event.target.value ? Number(event.target.value) : null)
              }
            >
              {(bundlesQuery.data ?? []).map((bundle) => (
                <option key={bundle.id} value={bundle.id}>
                  {bundle.name} ({bundle.symbols.length} symbols)
                </option>
              ))}
            </select>
          </label>
        </div>
        {riskScaled ? (
          <p className="mt-3 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            Risk scaled due to regime and policy constraints.
          </p>
        ) : null}
        {String(latestDecision?.execution_mode ?? "") === "SHADOW" ? (
          <p className="mt-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            SHADOW run completed. Live paper positions/cash were not modified.
          </p>
        ) : null}
        <EventRiskRefreshPanel refresh={latestEventRiskRefresh} compact />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            aria-pressed={autopilotEnabled}
            onClick={() => setAutopilotEnabled((value) => !value)}
            className={`focus-ring rounded-xl border px-3 py-2 text-sm ${
              autopilotEnabled
                ? "border-accent bg-accent/10 text-accent"
                : "border-border text-muted"
            }`}
          >
            Autopilot ({autopilotEnabled ? "On" : "Off"})
          </button>
          <button
            type="button"
            onClick={previewSignals}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            disabled={previewMutation.isPending}
          >
            {previewMutation.isPending ? "Previewing..." : "Preview Signals"}
          </button>
          <button
            type="button"
            onClick={() => runStep(false)}
            className="focus-ring rounded-xl bg-accent px-4 py-2 text-white"
            disabled={runStepMutation.isPending}
          >
            {runStepMutation.isPending ? "Queuing..." : "Run Step"}
          </button>
          <button
            type="button"
            onClick={() => runStep(true)}
            className="focus-ring rounded-xl border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning"
            disabled={runStepMutation.isPending}
          >
            {runStepMutation.isPending ? "Queuing..." : "Run Shadow Step"}
          </button>
          <button
            type="button"
            onClick={() => setWhyOpen(true)}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            disabled={!latestDecision}
          >
            Why
          </button>
          <button
            type="button"
            onClick={() => generateReportMutation.mutate()}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            disabled={generateReportMutation.isPending}
          >
            {generateReportMutation.isPending ? "Queuing report..." : "Generate Daily Report"}
          </button>
        </div>
        {reportId > 0 ? (
          <p className="mt-2 text-xs text-muted">
            Latest run report:{" "}
            <a href="/reports" className="text-accent underline-offset-2 hover:underline">
              Report #{reportId}
            </a>
          </p>
        ) : null}
        <p className="mt-2 text-xs text-muted">
          Shortcuts: Ctrl/Cmd+Enter run step, P preview signals.
        </p>
      </section>

      <section className="card p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold">Forward Test Journal</h3>
            <p className="mt-1 text-sm text-muted">
              Captures actionable signals and tracks whether they hit target, stop, or expire.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => captureJournalMutation.mutate()}
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
              disabled={captureJournalMutation.isPending || bundleId === null}
            >
              {captureJournalMutation.isPending ? "Capturing..." : "Capture Signals"}
            </button>
            <button
              type="button"
              onClick={() => evaluateJournalMutation.mutate()}
              className="focus-ring rounded-xl bg-accent px-3 py-2 text-sm text-white"
              disabled={evaluateJournalMutation.isPending || bundleId === null}
            >
              {evaluateJournalMutation.isPending ? "Evaluating..." : "Evaluate Outcomes"}
            </button>
          </div>
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-5">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Total: {forwardJournalSummary?.total ?? 0}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Open: {journalCounts.OPEN ?? 0}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            T1+: {(journalCounts.T1_HIT ?? 0) + (journalCounts.T2_HIT ?? 0)}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Stops: {journalCounts.STOP_HIT ?? 0}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            T1 rate: {((forwardJournalSummary?.t1_or_better_rate ?? 0) * 100).toFixed(1)}%
          </p>
        </div>
        {forwardJournalQuery.isLoading ? (
          <LoadingState label="Loading forward journal" />
        ) : forwardJournalQuery.isError ? (
          <ErrorState
            title="Could not load forward journal"
            action="Retry after the API is available."
            onRetry={() => {
              void forwardJournalQuery.refetch();
            }}
          />
        ) : forwardJournalRows.length === 0 ? (
          <EmptyState
            title="No captured signals"
            action="Capture signals after the daily data refresh."
          />
        ) : (
          <div className="mt-3 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[900px] text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Fill</th>
                  <th className="px-3 py-2">Plan</th>
                  <th className="px-3 py-2">Size</th>
                  <th className="px-3 py-2">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {forwardJournalRows.map((row) => (
                  <tr key={row.id} className="border-t border-border">
                    <td className="px-3 py-2">
                      <p className="font-medium">{row.symbol}</p>
                      <p className="text-xs text-muted">{row.template}</p>
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-[10px] ${journalStatusClass(row.status)}`}
                      >
                        {row.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 tabular-nums">
                      <p>{row.fill_date}</p>
                      <p className="text-xs text-muted">
                        {row.bars_observed}/{row.horizon_bars} bars
                      </p>
                    </td>
                    <td className="px-3 py-2 tabular-nums">
                      <div className="grid min-w-[150px] grid-cols-2 gap-x-3 gap-y-1 text-xs">
                        <span className="text-muted">Entry</span>
                        <span>{formatNumber(row.entry_price)}</span>
                        <span className="text-muted">Stop</span>
                        <span>{formatNumber(row.stop_price)}</span>
                        <span className="text-muted">T1</span>
                        <span>{formatNumber(row.target_1_price)}</span>
                        <span className="text-muted">T2</span>
                        <span>{formatNumber(row.target_2_price)}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2 tabular-nums">
                      <p>Qty {row.planned_qty}</p>
                      <p className="text-xs text-muted">
                        Risk {formatMoney(row.planned_risk_amount)}
                      </p>
                    </td>
                    <td className="px-3 py-2 tabular-nums">
                      <p>{formatNumber(row.close_return_pct)}%</p>
                      <p className="text-xs text-muted">
                        MFE {formatNumber(row.max_favorable_pct)}% / MAE{" "}
                        {formatNumber(row.max_adverse_pct)}%
                      </p>
                      <p className="text-xs text-muted">
                        Latest {formatNumber(row.latest_price)}{" "}
                        {row.latest_bar_date ? `on ${row.latest_bar_date}` : ""}
                      </p>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <article className="card p-4">
          <h3 className="text-base font-semibold">Promoted strategies</h3>
          {strategiesQuery.isLoading ? (
            <LoadingState label="Loading strategies" />
          ) : promoted.length === 0 ? (
            <EmptyState title="No promoted strategy" action="Promote one from Walk-Forward page." />
          ) : (
            <ul className="mt-3 space-y-2 text-sm">
              {promoted.map((item) => (
                <li key={String(item.id)} className="rounded-xl border border-border px-3 py-2">
                  {String(item.template)} ({String(item.name)})
                </li>
              ))}
            </ul>
          )}
        </article>

        <article className="card p-4">
          <h3 className="text-base font-semibold">Open positions</h3>
          {paperStateQuery.isLoading ? (
            <LoadingState label="Loading positions" />
          ) : positions.length === 0 ? (
            <EmptyState
              title="No open positions"
              action="Run paper step after promoting a strategy or using policy autopilot."
            />
          ) : (
            <ul className="mt-3 space-y-2 text-sm">
              {positions.map((position) => (
                <li key={position.id} className="rounded-xl border border-border px-3 py-2">
                  <button
                    type="button"
                    onClick={() => setSelectedPositionId(position.id)}
                    className="focus-ring text-left"
                  >
                    <span
                      className={`mr-2 inline-flex rounded-full px-2 py-0.5 text-[10px] ${
                        position.instrument_kind.includes("FUT")
                          ? "bg-warning/15 text-warning"
                          : "bg-accent/10 text-accent"
                      }`}
                    >
                      {position.instrument_kind.includes("FUT") ? "FUT" : "EQUITY"}
                    </span>
                    {position.symbol} qty {position.qty} ({position.qty_lots} lots) @{" "}
                    {position.avg_price}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </article>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Order blotter</h3>
        {paperStateQuery.isLoading ? (
          <LoadingState label="Loading orders" />
        ) : orders.length === 0 ? (
          <EmptyState title="No orders yet" action="Run paper step to generate fills." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">Instrument</th>
                  <th className="px-3 py-2">Side</th>
                  <th className="px-3 py-2">Qty</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-t border-border">
                    <td className="px-3 py-2">{order.symbol}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-[10px] ${
                          order.instrument_kind.includes("FUT")
                            ? "bg-warning/15 text-warning"
                            : "bg-accent/10 text-accent"
                        }`}
                      >
                        {order.instrument_kind.includes("FUT") ? "FUT" : "EQUITY"}
                      </span>
                    </td>
                    <td className="px-3 py-2">{order.side}</td>
                    <td className="px-3 py-2">
                      {order.qty} ({order.qty_lots} lots)
                    </td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => setSelectedOrderId(order.id)}
                        className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                      >
                        {order.status}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {(paperStateQuery.isError ||
        strategiesQuery.isError ||
        regimeQuery.isError ||
        policiesQuery.isError ||
        operateQuery.isError) && (
        <ErrorState
          title="Could not load paper trading state"
          action="Check API status and retry."
          onRetry={() => {
            void paperStateQuery.refetch();
            void strategiesQuery.refetch();
            void regimeQuery.refetch();
            void policiesQuery.refetch();
            void operateQuery.refetch();
          }}
        />
      )}

      <DetailsDrawer
        open={Boolean(selectedPosition)}
        onClose={() => setSelectedPositionId(null)}
        title={`Position ${selectedPosition?.symbol ?? ""}`}
      >
        {selectedPosition ? (
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-muted">Opened:</span> {selectedPosition.opened_at}
            </p>
            <p>
              <span className="text-muted">Side:</span> {selectedPosition.side}
            </p>
            <p>
              <span className="text-muted">Instrument:</span> {selectedPosition.instrument_kind}
            </p>
            <p>
              <span className="text-muted">Quantity:</span> {selectedPosition.qty}
            </p>
            <p>
              <span className="text-muted">Lots:</span> {selectedPosition.qty_lots}
            </p>
            <p>
              <span className="text-muted">Reserved margin:</span>{" "}
              {selectedPosition.margin_reserved}
            </p>
            <p>
              <span className="text-muted">Average price:</span> {selectedPosition.avg_price}
            </p>
            <p>
              <span className="text-muted">Stop:</span> {selectedPosition.stop_price ?? "-"}
            </p>
            <p>
              <span className="text-muted">Target:</span> {selectedPosition.target_price ?? "-"}
            </p>
            <p>
              <span className="text-muted">EOD square-off:</span>{" "}
              {selectedPosition.must_exit_by_eod ? "Yes" : "No"}
            </p>
          </div>
        ) : null}
      </DetailsDrawer>

      <DetailsDrawer
        open={Boolean(selectedOrder)}
        onClose={() => setSelectedOrderId(null)}
        title={`Order ${selectedOrder?.symbol ?? ""}`}
      >
        {selectedOrder ? (
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-muted">Created:</span> {selectedOrder.created_at}
            </p>
            <p>
              <span className="text-muted">Side:</span> {selectedOrder.side}
            </p>
            <p>
              <span className="text-muted">Instrument:</span> {selectedOrder.instrument_kind}
            </p>
            <p>
              <span className="text-muted">Quantity:</span> {selectedOrder.qty}
            </p>
            <p>
              <span className="text-muted">Lots:</span> {selectedOrder.qty_lots}
            </p>
            <p>
              <span className="text-muted">Fill:</span> {selectedOrder.fill_price ?? "-"}
            </p>
            <p>
              <span className="text-muted">Status:</span> {selectedOrder.status}
            </p>
            <p>
              <span className="text-muted">Reason:</span> {selectedOrder.reason ?? "-"}
            </p>
          </div>
        ) : null}
      </DetailsDrawer>

      <DetailsDrawer
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
        title="Signal preview"
      >
        {!preview ? (
          <EmptyState title="No preview data" action="Run Preview Signals to inspect candidates." />
        ) : (
          <div className="space-y-2 text-sm">
            <EventRiskRefreshPanel refresh={preview.event_risk_refresh} />
            {preview.signals.length === 0 ? (
              <EmptyState
                title="No signals generated"
                action="Adjust policy, dataset, or timeframe settings."
              />
            ) : (
              <>
                <p>
                  <span className="text-muted">Regime:</span> {preview.regime}
                </p>
                <p>
                  <span className="text-muted">Generated candidates:</span>{" "}
                  {preview.generated_signals_count}
                </p>
                <p>
                  <span className="text-muted">Candidate quality:</span> PASS{" "}
                  {preview.candidate_quality?.counts?.PASS ?? 0} / WARN{" "}
                  {preview.candidate_quality?.counts?.WARN ?? 0} / FAIL{" "}
                  {preview.candidate_quality?.counts?.FAIL ?? 0}
                </p>
                <p>
                  <span className="text-muted">Policy status:</span>{" "}
                  {preview.policy_status ?? "-"} / {preview.health_status ?? "-"}
                </p>
                <p>
                  <span className="text-muted">Bundle:</span> {preview.bundle_id ?? "-"}
                </p>
                <p>
                  <span className="text-muted">Scan:</span> {preview.scanned_symbols ?? 0}/
                  {preview.total_symbols ?? 0}
                  {preview.scan_truncated ? " (truncated)" : ""}
                </p>
                <p>
                  <span className="text-muted">Trade plan:</span> equity{" "}
                  {formatMoney(preview.trade_plan?.equity)}, risk/trade{" "}
                  {formatMoney(preview.trade_plan?.risk_amount)} (
                  {((preview.trade_plan?.risk_per_trade ?? 0) * 100).toFixed(2)}%), max
                  positions {preview.trade_plan?.max_positions ?? "-"}
                </p>
                <div className="max-h-[380px] overflow-auto rounded-xl border border-border">
                  <table className="w-full min-w-[1120px] text-xs">
                    <thead className="bg-surface text-left text-muted">
                      <tr>
                        <th className="px-2 py-2">Symbol</th>
                        <th className="px-2 py-2">Side</th>
                        <th className="px-2 py-2">Instrument</th>
                        <th className="px-2 py-2">Template</th>
                        <th className="px-2 py-2">Why</th>
                        <th className="px-2 py-2">Plan</th>
                        <th className="px-2 py-2">Size</th>
                        <th className="px-2 py-2">Quality</th>
                        <th className="px-2 py-2">Flags</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.signals.slice(0, 50).map((signal, index) => {
                        const eventRiskEvents = eventRiskEventsFromSignal(signal);
                        return (
                          <tr
                            key={`${signal.symbol}-${signal.template}-${signal.timeframe}-${index}`}
                            className="border-t border-border"
                          >
                            <td className="px-2 py-2">{signal.symbol}</td>
                            <td className="px-2 py-2">{signal.side}</td>
                            <td className="px-2 py-2">
                              {signal.instrument_kind ?? "EQUITY_CASH"} ({signal.lot_size ?? 1})
                            </td>
                            <td className="px-2 py-2">{signal.template}</td>
                            <td className="max-w-[300px] px-2 py-2 text-muted">
                              <p>{signal.explanation ?? "-"}</p>
                              <EventRiskEventsList events={eventRiskEvents} compact />
                            </td>
                            <td className="px-2 py-2 tabular-nums">
                              <div className="grid min-w-[156px] grid-cols-2 gap-x-3 gap-y-1">
                                <span className="text-muted">Entry</span>
                                <span>{formatNumber(signal.entry_price ?? signal.price)}</span>
                                <span className="text-muted">Stop</span>
                                <span>{formatNumber(signal.stop_price)}</span>
                                <span className="text-muted">T1</span>
                                <span>{formatNumber(signal.target_1_price)}</span>
                                <span className="text-muted">T2</span>
                                <span>{formatNumber(signal.target_2_price)}</span>
                              </div>
                            </td>
                            <td className="px-2 py-2 tabular-nums">
                              <div className="grid min-w-[150px] grid-cols-2 gap-x-3 gap-y-1">
                                <span className="text-muted">Qty</span>
                                <span>{signal.planned_qty ?? 0}</span>
                                <span className="text-muted">Risk</span>
                                <span>{formatMoney(signal.planned_risk_amount)}</span>
                                <span className="text-muted">Value</span>
                                <span>{formatMoney(signal.planned_position_value)}</span>
                                <span className="text-muted">Strength</span>
                                <span>{signal.signal_strength.toFixed(3)}</span>
                              </div>
                              {signal.position_size_status &&
                              signal.position_size_status !== "OK" ? (
                                <p className="mt-1 text-[10px] text-warning">
                                  Risk cap gives 0 qty
                                </p>
                              ) : null}
                            </td>
                            <td className="px-2 py-2">
                              <span
                                className={`inline-flex rounded-full px-2 py-0.5 text-[10px] ${qualityBadgeClass(signal.quality_status)}`}
                              >
                                {signal.quality_status ?? "PASS"}{" "}
                                {signal.quality_score?.toFixed(2) ?? ""}
                              </span>
                            </td>
                            <td className="px-2 py-2 text-muted">
                              {(signal.quality_flags ?? []).slice(0, 2).join(", ") || "-"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </DetailsDrawer>

      <DetailsDrawer open={whyOpen} onClose={() => setWhyOpen(false)} title="Why this run step">
        {!latestDecision ? (
          <EmptyState
            title="No completed step yet"
            action="Run a paper step to inspect decision reasons."
          />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              <span className="text-muted">Policy mode:</span>{" "}
              {String(latestDecision.policy_mode ?? "-")}
            </p>
            <p>
              <span className="text-muted">Selection reason:</span>{" "}
              {String(latestDecision.policy_selection_reason ?? "-")}
            </p>
            <p>
              <span className="text-muted">Signals source:</span>{" "}
              {String(latestDecision.signals_source ?? "-")}
            </p>
            <p>
              <span className="text-muted">Execution mode:</span>{" "}
              {String(latestDecision.execution_mode ?? "LIVE")}
            </p>
            <p>
              <span className="text-muted">Safe mode:</span>{" "}
              {String(
                (latestDecision.safe_mode as Record<string, unknown> | undefined)?.active
                  ? "active"
                  : "inactive",
              )}
              {(() => {
                const mode = latestDecision.safe_mode as Record<string, unknown> | undefined;
                const reason = mode?.reason;
                return reason ? ` (${String(reason)})` : "";
              })()}
            </p>
            <p>
              <span className="text-muted">Selected:</span> {selectedSignals.length} |{" "}
              <span className="text-muted">Skipped:</span> {skippedSignals.length}
            </p>
            <EventRiskRefreshPanel refresh={latestEventRiskRefresh} />
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Selected trades
              </p>
              {selectedSignals.length === 0 ? (
                <p className="text-xs text-muted">No selected signals.</p>
              ) : (
                <ul className="space-y-3 text-xs">
                  {selectedSignals.slice(0, 12).map((signal, index) => {
                    const fillBar = signal.fill_bar as Record<string, unknown> | undefined;
                    const fillLow = fillBar ? asNumber(fillBar.low) : null;
                    const fillHigh = fillBar ? asNumber(fillBar.high) : null;
                    const qualityFlags = Array.isArray(signal.quality_flags)
                      ? signal.quality_flags.map(String).join(", ")
                      : "";
                    const eventRiskEvents = eventRiskEventsFromSignal(signal);
                    return (
                      <li
                        key={`${asText(signal.symbol)}-${asText(signal.template)}-${index}`}
                        className="rounded-lg border border-border px-3 py-2"
                      >
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                          <span className="font-semibold">{asText(signal.symbol)}</span>
                          <span className="text-muted">{asText(signal.side)}</span>
                          <span className="text-muted">{asText(signal.template)}</span>
                          <span
                            className={`inline-flex rounded-full px-2 py-0.5 text-[10px] ${qualityBadgeClass(
                              asText(signal.quality_status),
                            )}`}
                          >
                            {asText(signal.quality_status)}
                          </span>
                        </div>
                        <p className="mt-1 text-muted">{asText(signal.explanation)}</p>
                        <div className="mt-2 grid gap-x-4 gap-y-1 tabular-nums sm:grid-cols-2">
                          <span>
                            <span className="text-muted">Signal:</span>{" "}
                            {asText(signal.signal_at)}
                          </span>
                          <span>
                            <span className="text-muted">Fill:</span> {asText(signal.fill_at)}
                          </span>
                          <span>
                            <span className="text-muted">Entry:</span>{" "}
                            {formatRecordNumber(signal, "fill_price")}
                          </span>
                          <span>
                            <span className="text-muted">Stop:</span>{" "}
                            {formatRecordNumber(signal, "stop_price")}
                          </span>
                          <span>
                            <span className="text-muted">Strength:</span>{" "}
                            {formatRecordNumber(signal, "signal_strength", 3)}
                          </span>
                          <span>
                            <span className="text-muted">Quality:</span>{" "}
                            {formatRecordNumber(signal, "quality_score", 2)}
                          </span>
                          <span>
                            <span className="text-muted">Fill bar H/L:</span>{" "}
                            {formatNumber(fillHigh)} / {formatNumber(fillLow)}
                          </span>
                          <span>
                            <span className="text-muted">Qty:</span>{" "}
                            {formatRecordNumber(signal, "qty", 0)}
                          </span>
                        </div>
                        {qualityFlags ? (
                          <p className="mt-1 text-muted">Flags: {qualityFlags}</p>
                        ) : null}
                        <EventRiskEventsList events={eventRiskEvents} />
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
            <p>
              <span className="text-muted">Cost total:</span> {String(costSummary.total_cost ?? 0)}
            </p>
            <p>
              <span className="text-muted">Engine:</span>{" "}
              {String(latestDecision.paper_engine ?? "legacy")}
            </p>
            <p>
              <span className="text-muted">Repro:</span>{" "}
              {String(latestDecision.engine_version ?? "-")} / {String(latestDecision.seed ?? "-")}
            </p>
            <p className="truncate">
              <span className="text-muted">Digest:</span>{" "}
              {String(latestDecision.data_digest ?? "-")}
            </p>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Skipped reasons
              </p>
              {skippedSignals.length === 0 ? (
                <p className="text-xs text-muted">No skipped signals.</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {skippedSignals.slice(0, 25).map((item, index) => {
                    const eventRiskEvents = eventRiskEventsFromSignal(item);
                    return (
                      <li key={`${String(item.symbol ?? "item")}-${index}`}>
                        <p>
                          {String(item.symbol ?? "-")}: {String(item.reason ?? "-")}
                        </p>
                        <EventRiskEventsList events={eventRiskEvents} compact />
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
        )}
      </DetailsDrawer>
      <DetailsDrawer
        open={contextOpen}
        onClose={() => setContextOpen(false)}
        title="Effective Trading Context"
      >
        {effectiveContext ? (
          <pre className="max-h-[360px] overflow-auto rounded-xl border border-border bg-surface p-3 text-xs text-muted">
            {JSON.stringify(effectiveContext, null, 2)}
          </pre>
        ) : (
          <EmptyState title="Context unavailable" action="Run a paper step to populate context." />
        )}
      </DetailsDrawer>
    </div>
  );
}
