"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import type { ApiPaperSignalPreview } from "@/src/lib/api/types";
import { qk } from "@/src/lib/query/keys";

function isTypingElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || target.isContentEditable;
}

export default function PaperTradingPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedPositionId, setSelectedPositionId] = useState<number | null>(null);
  const [selectedOrderId, setSelectedOrderId] = useState<number | null>(null);
  const [autopilotEnabled, setAutopilotEnabled] = useState(true);
  const [preview, setPreview] = useState<ApiPaperSignalPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [whyOpen, setWhyOpen] = useState(false);
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

  const state = paperStateQuery.data?.state;
  const paperMode = String(state?.settings_json?.paper_mode ?? "strategy");

  useEffect(() => {
    setAutopilotEnabled(paperMode === "policy");
  }, [paperMode]);

  const runStepMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => (await atlasApi.paperRunStep(payload)).data,
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
      toast.success(`Preview ready (${payload.generated_signals_count} signals)`);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not preview signals");
    },
  });

  const stream = useJobStream(activeJobId);

  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      queryClient.invalidateQueries({ queryKey: qk.paperState });
      queryClient.invalidateQueries({ queryKey: qk.paperPositions });
      queryClient.invalidateQueries({ queryKey: qk.paperOrders });
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      setLatestDecision((stream.result as Record<string, unknown> | null) ?? null);
      toast.success("Paper step complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Paper step failed");
    }
  }, [queryClient, stream.isTerminal, stream.result, stream.status]);

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

  const runStep = useCallback(() => {
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
      signals: fallbackSignals,
      mark_prices: {},
    });
  }, [autopilotEnabled, paperMode, regimeQuery.data?.regime, runStepMutation]);

  const previewSignals = useCallback(() => {
    previewMutation.mutate({
      regime: regimeQuery.data?.regime ?? "TREND_UP",
      max_symbols_scan: 50,
    });
  }, [previewMutation, regimeQuery.data?.regime]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingElement(event.target)) {
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        if (!runStepMutation.isPending) {
          runStep();
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

  const skippedSignals = (latestDecision?.skipped_signals as Array<Record<string, unknown>> | undefined) ?? [];
  const selectedSignals = (latestDecision?.selected_signals as Array<Record<string, unknown>> | undefined) ?? [];
  const costSummary = (latestDecision?.cost_summary as Record<string, unknown> | undefined) ?? {};
  const riskScaled = Boolean(latestDecision?.risk_scaled);

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Paper Step Job" />

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
        {riskScaled ? (
          <p className="mt-3 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            Risk scaled due to regime and policy constraints.
          </p>
        ) : null}
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
            onClick={runStep}
            className="focus-ring rounded-xl bg-accent px-4 py-2 text-white"
            disabled={runStepMutation.isPending}
          >
            {runStepMutation.isPending ? "Queuing..." : "Run Step"}
          </button>
          <button
            type="button"
            onClick={() => setWhyOpen(true)}
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
            disabled={!latestDecision}
          >
            Why
          </button>
        </div>
        <p className="mt-2 text-xs text-muted">Shortcuts: Ctrl/Cmd+Enter run step, P preview signals.</p>
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
                    {position.symbol} qty {position.qty} @ {position.avg_price}
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
                  <th className="px-3 py-2">Side</th>
                  <th className="px-3 py-2">Qty</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-t border-border">
                    <td className="px-3 py-2">{order.symbol}</td>
                    <td className="px-3 py-2">{order.side}</td>
                    <td className="px-3 py-2">{order.qty}</td>
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
        policiesQuery.isError) && (
        <ErrorState
          title="Could not load paper trading state"
          action="Check API status and retry."
          onRetry={() => {
            void paperStateQuery.refetch();
            void strategiesQuery.refetch();
            void regimeQuery.refetch();
            void policiesQuery.refetch();
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
              <span className="text-muted">Quantity:</span> {selectedPosition.qty}
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
              <span className="text-muted">Quantity:</span> {selectedOrder.qty}
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

      <DetailsDrawer open={previewOpen} onClose={() => setPreviewOpen(false)} title="Signal preview">
        {!preview ? (
          <EmptyState title="No preview data" action="Run Preview Signals to inspect candidates." />
        ) : preview.signals.length === 0 ? (
          <EmptyState title="No signals generated" action="Adjust policy, dataset, or timeframe settings." />
        ) : (
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-muted">Regime:</span> {preview.regime}
            </p>
            <p>
              <span className="text-muted">Generated:</span> {preview.generated_signals_count}
            </p>
            <div className="max-h-[380px] overflow-auto rounded-xl border border-border">
              <table className="w-full text-xs">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-2 py-2">Symbol</th>
                    <th className="px-2 py-2">Template</th>
                    <th className="px-2 py-2">Strength</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.signals.slice(0, 50).map((signal) => (
                    <tr key={`${signal.symbol}-${signal.template}-${signal.timeframe}`} className="border-t border-border">
                      <td className="px-2 py-2">{signal.symbol}</td>
                      <td className="px-2 py-2">{signal.template}</td>
                      <td className="px-2 py-2">{signal.signal_strength.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </DetailsDrawer>

      <DetailsDrawer open={whyOpen} onClose={() => setWhyOpen(false)} title="Why this run step">
        {!latestDecision ? (
          <EmptyState title="No completed step yet" action="Run a paper step to inspect decision reasons." />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              <span className="text-muted">Policy mode:</span> {String(latestDecision.policy_mode ?? "-")}
            </p>
            <p>
              <span className="text-muted">Selection reason:</span>{" "}
              {String(latestDecision.policy_selection_reason ?? "-")}
            </p>
            <p>
              <span className="text-muted">Signals source:</span> {String(latestDecision.signals_source ?? "-")}
            </p>
            <p>
              <span className="text-muted">Selected:</span> {selectedSignals.length} |{" "}
              <span className="text-muted">Skipped:</span> {skippedSignals.length}
            </p>
            <p>
              <span className="text-muted">Cost total:</span> {String(costSummary.total_cost ?? 0)}
            </p>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Skipped reasons</p>
              {skippedSignals.length === 0 ? (
                <p className="text-xs text-muted">No skipped signals.</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {skippedSignals.slice(0, 25).map((item, index) => (
                    <li key={`${String(item.symbol ?? "item")}-${index}`}>
                      {String(item.symbol ?? "-")}: {String(item.reason ?? "-")}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}

