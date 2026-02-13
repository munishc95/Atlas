"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import { qk } from "@/src/lib/query/keys";

export default function PaperTradingPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedPositionId, setSelectedPositionId] = useState<number | null>(null);
  const [selectedOrderId, setSelectedOrderId] = useState<number | null>(null);

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

  const runStepMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.paperRunStep({
          regime: "TREND_UP",
          signals: [
            {
              symbol: "NIFTY500",
              side: "BUY",
              template: "trend_breakout",
              price: 1800,
              stop_distance: 40,
              target_price: 1880,
            },
          ],
          mark_prices: { NIFTY500: 1810 },
        })
      ).data,
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      toast.success("Paper step queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not run paper step");
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
      toast.success("Paper step complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Paper step failed");
    }
  }, [queryClient, stream.isTerminal, stream.status]);

  const positions = paperStateQuery.data?.positions ?? [];
  const orders = paperStateQuery.data?.orders ?? [];
  const state = paperStateQuery.data?.state;
  const selectedPosition = positions.find((position) => position.id === selectedPositionId) ?? null;
  const selectedOrder = orders.find((order) => order.id === selectedOrderId) ?? null;
  const riskScaled = regimeQuery.data?.regime === "HIGH_VOL";

  const promoted = useMemo(
    () => (strategiesQuery.data ?? []).filter((item) => Boolean(item.enabled)),
    [strategiesQuery.data],
  );

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Paper Step Job" />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Paper Trading</h2>
        <p className="mt-1 text-sm text-muted">Live signal queue, paper blotter, and stop/target management.</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">Equity: {state?.equity ?? "-"}</p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">Cash: {state?.cash ?? "-"}</p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">Drawdown: {state?.drawdown ?? "-"}</p>
        </div>
        {riskScaled ? (
          <p className="mt-3 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            Risk scaled due to regime: HIGH_VOL policy active (lower risk per trade and max positions).
          </p>
        ) : null}
        <button
          type="button"
          onClick={() => runStepMutation.mutate()}
          className="focus-ring mt-4 rounded-xl bg-accent px-4 py-2 text-white"
          disabled={runStepMutation.isPending}
        >
          {runStepMutation.isPending ? "Queuing..." : "Run Step"}
        </button>
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
            <EmptyState title="No open positions" action="Run paper step after promoting a strategy." />
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

      {(paperStateQuery.isError || strategiesQuery.isError || regimeQuery.isError) && (
        <ErrorState
          title="Could not load paper trading state"
          action="Check API status and retry."
          onRetry={() => {
            void paperStateQuery.refetch();
            void strategiesQuery.refetch();
            void regimeQuery.refetch();
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
            <p><span className="text-muted">Opened:</span> {selectedPosition.opened_at}</p>
            <p><span className="text-muted">Quantity:</span> {selectedPosition.qty}</p>
            <p><span className="text-muted">Average price:</span> {selectedPosition.avg_price}</p>
            <p><span className="text-muted">Stop:</span> {selectedPosition.stop_price ?? "-"}</p>
            <p><span className="text-muted">Target:</span> {selectedPosition.target_price ?? "-"}</p>
            <p className="text-xs text-muted">
              Stop/target update timeline is not yet persisted per event in the current paper schema.
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
            <p><span className="text-muted">Created:</span> {selectedOrder.created_at}</p>
            <p><span className="text-muted">Side:</span> {selectedOrder.side}</p>
            <p><span className="text-muted">Quantity:</span> {selectedOrder.qty}</p>
            <p><span className="text-muted">Fill:</span> {selectedOrder.fill_price ?? "-"}</p>
            <p><span className="text-muted">Status:</span> {selectedOrder.status}</p>
            <p><span className="text-muted">Reason:</span> {selectedOrder.reason ?? "-"}</p>
          </div>
        ) : null}
      </DetailsDrawer>
    </div>
  );
}
