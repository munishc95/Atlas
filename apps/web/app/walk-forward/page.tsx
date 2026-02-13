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

type FoldPayload = {
  fold_index: number;
  test_end: string;
  oos_score: number;
  stress_pass: boolean;
  params: Record<string, unknown>;
};

export default function WalkForwardPage() {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState("NIFTY500");
  const [timeframe, setTimeframe] = useState("1d");
  const [template, setTemplate] = useState("trend_breakout");
  const [trainMonths, setTrainMonths] = useState(60);
  const [testMonths, setTestMonths] = useState(9);
  const [stepMonths, setStepMonths] = useState(3);
  const [trials, setTrials] = useState(20);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [selectedFoldIndex, setSelectedFoldIndex] = useState<number | null>(null);

  const templatesQuery = useQuery({
    queryKey: qk.strategyTemplates,
    queryFn: async () => (await atlasApi.strategyTemplates()).data,
  });

  useEffect(() => {
    if (!templatesQuery.data || templatesQuery.data.length === 0) {
      return;
    }
    if (!templatesQuery.data.some((item) => String(item.key) === template)) {
      const first = templatesQuery.data[0];
      if (first) {
        setTemplate(String(first.key));
      }
    }
  }, [template, templatesQuery.data]);

  const runMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runWalkForward({
          symbol,
          timeframe,
          strategy_template: template,
          config: {
            train_months: trainMonths,
            test_months: testMonths,
            step_months: stepMonths,
            trials,
            sampler: "tpe",
            pruner: "median",
          },
        })
      ).data,
    onSuccess: (result) => {
      setActiveJobId(result.job_id);
      toast.success("Walk-forward queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not start walk-forward run");
    },
  });

  const stream = useJobStream(activeJobId);

  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      const nextRun = Number((stream.result as { run_id?: number } | null)?.run_id);
      if (Number.isFinite(nextRun) && nextRun > 0) {
        setRunId(nextRun);
      }
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Walk-forward complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Walk-forward failed");
    }
  }, [queryClient, stream.isTerminal, stream.result, stream.status]);

  const walkForwardQuery = useQuery({
    queryKey: qk.walkForward(runId),
    queryFn: async () => (await atlasApi.walkForwardById(runId as number)).data,
    enabled: runId !== null,
  });

  const summary = useMemo(
    () => (walkForwardQuery.data?.summary_json ?? {}) as Record<string, unknown>,
    [walkForwardQuery.data],
  );
  const oosSummary = useMemo(
    () =>
      (summary.oos_only ?? {
        fold_profitability_pct: 0,
        worst_fold_drawdown: 0,
        parameter_stability_score: 0,
        stress_pass_rate: 0,
      }) as Record<string, number>,
    [summary],
  );
  const folds = useMemo(() => (summary.folds ?? []) as FoldPayload[], [summary]);
  const selectedFold = useMemo(
    () => folds.find((fold) => fold.fold_index === selectedFoldIndex) ?? null,
    [folds, selectedFoldIndex],
  );
  const eligible = Boolean(summary.eligible_for_promotion);

  const bestFold = useMemo(() => {
    if (folds.length === 0) {
      return null;
    }
    return [...folds].sort((a, b) => b.oos_score - a.oos_score)[0];
  }, [folds]);

  const promoteMutation = useMutation({
    mutationFn: async () => {
      if (!bestFold) {
        throw new Error("No valid fold to promote");
      }
      return (
        await atlasApi.promoteStrategy({
          strategy_name: `${template}-${new Date().toISOString().slice(0, 10)}`,
          template,
          params_json: bestFold.params,
        })
      ).data;
    },
    onSuccess: () => {
      toast.success("Strategy promoted to paper mode");
      queryClient.invalidateQueries({ queryKey: qk.strategies });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Promotion failed");
    },
  });

  return (
    <div className="space-y-5">
      <JobDrawer
        jobId={activeJobId}
        onClose={() => setActiveJobId(null)}
        title="Walk-Forward Job"
      />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Walk-Forward & Robustness</h2>
        <p className="mt-1 text-sm text-muted">
          IS/OOS split, fold stability, and stress tests before promotion.
        </p>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <label className="text-sm text-muted">
            Symbol
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
            />
          </label>
          <label className="text-sm text-muted">
            Timeframe
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
            >
              <option value="1d">1d</option>
              <option value="4h_ish">4h_ish</option>
            </select>
          </label>
          <label className="text-sm text-muted">
            Strategy
            <select
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              value={template}
              onChange={(event) => setTemplate(event.target.value)}
            >
              {(templatesQuery.data ?? []).map((item) => (
                <option key={String(item.key)} value={String(item.key)}>
                  {String(item.name)}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm text-muted">
            Train window (months)
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              type="number"
              value={trainMonths}
              onChange={(event) => setTrainMonths(Number(event.target.value))}
            />
          </label>
          <label className="text-sm text-muted">
            Test window (months)
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              type="number"
              value={testMonths}
              onChange={(event) => setTestMonths(Number(event.target.value))}
            />
          </label>
          <label className="text-sm text-muted">
            Step (months)
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              type="number"
              value={stepMonths}
              onChange={(event) => setStepMonths(Number(event.target.value))}
            />
          </label>
          <label className="text-sm text-muted">
            Optuna trials
            <input
              className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
              type="number"
              value={trials}
              onChange={(event) => setTrials(Number(event.target.value))}
            />
          </label>
        </div>

        <button
          type="button"
          onClick={() => runMutation.mutate()}
          className="focus-ring mt-4 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
          disabled={runMutation.isPending}
        >
          {runMutation.isPending ? "Queuing..." : "Run Walk-Forward"}
        </button>
      </section>

      <section className="card p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold">Fold summary</h3>
          <span
            className={`badge ${eligible ? "bg-success/15 text-success" : "bg-warning/15 text-warning"}`}
          >
            {eligible ? "Eligible for promotion" : "Blocked"}
          </span>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-4">
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            OOS profitable folds:{" "}
            {(Number(oosSummary.fold_profitability_pct ?? 0) * 100).toFixed(1)}%
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Worst fold drawdown: {Number(oosSummary.worst_fold_drawdown ?? 0).toFixed(3)}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Parameter stability: {Number(oosSummary.parameter_stability_score ?? 0).toFixed(3)}
          </p>
          <p className="rounded-xl border border-border px-3 py-2 text-sm">
            Stress pass rate: {(Number(oosSummary.stress_pass_rate ?? 0) * 100).toFixed(1)}%
          </p>
        </div>

        {walkForwardQuery.isLoading && runId !== null ? (
          <LoadingState label="Loading walk-forward summary" />
        ) : walkForwardQuery.isError ? (
          <ErrorState
            title="Could not load walk-forward results"
            action="Retry after job completion."
            onRetry={() => void walkForwardQuery.refetch()}
          />
        ) : folds.length === 0 ? (
          <EmptyState title="No folds yet" action="Run a walk-forward job." />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Fold</th>
                  <th className="px-3 py-2">Test end</th>
                  <th className="px-3 py-2">OOS score</th>
                  <th className="px-3 py-2">Stress</th>
                </tr>
              </thead>
              <tbody>
                {folds.map((fold) => (
                  <tr key={fold.fold_index} className="border-t border-border">
                    <td className="px-3 py-2">{fold.fold_index}</td>
                    <td className="px-3 py-2">{fold.test_end}</td>
                    <td className="px-3 py-2">{fold.oos_score.toFixed(4)}</td>
                    <td
                      className={`px-3 py-2 ${fold.stress_pass ? "text-success" : "text-warning"}`}
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedFoldIndex(fold.fold_index)}
                        className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                      >
                        {fold.stress_pass ? "Pass" : "Fail"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          <button
            type="button"
            className="focus-ring rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
            onClick={() => promoteMutation.mutate()}
            disabled={!eligible || promoteMutation.isPending || !bestFold}
          >
            {promoteMutation.isPending ? "Promoting..." : "Promote to Paper"}
          </button>
          {!eligible && (
            <p className="text-xs text-muted">
              Promotion blocked. Check rejection reasons in summary payload.
            </p>
          )}
        </div>
      </section>

      <DetailsDrawer
        open={Boolean(selectedFold)}
        onClose={() => setSelectedFoldIndex(null)}
        title={`Fold ${selectedFold?.fold_index ?? ""} Details`}
      >
        {selectedFold ? (
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-muted">Test end:</span> {selectedFold.test_end}
            </p>
            <p>
              <span className="text-muted">OOS score:</span> {selectedFold.oos_score.toFixed(4)}
            </p>
            <p>
              <span className="text-muted">Stress:</span>{" "}
              {selectedFold.stress_pass ? "Pass" : "Fail"}
            </p>
            <p className="text-sm font-semibold">Parameters</p>
            <pre className="overflow-auto rounded-xl bg-surface p-2 text-xs">
              {JSON.stringify(selectedFold.params, null, 2)}
            </pre>
          </div>
        ) : null}
      </DetailsDrawer>
    </div>
  );
}
