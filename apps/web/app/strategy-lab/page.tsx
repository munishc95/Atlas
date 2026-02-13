"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { EquityChart } from "@/components/equity-chart";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { MetricTile } from "@/components/metric-tile";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { asMetricHelpKey } from "@/lib/metric-help";
import { atlasApi } from "@/src/lib/api/endpoints";
import { useJobStream } from "@/src/hooks/useJobStream";
import { qk } from "@/src/lib/query/keys";

function parseNumeric(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function StrategyLabPage() {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState("NIFTY500");
  const [timeframe, setTimeframe] = useState("1d");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("trend_breakout");
  const [params, setParams] = useState<Record<string, number>>({});
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [backtestId, setBacktestId] = useState<number | null>(null);
  const [tradePage, setTradePage] = useState(1);
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);
  const [runsPage, setRunsPage] = useState(1);
  const [runsTemplateFilter, setRunsTemplateFilter] = useState("");
  const [runsTimeframeFilter, setRunsTimeframeFilter] = useState("");
  const [runsSortBy, setRunsSortBy] = useState("created_at");
  const [runsSortDir, setRunsSortDir] = useState<"asc" | "desc">("desc");
  const [compareIds, setCompareIds] = useState<number[]>([]);

  const templatesQuery = useQuery({
    queryKey: qk.strategyTemplates,
    queryFn: async () => (await atlasApi.strategyTemplates()).data,
  });

  useEffect(() => {
    if (!templatesQuery.data || templatesQuery.data.length === 0) {
      return;
    }
    const fallback = templatesQuery.data.find((x) => String(x.key) === selectedTemplate) ?? templatesQuery.data[0];
    if (!fallback) {
      return;
    }
    const defaults = (fallback.default_params ?? {}) as Record<string, number>;
    setSelectedTemplate(String(fallback.key));
    setParams(defaults);
  }, [templatesQuery.data, selectedTemplate]);

  const runBacktestMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runBacktest({
          symbol,
          timeframe,
          strategy_template: selectedTemplate,
          params,
          config: {
            atr_stop_mult: params.atr_stop_mult ?? 2,
            atr_trail_mult: params.atr_trail_mult ?? 2,
          },
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast.success("Backtest queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Backtest failed to start");
    },
  });

  const stream = useJobStream(activeJobId);

  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      const maybeBacktestId = Number((stream.result as { backtest_id?: number } | null)?.backtest_id);
      if (Number.isFinite(maybeBacktestId) && maybeBacktestId > 0) {
        setBacktestId(maybeBacktestId);
        setTradePage(1);
      }
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Backtest complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Backtest failed");
    }
  }, [queryClient, stream.isTerminal, stream.result, stream.status]);

  const backtestQuery = useQuery({
    queryKey: qk.backtest(backtestId),
    queryFn: async () => (await atlasApi.backtestById(backtestId as number)).data,
    enabled: backtestId !== null,
  });

  const equityQuery = useQuery({
    queryKey: qk.backtestEquity(backtestId),
    queryFn: async () => (await atlasApi.backtestEquity(backtestId as number)).data,
    enabled: backtestId !== null,
  });

  const tradesQuery = useQuery({
    queryKey: qk.backtestTrades(backtestId, tradePage, 20),
    queryFn: async () => await atlasApi.backtestTrades(backtestId as number, tradePage, 20),
    enabled: backtestId !== null,
  });

  const templateCards = templatesQuery.data ?? [];
  const runsFiltersKey = useMemo(
    () =>
      JSON.stringify({
        page: runsPage,
        template: runsTemplateFilter || undefined,
        timeframe: runsTimeframeFilter || undefined,
        sort_by: runsSortBy,
        sort_dir: runsSortDir,
      }),
    [runsPage, runsSortBy, runsSortDir, runsTemplateFilter, runsTimeframeFilter],
  );
  const runsQuery = useQuery({
    queryKey: qk.backtestRuns(runsFiltersKey),
    queryFn: async () =>
      await atlasApi.backtests({
        page: runsPage,
        page_size: 12,
        template: runsTemplateFilter || undefined,
        timeframe: runsTimeframeFilter || undefined,
        sort_by: runsSortBy,
        sort_dir: runsSortDir,
      }),
  });
  const compareQuery = useQuery({
    queryKey: qk.backtestCompare(compareIds),
    queryFn: async () => (await atlasApi.compareBacktests(compareIds)).data,
    enabled: compareIds.length >= 2,
  });
  const equityPoints =
    equityQuery.data?.map((row) => ({ time: row.datetime.slice(0, 10), value: row.equity })) ?? [];

  const metricEntries = useMemo(() => {
    if (!backtestQuery.data) {
      return [] as Array<{ key: string; value: string }>;
    }
    return Object.entries(backtestQuery.data.metrics_json ?? {}).slice(0, 8).map(([key, value]) => ({
      key,
      value: typeof value === "number" ? value.toFixed(4) : String(value),
    }));
  }, [backtestQuery.data]);

  const tradeRows = tradesQuery.data?.data ?? [];
  const tradeMeta = tradesQuery.data?.meta ?? {};
  const selectedTrade = tradeRows.find((row) => row.id === selectedTradeId) ?? null;
  const runRows = runsQuery.data?.data ?? [];
  const runMeta = runsQuery.data?.meta ?? {};
  const compareSeries =
    compareQuery.data?.map((row) => ({
      id: String(row.id),
      points: row.equity_curve.map((item) => ({
        time: item.datetime.slice(0, 10),
        value: item.equity,
      })),
    })) ?? [];

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Backtest Job" />

      <section className="card p-4">
        <h2 className="text-xl font-semibold">Strategy Lab</h2>
        <p className="mt-1 text-sm text-muted">Template-based strategy research with realistic backtesting defaults.</p>

        {templatesQuery.isLoading ? (
          <LoadingState label="Loading templates" />
        ) : templatesQuery.isError ? (
          <ErrorState
            title="Could not load templates"
            action="Check backend and retry."
            onRetry={() => void templatesQuery.refetch()}
          />
        ) : (
          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            {templateCards.map((template) => {
              const key = String(template.key);
              const selected = key === selectedTemplate;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => {
                    setSelectedTemplate(key);
                    setParams((template.default_params ?? {}) as Record<string, number>);
                  }}
                  className={`rounded-xl border p-3 text-left ${
                    selected ? "border-accent bg-accent/5" : "border-border"
                  }`}
                >
                  <p className="text-sm font-semibold">{String(template.name)}</p>
                  <p className="mt-1 text-xs text-muted">{String(template.description)}</p>
                </button>
              );
            })}
          </div>
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Backtest configuration</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-4">
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
          {Object.entries(params)
            .slice(0, 6)
            .map(([key, value]) => (
              <label key={key} className="text-sm text-muted">
                {key}
                <input
                  className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                  value={String(value)}
                  onChange={(event) =>
                    setParams((prev) => ({
                      ...prev,
                      [key]: parseNumeric(event.target.value),
                    }))
                  }
                />
              </label>
            ))}
        </div>
        <button
          type="button"
          onClick={() => runBacktestMutation.mutate()}
          className="focus-ring mt-4 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
          disabled={runBacktestMutation.isPending}
        >
          {runBacktestMutation.isPending ? "Queuing..." : "Run Backtest"}
        </button>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        {backtestQuery.isLoading && backtestId !== null ? (
          <LoadingState label="Loading backtest metrics" />
        ) : metricEntries.length === 0 ? (
          <EmptyState title="No results yet" action="Run a backtest to render metrics and trades." />
        ) : (
          metricEntries.map((entry) => (
            <MetricTile
              key={entry.key}
              label={entry.key}
              value={entry.value}
              helpKey={asMetricHelpKey(entry.key) ?? undefined}
            />
          ))
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Equity curve</h3>
        {backtestId ? (
          <div className="mb-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => window.open(atlasApi.backtestTradesExportUrl(backtestId), "_blank")}
              className="focus-ring rounded-lg border border-border px-3 py-1 text-xs"
            >
              Export trades CSV
            </button>
            <button
              type="button"
              onClick={() => window.open(atlasApi.backtestSummaryExportUrl(backtestId), "_blank")}
              className="focus-ring rounded-lg border border-border px-3 py-1 text-xs"
            >
              Export summary JSON
            </button>
          </div>
        ) : null}
        {equityQuery.isLoading && backtestId !== null ? (
          <LoadingState label="Loading equity series" />
        ) : equityPoints.length === 0 ? (
          <EmptyState title="No equity data" action="Run a backtest first." />
        ) : (
          <EquityChart points={equityPoints} />
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Trades</h3>
        {tradesQuery.isLoading && backtestId !== null ? (
          <LoadingState label="Loading trades" />
        ) : tradeRows.length === 0 ? (
          <EmptyState title="No trades" action="Try broader parameter ranges." />
        ) : (
          <>
            <div className="mt-3 overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-3 py-2">Symbol</th>
                    <th className="px-3 py-2">Entry</th>
                    <th className="px-3 py-2">Exit</th>
                    <th className="px-3 py-2">Qty</th>
                    <th className="px-3 py-2">PnL</th>
                    <th className="px-3 py-2">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {tradeRows.map((trade) => (
                    <tr key={trade.id} className="border-t border-border">
                      <td className="px-3 py-2">{trade.symbol}</td>
                      <td className="px-3 py-2">{trade.entry_dt}</td>
                      <td className="px-3 py-2">{trade.exit_dt}</td>
                      <td className="px-3 py-2">{trade.qty}</td>
                      <td className="px-3 py-2">{trade.pnl.toFixed(2)}</td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          onClick={() => setSelectedTradeId(trade.id)}
                        >
                          {trade.reason}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-muted">
              <span>
                Page {tradePage} / {Math.max(1, Math.ceil(Number(tradeMeta.total ?? 0) / Number(tradeMeta.page_size ?? 20)))}
              </span>
              <div className="space-x-2">
                <button
                  type="button"
                  className="rounded-lg border border-border px-3 py-1"
                  onClick={() => setTradePage((p) => Math.max(1, p - 1))}
                  disabled={tradePage <= 1}
                >
                  Prev
                </button>
                <button
                  type="button"
                  className="rounded-lg border border-border px-3 py-1"
                  onClick={() => setTradePage((p) => p + 1)}
                  disabled={!Boolean(tradeMeta.has_next)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Backtest leaderboard</h3>
        <p className="mt-1 text-sm text-muted">Filter, rank, and compare recent runs across templates and timeframes.</p>

        <div className="mt-3 grid gap-2 md:grid-cols-5">
          <input
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
            placeholder="Template filter"
            value={runsTemplateFilter}
            onChange={(event) => {
              setRunsPage(1);
              setRunsTemplateFilter(event.target.value);
            }}
          />
          <select
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
            value={runsTimeframeFilter}
            onChange={(event) => {
              setRunsPage(1);
              setRunsTimeframeFilter(event.target.value);
            }}
          >
            <option value="">All timeframes</option>
            <option value="1d">1d</option>
            <option value="4h_ish">4h_ish</option>
          </select>
          <select
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
            value={runsSortBy}
            onChange={(event) => setRunsSortBy(event.target.value)}
          >
            <option value="created_at">Sort: created</option>
            <option value="calmar">Sort: Calmar</option>
            <option value="max_drawdown">Sort: MaxDD</option>
            <option value="cvar_95">Sort: CVaR</option>
            <option value="turnover">Sort: Turnover</option>
          </select>
          <select
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
            value={runsSortDir}
            onChange={(event) => setRunsSortDir(event.target.value === "asc" ? "asc" : "desc")}
          >
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
          <button
            type="button"
            className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
            onClick={() => {
              setRunsTemplateFilter("");
              setRunsTimeframeFilter("");
              setRunsSortBy("created_at");
              setRunsSortDir("desc");
              setRunsPage(1);
            }}
          >
            Reset
          </button>
        </div>

        {runsQuery.isLoading ? (
          <LoadingState label="Loading leaderboard" />
        ) : runsQuery.isError ? (
          <ErrorState title="Could not load runs" action="Retry runs query." onRetry={() => void runsQuery.refetch()} />
        ) : runRows.length === 0 ? (
          <EmptyState title="No runs available" action="Run at least one backtest to populate the leaderboard." />
        ) : (
          <>
            <div className="mt-3 overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-surface text-left text-muted">
                  <tr>
                    <th className="px-3 py-2">Compare</th>
                    <th className="px-3 py-2">Strategy</th>
                    <th className="px-3 py-2">Timeframe</th>
                    <th className="px-3 py-2">Dates</th>
                    <th className="px-3 py-2">Calmar</th>
                    <th className="px-3 py-2">MaxDD</th>
                    <th className="px-3 py-2">CVaR</th>
                    <th className="px-3 py-2">Turnover</th>
                    <th className="px-3 py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {runRows.map((row) => {
                    const selected = compareIds.includes(row.id);
                    return (
                      <tr key={row.id} className="border-t border-border">
                        <td className="px-3 py-2">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={(event) => {
                              setCompareIds((prev) => {
                                if (event.target.checked) {
                                  if (prev.length >= 3) {
                                    toast.error("Compare supports up to 3 runs");
                                    return prev;
                                  }
                                  return [...prev, row.id];
                                }
                                return prev.filter((id) => id !== row.id);
                              });
                            }}
                          />
                        </td>
                        <td className="px-3 py-2">
                          <button
                            type="button"
                            onClick={() => {
                              setBacktestId(row.id);
                              setTradePage(1);
                            }}
                            className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          >
                            {row.strategy_template}
                          </button>
                        </td>
                        <td className="px-3 py-2">{row.timeframe}</td>
                        <td className="px-3 py-2">{row.start_date} {"->"} {row.end_date}</td>
                        <td className="px-3 py-2">{Number(row.metrics.calmar ?? 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{Number(row.metrics.max_drawdown ?? 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{Number(row.metrics.cvar_95 ?? 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{Number(row.metrics.turnover ?? 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{row.status}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-muted">
              <span>
                Page {runsPage} / {Math.max(1, Math.ceil(Number(runMeta.total ?? 0) / Number(runMeta.page_size ?? 12)))}
              </span>
              <div className="space-x-2">
                <button
                  type="button"
                  className="rounded-lg border border-border px-3 py-1"
                  onClick={() => setRunsPage((p) => Math.max(1, p - 1))}
                  disabled={runsPage <= 1}
                >
                  Prev
                </button>
                <button
                  type="button"
                  className="rounded-lg border border-border px-3 py-1"
                  onClick={() => setRunsPage((p) => p + 1)}
                  disabled={!Boolean(runMeta.has_next)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="card p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-base font-semibold">Compare runs</h3>
          <span className="text-xs text-muted">Select 2-3 runs from leaderboard</span>
        </div>
        {compareIds.length < 2 ? (
          <EmptyState title="Comparison inactive" action="Pick at least two runs to overlay equity curves." />
        ) : compareQuery.isLoading ? (
          <LoadingState label="Loading comparison" />
        ) : compareQuery.isError ? (
          <ErrorState title="Comparison failed" action="Retry compare query." onRetry={() => void compareQuery.refetch()} />
        ) : (
          <>
            <EquityChart series={compareSeries} />
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              {(compareQuery.data ?? []).map((item) => (
                <article key={item.id} className="rounded-xl border border-border p-3 text-sm">
                  <p className="font-semibold">{item.label}</p>
                  <p className="mt-1 text-muted">Calmar: {Number(item.metrics.calmar ?? 0).toFixed(3)}</p>
                  <p className="text-muted">MaxDD: {Number(item.metrics.max_drawdown ?? 0).toFixed(3)}</p>
                  <p className="text-muted">CVaR95: {Number(item.metrics.cvar_95 ?? 0).toFixed(3)}</p>
                  <p className="text-muted">Turnover: {Number(item.metrics.turnover ?? 0).toFixed(3)}</p>
                </article>
              ))}
            </div>
          </>
        )}
      </section>

      {(backtestQuery.isError || equityQuery.isError || tradesQuery.isError) && (
        <ErrorState
          title="Failed to load backtest results"
          action="Try rerunning the job."
          onRetry={() => {
            void backtestQuery.refetch();
            void equityQuery.refetch();
            void tradesQuery.refetch();
          }}
        />
      )}

      <DetailsDrawer open={Boolean(selectedTrade)} onClose={() => setSelectedTradeId(null)} title="Trade Details">
        {selectedTrade ? (
          <div className="space-y-2 text-sm">
            <p><span className="text-muted">Symbol:</span> {selectedTrade.symbol}</p>
            <p><span className="text-muted">Entry:</span> {selectedTrade.entry_dt}</p>
            <p><span className="text-muted">Exit:</span> {selectedTrade.exit_dt}</p>
            <p><span className="text-muted">Quantity:</span> {selectedTrade.qty}</p>
            <p><span className="text-muted">Entry price:</span> {selectedTrade.entry_px.toFixed(2)}</p>
            <p><span className="text-muted">Exit price:</span> {selectedTrade.exit_px.toFixed(2)}</p>
            <p><span className="text-muted">PnL:</span> {selectedTrade.pnl.toFixed(2)}</p>
            <p><span className="text-muted">R multiple:</span> {selectedTrade.r_multiple.toFixed(3)}</p>
            <p><span className="text-muted">Exit reason:</span> {selectedTrade.reason}</p>
            <p className="text-xs text-muted">
              Stop/target trail history is not yet persisted per trade in current schema.
            </p>
          </div>
        ) : null}
      </DetailsDrawer>
    </div>
  );
}
