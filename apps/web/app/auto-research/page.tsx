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

type TabKey = "summary" | "candidates" | "policy";

export default function AutoResearchPage() {
  const forceInlineJobs = process.env.NEXT_PUBLIC_FORCE_INLINE_JOBS === "true";
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [candidatePage, setCandidatePage] = useState(1);
  const [candidateDrawerId, setCandidateDrawerId] = useState<number | null>(null);
  const [tab, setTab] = useState<TabKey>("summary");
  const [showPolicyModal, setShowPolicyModal] = useState(false);
  const [newPolicyName, setNewPolicyName] = useState("");
  const [createdPolicyId, setCreatedPolicyId] = useState<number | null>(null);
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [timeframes, setTimeframes] = useState<string[]>(["1d"]);
  const [templateFlags, setTemplateFlags] = useState<Record<string, boolean>>({
    trend_breakout: true,
    pullback_trend: true,
    squeeze_breakout: true,
  });
  const [trialsPerStrategy, setTrialsPerStrategy] = useState(30);
  const [maxSymbols, setMaxSymbols] = useState(5);
  const [maxEvaluations, setMaxEvaluations] = useState(0);
  const [maxDrawdown, setMaxDrawdown] = useState(0.2);
  const [minTrades, setMinTrades] = useState(20);
  const [stressPassRateThreshold, setStressPassRateThreshold] = useState(0.6);
  const [sampler, setSampler] = useState("tpe");
  const [pruner, setPruner] = useState("median");
  const [seed, setSeed] = useState(17);

  const dataStatusQuery = useQuery({
    queryKey: qk.dataStatus,
    queryFn: async () => (await atlasApi.dataStatus()).data,
  });

  useEffect(() => {
    if (datasetId !== null) {
      return;
    }
    const firstId = Number((dataStatusQuery.data ?? [])[0]?.id);
    if (Number.isFinite(firstId) && firstId > 0) {
      setDatasetId(firstId);
    }
  }, [dataStatusQuery.data, datasetId]);

  const runsQuery = useQuery({
    queryKey: qk.researchRuns(1, 20),
    queryFn: async () => await atlasApi.researchRuns(1, 20),
    refetchInterval: 8_000,
  });

  useEffect(() => {
    const first = runsQuery.data?.data?.[0];
    if (!runId && first?.id) {
      setRunId(first.id);
    }
  }, [runId, runsQuery.data]);

  const runQuery = useQuery({
    queryKey: qk.researchRun(runId),
    queryFn: async () => (await atlasApi.researchRunById(runId as number)).data,
    enabled: runId !== null,
  });

  const candidatesQuery = useQuery({
    queryKey: qk.researchCandidates(runId, candidatePage, 20),
    queryFn: async () => await atlasApi.researchCandidates(runId as number, candidatePage, 20),
    enabled: runId !== null,
  });

  const policiesQuery = useQuery({
    queryKey: qk.policies(1, 50),
    queryFn: async () => (await atlasApi.policies(1, 50)).data,
  });

  const runResearchMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runResearch({
          dataset_id: datasetId ?? undefined,
          timeframes,
          strategy_templates: Object.entries(templateFlags)
            .filter(([, enabled]) => enabled)
            .map(([key]) => key),
          symbol_scope: "liquid",
          config: {
            trials_per_strategy: trialsPerStrategy,
            max_symbols: maxSymbols,
            max_evaluations: maxEvaluations,
            objective: "oos_robustness",
            max_drawdown: maxDrawdown,
            min_trades: minTrades,
            stress_pass_rate_threshold: stressPassRateThreshold,
            sampler,
            pruner,
            seed,
            force_inline: forceInlineJobs,
          },
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast.success("Auto Research queued");
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to queue Auto Research");
    },
  });

  const stream = useJobStream(activeJobId);

  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      const nextRunId = Number((stream.result as { run_id?: number } | null)?.run_id);
      if (Number.isFinite(nextRunId) && nextRunId > 0) {
        setRunId(nextRunId);
        setCandidatePage(1);
        setTab("summary");
      }
      queryClient.invalidateQueries({ queryKey: qk.researchRuns(1, 20) });
      queryClient.invalidateQueries({ queryKey: qk.researchRun(runId) });
      queryClient.invalidateQueries({ queryKey: qk.researchCandidates(runId, candidatePage, 20) });
      toast.success("Auto Research complete");
    }
    if (stream.status === "FAILED") {
      toast.error("Auto Research failed");
    }
  }, [candidatePage, queryClient, runId, stream.isTerminal, stream.result, stream.status]);

  const createPolicyMutation = useMutation({
    mutationFn: async (name: string) => {
      if (!runId) {
        throw new Error("Select a research run first");
      }
      return (await atlasApi.createPolicy({ research_run_id: runId, name })).data;
    },
    onSuccess: (policy) => {
      setCreatedPolicyId(policy.id);
      setShowPolicyModal(false);
      queryClient.invalidateQueries({ queryKey: qk.policies(1, 50) });
      toast.success("Policy created");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not create policy");
    },
  });

  const promotePolicyMutation = useMutation({
    mutationFn: async (policyId: number) => (await atlasApi.promotePolicyToPaper(policyId)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: qk.settings });
      queryClient.invalidateQueries({ queryKey: qk.paperState });
      toast.success("Policy promoted to paper mode");
      window.location.href = "/paper-trading";
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not promote policy to paper");
    },
  });

  const selectedRun = runQuery.data;
  const selectedRunSummary = (selectedRun?.summary_json ?? {}) as Record<string, unknown>;
  const runRows = runsQuery.data?.data ?? [];
  const candidateRows = candidatesQuery.data?.data ?? [];
  const candidateMeta = candidatesQuery.data?.meta ?? {};
  const selectedCandidate =
    candidateRows.find((candidate) => candidate.id === candidateDrawerId) ?? null;
  const policyPreview = (selectedRunSummary.policy_preview ?? {}) as Record<string, unknown>;
  const activePolicy = useMemo(() => {
    if (createdPolicyId) {
      return (policiesQuery.data ?? []).find((policy) => policy.id === createdPolicyId) ?? null;
    }
    return (policiesQuery.data ?? [])[0] ?? null;
  }, [createdPolicyId, policiesQuery.data]);

  const evaluationsDoneText =
    stream.status === "IDLE"
      ? null
      : `Research progress: ${stream.progress}% (${stream.logs.length} log updates)`;

  return (
    <div className="space-y-5">
      <JobDrawer
        jobId={activeJobId}
        onClose={() => setActiveJobId(null)}
        title="Auto Research Job"
      />

      {showPolicyModal ? (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/30 p-4"
          role="dialog"
          aria-label="Create policy"
        >
          <div className="card w-full max-w-md p-4">
            <h3 className="text-base font-semibold">Create Policy</h3>
            <p className="mt-1 text-sm text-muted">
              Name the policy generated from this research run.
            </p>
            <input
              className="focus-ring mt-3 w-full rounded-xl border border-border px-3 py-2 text-sm"
              value={newPolicyName}
              onChange={(event) => setNewPolicyName(event.target.value)}
              placeholder="Atlas Policy - 2026-02-13"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
                onClick={() => setShowPolicyModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="focus-ring rounded-xl bg-accent px-3 py-2 text-sm font-semibold text-white"
                onClick={() => createPolicyMutation.mutate(newPolicyName.trim())}
                disabled={!newPolicyName.trim() || createPolicyMutation.isPending}
              >
                {createPolicyMutation.isPending ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-[360px,1fr]">
        <article className="card p-4">
          <h2 className="text-xl font-semibold">Auto Research</h2>
          <p className="mt-1 text-sm text-muted">
            Systematic multi-template walk-forward scan with stress-aware candidate ranking.
          </p>

          <div className="mt-4 space-y-3">
            <label className="block text-sm text-muted">
              Universe dataset
              <select
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={datasetId ?? ""}
                onChange={(event) =>
                  setDatasetId(event.target.value ? Number(event.target.value) : null)
                }
              >
                {(dataStatusQuery.data ?? []).map((row) => (
                  <option key={String(row.id)} value={String(row.id)}>
                    {String(row.symbol)} / {String(row.timeframe)} ({String(row.start_date)} {"->"}{" "}
                    {String(row.end_date)})
                  </option>
                ))}
              </select>
            </label>

            <label className="block text-sm text-muted">
              Timeframes
              <div className="mt-2 flex flex-wrap gap-2">
                {["1d", "4h_ish"].map((value) => {
                  const enabled = timeframes.includes(value);
                  return (
                    <button
                      key={value}
                      type="button"
                      onClick={() =>
                        setTimeframes((prev) => {
                          if (enabled) {
                            if (prev.length === 1) {
                              return prev;
                            }
                            return prev.filter((item) => item !== value);
                          }
                          return [...prev, value];
                        })
                      }
                      className={`focus-ring rounded-lg border px-3 py-1 text-xs ${
                        enabled
                          ? "border-accent bg-accent/10 text-accent"
                          : "border-border text-muted"
                      }`}
                    >
                      {value}
                    </button>
                  );
                })}
              </div>
            </label>

            <label className="block text-sm text-muted">
              Strategy templates
              <div className="mt-2 space-y-1">
                {Object.keys(templateFlags).map((template) => (
                  <label key={template} className="flex items-center gap-2 text-sm text-ink">
                    <input
                      type="checkbox"
                      checked={Boolean(templateFlags[template])}
                      onChange={(event) =>
                        setTemplateFlags((prev) => ({
                          ...prev,
                          [template]: event.target.checked,
                        }))
                      }
                    />
                    <span>{template}</span>
                  </label>
                ))}
              </div>
            </label>

            <label className="block text-sm text-muted">
              Trials per strategy
              <input
                type="number"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={trialsPerStrategy}
                onChange={(event) => setTrialsPerStrategy(Number(event.target.value))}
              />
            </label>
            <label className="block text-sm text-muted">
              Max symbols sampled
              <input
                type="number"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={maxSymbols}
                onChange={(event) => setMaxSymbols(Number(event.target.value))}
              />
            </label>
            <label className="block text-sm text-muted">
              Max evaluations (0 = no cap)
              <input
                type="number"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={maxEvaluations}
                onChange={(event) => setMaxEvaluations(Number(event.target.value))}
              />
            </label>
            <label className="block text-sm text-muted">
              Max OOS drawdown threshold
              <input
                type="number"
                step="0.01"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={maxDrawdown}
                onChange={(event) => setMaxDrawdown(Number(event.target.value))}
              />
            </label>
            <label className="block text-sm text-muted">
              Minimum average trades
              <input
                type="number"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={minTrades}
                onChange={(event) => setMinTrades(Number(event.target.value))}
              />
            </label>
            <label className="block text-sm text-muted">
              Stress pass threshold
              <input
                type="number"
                step="0.05"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={stressPassRateThreshold}
                onChange={(event) => setStressPassRateThreshold(Number(event.target.value))}
              />
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block text-sm text-muted">
                Sampler
                <select
                  className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                  value={sampler}
                  onChange={(event) => setSampler(event.target.value)}
                >
                  <option value="tpe">tpe</option>
                  <option value="random">random</option>
                  <option value="cmaes">cmaes</option>
                </select>
              </label>
              <label className="block text-sm text-muted">
                Pruner
                <select
                  className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                  value={pruner}
                  onChange={(event) => setPruner(event.target.value)}
                >
                  <option value="median">median</option>
                  <option value="halving">halving</option>
                  <option value="none">none</option>
                </select>
              </label>
            </div>
            <label className="block text-sm text-muted">
              Deterministic seed
              <input
                type="number"
                className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
          </div>

          <button
            type="button"
            className="focus-ring mt-4 w-full rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
            onClick={() => runResearchMutation.mutate()}
            disabled={runResearchMutation.isPending}
          >
            {runResearchMutation.isPending ? "Queuing..." : "Run Auto Research"}
          </button>
        </article>

        <article className="card p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-semibold">Results</h3>
              <p className="text-sm text-muted">
                Run summary, ranked candidates, and policy preview.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                className={`focus-ring rounded-lg border px-3 py-1 text-xs ${
                  tab === "summary" ? "border-accent text-accent" : "border-border text-muted"
                }`}
                onClick={() => setTab("summary")}
              >
                Summary
              </button>
              <button
                type="button"
                className={`focus-ring rounded-lg border px-3 py-1 text-xs ${
                  tab === "candidates" ? "border-accent text-accent" : "border-border text-muted"
                }`}
                onClick={() => setTab("candidates")}
              >
                Candidates
              </button>
              <button
                type="button"
                className={`focus-ring rounded-lg border px-3 py-1 text-xs ${
                  tab === "policy" ? "border-accent text-accent" : "border-border text-muted"
                }`}
                onClick={() => setTab("policy")}
              >
                Policy
              </button>
            </div>
          </div>

          {evaluationsDoneText ? (
            <p className="mt-3 rounded-xl border border-border px-3 py-2 text-sm text-muted">
              {evaluationsDoneText}
            </p>
          ) : null}

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {runRows.map((row) => (
              <button
                key={row.id}
                type="button"
                onClick={() => {
                  setRunId(row.id);
                  setCandidatePage(1);
                }}
                className={`focus-ring rounded-lg border px-3 py-1 text-xs ${
                  runId === row.id
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-border text-muted"
                }`}
              >
                Run #{row.id} ({row.status})
              </button>
            ))}
          </div>

          {runQuery.isLoading && runId !== null ? (
            <LoadingState label="Loading research run" />
          ) : runQuery.isError ? (
            <ErrorState
              title="Could not load research run"
              action="Retry the selected run."
              onRetry={() => void runQuery.refetch()}
            />
          ) : !selectedRun ? (
            <EmptyState
              title="No research run yet"
              action="Run Auto Research to generate candidates."
            />
          ) : (
            <>
              {tab === "summary" ? (
                <div className="mt-4 grid gap-3 md:grid-cols-3">
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Status: {selectedRun.status}
                  </p>
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Candidates: {String(selectedRunSummary.candidate_count ?? 0)}
                  </p>
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Accepted: {String(selectedRunSummary.accepted_count ?? 0)}
                  </p>
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Evaluations: {String(selectedRunSummary.evaluations ?? 0)}
                  </p>
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Symbols:{" "}
                    {Array.isArray(selectedRunSummary.symbols)
                      ? selectedRunSummary.symbols.length
                      : 0}
                  </p>
                  <p className="rounded-xl border border-border px-3 py-2 text-sm">
                    Templates:{" "}
                    {Array.isArray(selectedRunSummary.strategy_templates)
                      ? selectedRunSummary.strategy_templates.length
                      : 0}
                  </p>
                </div>
              ) : null}

              {tab === "candidates" ? (
                candidatesQuery.isLoading ? (
                  <LoadingState label="Loading candidates" />
                ) : candidatesQuery.isError ? (
                  <ErrorState
                    title="Could not load candidates"
                    action="Retry candidates query."
                    onRetry={() => void candidatesQuery.refetch()}
                  />
                ) : candidateRows.length === 0 ? (
                  <EmptyState title="No candidates available" action="Run Auto Research first." />
                ) : (
                  <>
                    <div className="mt-4 overflow-hidden rounded-xl border border-border">
                      <table className="w-full text-sm">
                        <thead className="bg-surface text-left text-muted">
                          <tr>
                            <th className="px-3 py-2">Rank</th>
                            <th className="px-3 py-2">Symbol</th>
                            <th className="px-3 py-2">Template</th>
                            <th className="px-3 py-2">Score</th>
                            <th className="px-3 py-2">Stress pass</th>
                            <th className="px-3 py-2">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {candidateRows.map((row) => (
                            <tr key={row.id} className="border-t border-border">
                              <td className="px-3 py-2">{row.rank}</td>
                              <td className="px-3 py-2">{row.symbol}</td>
                              <td className="px-3 py-2">{row.strategy_key}</td>
                              <td className="px-3 py-2">{row.score.toFixed(4)}</td>
                              <td className="px-3 py-2">
                                {(row.stress_pass_rate * 100).toFixed(1)}%
                              </td>
                              <td className="px-3 py-2">
                                <button
                                  type="button"
                                  className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                                  onClick={() => setCandidateDrawerId(row.id)}
                                >
                                  {row.accepted ? "Accepted" : "Rejected"}
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="mt-3 flex items-center justify-between text-sm text-muted">
                      <span>
                        Page {candidatePage} /{" "}
                        {Math.max(
                          1,
                          Math.ceil(
                            Number(candidateMeta.total ?? 0) /
                              Number(candidateMeta.page_size ?? 20),
                          ),
                        )}
                      </span>
                      <div className="space-x-2">
                        <button
                          type="button"
                          className="rounded-lg border border-border px-3 py-1"
                          onClick={() => setCandidatePage((page) => Math.max(1, page - 1))}
                          disabled={candidatePage <= 1}
                        >
                          Prev
                        </button>
                        <button
                          type="button"
                          className="rounded-lg border border-border px-3 py-1"
                          onClick={() => setCandidatePage((page) => page + 1)}
                          disabled={!Boolean(candidateMeta.has_next)}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  </>
                )
              ) : null}

              {tab === "policy" ? (
                <div className="mt-4 space-y-3">
                  {Object.keys(policyPreview).length === 0 ? (
                    <EmptyState
                      title="Policy preview unavailable"
                      action="Complete an Auto Research run to build a regime policy preview."
                    />
                  ) : (
                    <pre className="max-h-80 overflow-auto rounded-xl border border-border bg-surface p-3 text-xs text-muted">
                      {JSON.stringify(policyPreview, null, 2)}
                    </pre>
                  )}

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      className="focus-ring rounded-xl bg-accent px-3 py-2 text-sm font-semibold text-white"
                      onClick={() => {
                        const stamp = new Date().toISOString().slice(0, 10);
                        setNewPolicyName(`Atlas Policy ${stamp}`);
                        setShowPolicyModal(true);
                      }}
                      disabled={!runId}
                    >
                      Create Policy
                    </button>
                    <button
                      type="button"
                      className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
                      onClick={() => activePolicy && promotePolicyMutation.mutate(activePolicy.id)}
                      disabled={!activePolicy || promotePolicyMutation.isPending}
                    >
                      {promotePolicyMutation.isPending ? "Promoting..." : "Use in Paper"}
                    </button>
                  </div>
                  {activePolicy ? (
                    <p className="text-xs text-muted">
                      Selected policy: {activePolicy.name} (id {activePolicy.id})
                    </p>
                  ) : (
                    <p className="text-xs text-muted">No policy has been created yet.</p>
                  )}
                </div>
              ) : null}
            </>
          )}
        </article>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Dataset readiness</h3>
        {dataStatusQuery.isLoading ? (
          <LoadingState label="Loading dataset status" />
        ) : dataStatusQuery.isError ? (
          <ErrorState
            title="Could not load datasets"
            action="Retry data status query."
            onRetry={() => void dataStatusQuery.refetch()}
          />
        ) : (dataStatusQuery.data ?? []).length === 0 ? (
          <EmptyState
            title="No imported data"
            action="Import sample CSV from Universe & Data first."
          />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">Timeframe</th>
                  <th className="px-3 py-2">Range</th>
                  <th className="px-3 py-2">Updated</th>
                </tr>
              </thead>
              <tbody>
                {(dataStatusQuery.data ?? []).slice(0, 8).map((row, index) => (
                  <tr key={`${String(row.symbol)}-${index}`} className="border-t border-border">
                    <td className="px-3 py-2">{String(row.symbol ?? "-")}</td>
                    <td className="px-3 py-2">{String(row.timeframe ?? "-")}</td>
                    <td className="px-3 py-2">
                      {String(row.start_date ?? "-")} {"->"} {String(row.end_date ?? "-")}
                    </td>
                    <td className="px-3 py-2">{String(row.last_updated ?? "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <DetailsDrawer
        open={Boolean(selectedCandidate)}
        onClose={() => setCandidateDrawerId(null)}
        title={`Candidate #${selectedCandidate?.rank ?? ""}`}
      >
        {selectedCandidate ? (
          <div className="space-y-2 text-sm">
            <p>
              <span className="text-muted">Symbol:</span> {selectedCandidate.symbol}
            </p>
            <p>
              <span className="text-muted">Timeframe:</span> {selectedCandidate.timeframe}
            </p>
            <p>
              <span className="text-muted">Strategy:</span> {selectedCandidate.strategy_key}
            </p>
            <p>
              <span className="text-muted">Score:</span> {selectedCandidate.score.toFixed(4)}
            </p>
            <p>
              <span className="text-muted">Stress pass:</span>{" "}
              {(selectedCandidate.stress_pass_rate * 100).toFixed(1)}%
            </p>
            <p>
              <span className="text-muted">Dispersion:</span>{" "}
              {selectedCandidate.param_dispersion.toFixed(4)}
            </p>
            <p>
              <span className="text-muted">Fold variance:</span>{" "}
              {selectedCandidate.fold_variance.toFixed(4)}
            </p>
            <p className="text-sm font-semibold">Best params</p>
            <pre className="max-h-48 overflow-auto rounded-xl border border-border bg-surface p-2 text-xs">
              {JSON.stringify(selectedCandidate.best_params_json, null, 2)}
            </pre>
            <p className="text-sm font-semibold">Explanations</p>
            <ul className="space-y-1 text-xs text-muted">
              {selectedCandidate.explanations_json.map((line, index) => (
                <li key={`${line}-${index}`}>{line}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </DetailsDrawer>
    </div>
  );
}
