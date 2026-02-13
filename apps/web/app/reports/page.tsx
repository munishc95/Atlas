"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { DetailsDrawer } from "@/components/details-drawer";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useJobStream } from "@/src/hooks/useJobStream";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

export default function ReportsPage() {
  const queryClient = useQueryClient();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [dateFilter, setDateFilter] = useState("");
  const [monthFilter, setMonthFilter] = useState("");

  const reportsQuery = useQuery({
    queryKey: qk.dailyReports(dateFilter || undefined, null, null),
    queryFn: async () =>
      (
        await atlasApi.dailyReports({
          date: dateFilter || undefined,
        })
      ).data,
  });
  const monthlyReportsQuery = useQuery({
    queryKey: qk.monthlyReports(monthFilter || undefined, null, null),
    queryFn: async () =>
      (
        await atlasApi.monthlyReports({
          month: monthFilter || undefined,
        })
      ).data,
  });

  const generateMutation = useMutation({
    mutationFn: async () => (await atlasApi.generateDailyReport({ date: dateFilter || undefined })).data,
    onSuccess: (payload) => {
      setActiveJobId(payload.job_id);
      toast.success("Daily report job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue report generation");
    },
  });
  const generateMonthlyMutation = useMutation({
    mutationFn: async () =>
      (await atlasApi.generateMonthlyReport({ month: monthFilter || undefined })).data,
    onSuccess: (payload) => {
      setActiveJobId(payload.job_id);
      toast.success("Monthly report job queued");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue monthly report generation");
    },
  });

  const stream = useJobStream(activeJobId);
  useEffect(() => {
    if (!stream.isTerminal) {
      return;
    }
    if (stream.status === "SUCCEEDED" || stream.status === "DONE") {
      toast.success("Report generated");
      queryClient.invalidateQueries({
        queryKey: qk.dailyReports(dateFilter || undefined, null, null),
      });
      queryClient.invalidateQueries({
        queryKey: qk.monthlyReports(monthFilter || undefined, null, null),
      });
    } else if (stream.status === "FAILED") {
      toast.error("Report generation failed");
    }
  }, [dateFilter, monthFilter, queryClient, stream.isTerminal, stream.status]);

  const reports = reportsQuery.data ?? [];
  const selectedReport =
    selectedReportId === null
      ? null
      : (reports.find((item) => item.id === selectedReportId) ?? null);

  const reportSummary = useMemo(() => {
    if (!selectedReport) {
      return null;
    }
    const content = selectedReport.content_json ?? {};
    const summary = (content.summary ?? {}) as Record<string, unknown>;
    const explainability = (content.explainability ?? {}) as Record<string, unknown>;
    const risk = (content.risk ?? {}) as Record<string, unknown>;
    return { summary, explainability, risk };
  }, [selectedReport]);

  return (
    <div className="space-y-5">
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Report Job" />

      <section className="card p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold">Reports</h2>
            <p className="mt-1 text-sm text-muted">
              Daily operating reports for paper runs, explainability, and risk snapshots.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="date"
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
              value={dateFilter}
              onChange={(event) => setDateFilter(event.target.value)}
            />
            <input
              type="month"
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm"
              value={monthFilter}
              onChange={(event) => setMonthFilter(event.target.value)}
            />
            <button
              type="button"
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
              onClick={() => generateMutation.mutate()}
              disabled={generateMutation.isPending}
            >
              {generateMutation.isPending ? "Queuing..." : "Generate Daily Report"}
            </button>
            <button
              type="button"
              className="focus-ring rounded-xl border border-border px-3 py-2 text-sm text-muted"
              onClick={() => generateMonthlyMutation.mutate()}
              disabled={generateMonthlyMutation.isPending}
            >
              {generateMonthlyMutation.isPending ? "Queuing..." : "Generate Monthly Report"}
            </button>
          </div>
        </div>
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Daily reports</h3>
        {reportsQuery.isLoading ? (
          <LoadingState label="Loading daily reports" />
        ) : reportsQuery.isError ? (
          <ErrorState
            title="Could not load daily reports"
            action="Retry after API connectivity is restored."
            onRetry={() => void reportsQuery.refetch()}
          />
        ) : reports.length === 0 ? (
          <EmptyState
            title="No reports available"
            action="Generate a daily report after running Paper Trading."
          />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Bundle</th>
                  <th className="px-3 py-2">Policy</th>
                  <th className="px-3 py-2">Net PnL</th>
                  <th className="px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => {
                  const summary = (report.content_json?.summary ?? {}) as Record<string, unknown>;
                  return (
                    <tr key={report.id} className="border-t border-border">
                      <td className="px-3 py-2">{report.date}</td>
                      <td className="px-3 py-2">{report.bundle_id ?? "-"}</td>
                      <td className="px-3 py-2">{report.policy_id ?? "-"}</td>
                      <td className="px-3 py-2">{String(summary.net_pnl ?? "-")}</td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          onClick={() => setSelectedReportId(report.id)}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card p-4">
        <h3 className="text-base font-semibold">Monthly reports</h3>
        {monthlyReportsQuery.isLoading ? (
          <LoadingState label="Loading monthly reports" />
        ) : monthlyReportsQuery.isError ? (
          <ErrorState
            title="Could not load monthly reports"
            action="Retry after API connectivity is restored."
            onRetry={() => void monthlyReportsQuery.refetch()}
          />
        ) : (monthlyReportsQuery.data ?? []).length === 0 ? (
          <EmptyState
            title="No monthly reports available"
            action="Generate a monthly report after at least one day of paper runs."
          />
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead className="bg-surface text-left text-muted">
                <tr>
                  <th className="px-3 py-2">Month</th>
                  <th className="px-3 py-2">Bundle</th>
                  <th className="px-3 py-2">Policy</th>
                  <th className="px-3 py-2">Net PnL</th>
                  <th className="px-3 py-2">Exports</th>
                </tr>
              </thead>
              <tbody>
                {(monthlyReportsQuery.data ?? []).map((report) => {
                  const summary = (report.content_json?.summary ?? {}) as Record<string, unknown>;
                  return (
                    <tr key={report.id} className="border-t border-border">
                      <td className="px-3 py-2">{report.month}</td>
                      <td className="px-3 py-2">{report.bundle_id ?? "-"}</td>
                      <td className="px-3 py-2">{report.policy_id ?? "-"}</td>
                      <td className="px-3 py-2">{String(summary.net_pnl ?? "-")}</td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-2">
                          <a
                            href={atlasApi.monthlyReportExportJsonUrl(report.id)}
                            className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          >
                            JSON
                          </a>
                          <a
                            href={atlasApi.monthlyReportExportPdfUrl(report.id)}
                            className="focus-ring rounded-md border border-border px-2 py-1 text-xs"
                          >
                            PDF
                          </a>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <DetailsDrawer
        open={Boolean(selectedReport)}
        onClose={() => setSelectedReportId(null)}
        title={`Daily Report ${selectedReport?.date ?? ""}`}
      >
        {!selectedReport || !reportSummary ? (
          <EmptyState title="No report selected" action="Select a report row to inspect details." />
        ) : (
          <div className="space-y-3 text-sm">
            <p>
              <span className="text-muted">Bundle:</span> {selectedReport.bundle_id ?? "-"}
            </p>
            <p>
              <span className="text-muted">Policy:</span> {selectedReport.policy_id ?? "-"}
            </p>
            <p>
              <span className="text-muted">Runs:</span> {String(reportSummary.summary.runs ?? "-")}
            </p>
            <p>
              <span className="text-muted">Net PnL:</span> {String(reportSummary.summary.net_pnl ?? "-")}
            </p>
            <p>
              <span className="text-muted">Costs:</span> {String(reportSummary.summary.costs ?? "-")}
            </p>
            <p>
              <span className="text-muted">Drawdown:</span> {String(reportSummary.summary.drawdown ?? "-")}
            </p>
            <p>
              <span className="text-muted">Mode:</span> {String(reportSummary.summary.mode ?? "LIVE")}
            </p>
            {reportSummary.summary.shadow_note ? (
              <p className="rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
                {String(reportSummary.summary.shadow_note)}
              </p>
            ) : null}
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Selected reasons
              </p>
              <pre className="overflow-auto text-xs text-muted">
                {JSON.stringify(reportSummary.explainability.selected_reason_histogram ?? {}, null, 2)}
              </pre>
            </div>
            <div className="rounded-xl border border-border p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
                Skipped reasons
              </p>
              <pre className="overflow-auto text-xs text-muted">
                {JSON.stringify(reportSummary.explainability.skipped_reason_histogram ?? {}, null, 2)}
              </pre>
            </div>
            <div className="flex flex-wrap gap-2">
              <a
                href={atlasApi.dailyReportExportCsvUrl(selectedReport.id)}
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
              >
                Export CSV
              </a>
              <a
                href={atlasApi.dailyReportExportJsonUrl(selectedReport.id)}
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
              >
                Export JSON
              </a>
              <a
                href={atlasApi.dailyReportExportPdfUrl(selectedReport.id)}
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs text-muted"
              >
                Export PDF
              </a>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  );
}
