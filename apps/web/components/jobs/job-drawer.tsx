"use client";

import { useEffect, useMemo } from "react";

import { useJobStream } from "@/src/hooks/useJobStream";

type JobDrawerProps = {
  jobId: string | null;
  title?: string;
  onClose: () => void;
};

function titleCase(status: string): string {
  return status.toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

export function JobDrawer({ jobId, title = "Job Progress", onClose }: JobDrawerProps) {
  const stream = useJobStream(jobId);

  useEffect(() => {
    if (!jobId) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [jobId, onClose]);

  const badgeClass = useMemo(() => {
    if (stream.status === "SUCCEEDED") return "bg-success/15 text-success";
    if (stream.status === "FAILED") return "bg-warning/20 text-warning";
    if (stream.status === "RUNNING") return "bg-accent/15 text-accent";
    return "bg-surface text-muted";
  }, [stream.status]);

  if (!jobId) {
    return null;
  }

  return (
    <aside className="fixed right-3 top-3 z-30 w-[360px] max-w-[calc(100vw-1.5rem)]" role="dialog" aria-label={title}>
      <div className="card p-4">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">{title}</p>
            <p className="text-xs text-muted">Job ID: {jobId}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="focus-ring rounded-md border border-border px-2 py-1 text-xs text-muted"
          >
            Close
          </button>
        </div>

        <div className="mb-2 flex items-center justify-between">
          <span className={`badge ${badgeClass}`}>{titleCase(stream.status)}</span>
          <span className="text-xs text-muted">{stream.progress}%</span>
        </div>

        <div className="h-2 rounded-full bg-surface">
          <div
            className="h-2 rounded-full bg-accent transition-all"
            style={{ width: `${Math.max(4, Math.min(100, stream.progress))}%` }}
          />
        </div>

        <div className="mt-3 max-h-44 overflow-auto rounded-xl border border-border p-2 text-xs text-muted">
          {stream.logs.length === 0 ? (
            <p>No logs yet.</p>
          ) : (
            stream.logs.map((line, index) => <p key={`${line}-${index}`}>{line}</p>)
          )}
        </div>
      </div>
    </aside>
  );
}
