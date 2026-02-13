export function LoadingState({ label }: { label: string }) {
  return (
    <div className="card p-4" role="status" aria-live="polite">
      <p className="mb-3 text-sm text-muted">{label}</p>
      <div className="skeleton h-4 w-1/2 rounded-md" />
      <div className="mt-2 skeleton h-4 w-2/3 rounded-md" />
      <div className="mt-2 skeleton h-24 w-full rounded-xl" />
    </div>
  );
}

export function EmptyState({
  title,
  action,
  cta,
  onCta,
}: {
  title: string;
  action: string;
  cta?: string;
  onCta?: () => void;
}) {
  return (
    <div className="card rounded-2xl border-dashed p-8 text-center">
      <p className="text-base font-semibold">{title}</p>
      <p className="mt-1 text-sm text-muted">{action}</p>
      {cta && onCta ? (
        <button
          type="button"
          onClick={onCta}
          className="focus-ring mt-4 rounded-xl border border-border px-3 py-1.5 text-sm font-medium"
        >
          {cta}
        </button>
      ) : null}
    </div>
  );
}

export function ErrorState({
  title,
  action,
  code,
  onRetry,
}: {
  title: string;
  action: string;
  code?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="card border-warning/40 bg-warning/5 p-4">
      <p className="font-semibold text-warning">{title}</p>
      <p className="mt-1 text-sm text-muted">{action}</p>
      {code ? <p className="mt-1 text-xs text-muted">Error code: {code}</p> : null}
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="focus-ring mt-3 rounded-lg border border-border bg-panel px-3 py-1.5 text-xs font-semibold"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
