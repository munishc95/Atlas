"use client";

import { useEffect } from "react";

type DetailsDrawerProps = {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
};

export function DetailsDrawer({ open, title, onClose, children }: DetailsDrawerProps) {
  useEffect(() => {
    if (!open) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/20 backdrop-blur-[1px]"
      role="presentation"
    >
      <button
        type="button"
        className="h-full flex-1 cursor-default"
        aria-label="Close details drawer"
        onClick={onClose}
      />
      <aside
        className="h-full w-[440px] max-w-[calc(100vw-1rem)] overflow-auto border-l border-border bg-panel p-4"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-semibold">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="focus-ring rounded-md border border-border px-2 py-1 text-xs text-muted"
          >
            Close
          </button>
        </div>
        {children}
      </aside>
    </div>
  );
}
