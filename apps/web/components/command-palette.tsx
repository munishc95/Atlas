"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type CommandItem = {
  id: string;
  label: string;
  subtitle?: string;
  keywords?: string[];
};

type CommandPaletteProps = {
  open: boolean;
  commands: CommandItem[];
  onClose: () => void;
  onSelect: (id: string) => void;
};

export function CommandPalette({ open, commands, onClose, onSelect }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setQuery("");
    const id = window.setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
    return () => window.clearTimeout(id);
  }, [open]);

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

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      return commands;
    }
    return commands.filter((item) => {
      const haystack = [item.label, item.subtitle ?? "", ...(item.keywords ?? [])].join(" ").toLowerCase();
      return haystack.includes(q);
    });
  }, [commands, query]);

  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/20 backdrop-blur-[1px]" role="presentation">
      <div className="mx-auto mt-[8vh] w-[680px] max-w-[calc(100vw-1rem)]">
        <div className="card overflow-hidden">
          <div className="border-b border-border p-3">
            <input
              ref={inputRef}
              className="focus-ring w-full rounded-xl border border-border bg-panel px-3 py-2 text-sm"
              placeholder="Search commands..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="Search commands"
            />
          </div>
          <div className="max-h-[55vh] overflow-auto p-2">
            {filtered.length === 0 ? (
              <p className="rounded-lg px-3 py-2 text-sm text-muted">No command found for this search.</p>
            ) : (
              <ul className="space-y-1">
                {filtered.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      onClick={() => onSelect(item.id)}
                      className="focus-ring w-full rounded-xl px-3 py-2 text-left hover:bg-surface"
                    >
                      <p className="text-sm font-medium">{item.label}</p>
                      {item.subtitle ? <p className="text-xs text-muted">{item.subtitle}</p> : null}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
      <button type="button" className="absolute inset-0 -z-10" onClick={onClose} aria-label="Close command palette" />
    </div>
  );
}
