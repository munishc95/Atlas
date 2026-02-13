"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { buildApiUrl } from "@/src/lib/api/client";

const TERMINAL = new Set(["SUCCEEDED", "FAILED", "DONE"]);

type ProgressPayload = {
  id: string;
  status: string;
  progress: number;
  started_at: string | null;
  ended_at: string | null;
  result?: Record<string, unknown> | null;
};

type StreamState = {
  status: string;
  progress: number;
  logs: string[];
  result: Record<string, unknown> | null;
  connected: boolean;
  error: string | null;
};

export function useJobStream(jobId: string | null) {
  const [state, setState] = useState<StreamState>({
    status: "IDLE",
    progress: 0,
    logs: [],
    result: null,
    connected: false,
    error: null,
  });
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusRef = useRef<string>("IDLE");

  useEffect(() => {
    if (!jobId) {
      statusRef.current = "IDLE";
      setState({
        status: "IDLE",
        progress: 0,
        logs: [],
        result: null,
        connected: false,
        error: null,
      });
      return;
    }

    let closed = false;
    let source: EventSource | null = null;
    statusRef.current = "QUEUED";

    const connect = () => {
      if (closed) {
        return;
      }

      source = new EventSource(buildApiUrl(`/api/jobs/${jobId}/stream`));
      setState((prev) => ({ ...prev, connected: true, error: null }));

      source.addEventListener("progress", (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as ProgressPayload;
          setState((prev) => ({
            ...prev,
            status: payload.status,
            progress: payload.progress,
            result: payload.result ?? prev.result,
            connected: true,
          }));
          statusRef.current = payload.status;

          if (TERMINAL.has(payload.status)) {
            source?.close();
          }
        } catch {
          setState((prev) => ({ ...prev, error: "Invalid progress event payload" }));
        }
      });

      source.addEventListener("log", (event) => {
        const line = (event as MessageEvent).data;
        setState((prev) => ({ ...prev, logs: [...prev.logs, line] }));
      });

      source.onerror = () => {
        setState((prev) => ({ ...prev, connected: false }));
        source?.close();

        // Reconnect only when we are not in terminal state.
        if (!closed && !TERMINAL.has(statusRef.current)) {
          reconnectTimer.current = setTimeout(connect, 1500);
        }
      };
    };

    connect();

    return () => {
      closed = true;
      source?.close();
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
    };
  }, [jobId]);

  const isTerminal = useMemo(() => TERMINAL.has(state.status), [state.status]);

  return {
    ...state,
    isTerminal,
  };
}
