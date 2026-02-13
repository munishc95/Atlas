import type { DataEnvelope, ErrorEnvelope } from "@/src/lib/api/types";

export class ApiClientError extends Error {
  code: string;
  details?: unknown;
  status: number;

  constructor(opts: { code: string; message: string; details?: unknown; status: number }) {
    super(opts.message);
    this.code = opts.code;
    this.details = opts.details;
    this.status = opts.status;
    this.name = "ApiClientError";
  }
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!text) {
    return {} as T;
  }
  return JSON.parse(text) as T;
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<{ data: T; meta?: Record<string, unknown> }> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let envelope: ErrorEnvelope | null = null;
    try {
      envelope = await parseJson<ErrorEnvelope>(response);
    } catch {
      envelope = null;
    }

    throw new ApiClientError({
      code: envelope?.error?.code ?? "http_error",
      message: envelope?.error?.message ?? `HTTP ${response.status}`,
      details: envelope?.error?.details,
      status: response.status,
    });
  }

  const payload = await parseJson<DataEnvelope<T>>(response);
  return { data: payload.data, meta: payload.meta as Record<string, unknown> | undefined };
}

export function buildApiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
