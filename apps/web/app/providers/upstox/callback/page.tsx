"use client";

import { useEffect, useMemo, useState } from "react";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { atlasApi } from "@/src/lib/api/endpoints";

type ExchangeState = "idle" | "running" | "succeeded" | "failed";

function resolveRedirectUri(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return `${window.location.origin}/providers/upstox/callback`;
}

export default function UpstoxCallbackPage() {
  const params = useSearchParams();
  const [state, setState] = useState<ExchangeState>("idle");
  const [message, setMessage] = useState<string>("Waiting for authorization response...");

  const code = useMemo(() => String(params.get("code") ?? "").trim(), [params]);
  const oauthState = useMemo(() => String(params.get("state") ?? "").trim(), [params]);
  const authError = useMemo(() => String(params.get("error") ?? "").trim(), [params]);

  useEffect(() => {
    let cancelled = false;
    async function runExchange(): Promise<void> {
      if (authError) {
        setState("failed");
        setMessage(`Authorization failed: ${authError}`);
        return;
      }
      if (!code || !oauthState) {
        setState("failed");
        setMessage("Missing code/state in callback URL. Restart the connect flow.");
        return;
      }
      setState("running");
      setMessage("Exchanging authorization code with Upstox...");
      try {
        const redirectUri = resolveRedirectUri();
        const response = await atlasApi.upstoxTokenExchange({
          code,
          state: oauthState,
          redirect_uri: redirectUri || undefined,
          persist_token: false,
        });
        if (cancelled) {
          return;
        }
        setState("succeeded");
        const expiresAt = String(response.data.expires_at ?? "");
        setMessage(
          expiresAt
            ? `Connected. Token expires at ${expiresAt}.`
            : "Connected. Token is now stored in Atlas secure local credentials.",
        );
      } catch (error) {
        if (cancelled) {
          return;
        }
        const messageText =
          error instanceof Error ? error.message : "Token exchange failed. Please retry.";
        setState("failed");
        setMessage(messageText);
      }
    }

    if (state === "idle") {
      void runExchange();
    }
    return () => {
      cancelled = true;
    };
  }, [authError, code, oauthState, state]);

  const toneClass =
    state === "succeeded"
      ? "bg-success/10 text-success border-success/30"
      : state === "failed"
        ? "bg-danger/10 text-danger border-danger/30"
        : "bg-surface text-muted border-border";

  return (
    <div className="mx-auto max-w-2xl space-y-4 py-12">
      <section className="card p-6">
        <h1 className="text-2xl font-semibold">Connect Upstox</h1>
        <p className="mt-1 text-sm text-muted">
          Atlas is finalizing your Upstox connection.
        </p>
        <p className={`mt-4 rounded-xl border px-4 py-3 text-sm ${toneClass}`}>{message}</p>
        <div className="mt-5 flex gap-2">
          <Link
            href="/settings"
            className="focus-ring rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
          >
            Back to Settings
          </Link>
          <Link
            href="/ops"
            className="focus-ring rounded-xl border border-border px-4 py-2 text-sm text-muted"
          >
            Open Ops
          </Link>
        </div>
      </section>
    </div>
  );
}

