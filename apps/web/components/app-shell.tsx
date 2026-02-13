"use client";

import { useEffect, useMemo, useState } from "react";

import Link from "next/link";
import type { Route } from "next";
import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { CommandPalette, type CommandItem } from "@/components/command-palette";
import { JobDrawer } from "@/components/jobs/job-drawer";
import { navItems, disclaimer } from "@/lib/constants";
import { ThemeToggle } from "@/components/theme-toggle";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

export function AppShell({ children }: { children: React.ReactNode }) {
  const forceInlineJobs = process.env.NEXT_PUBLIC_FORCE_INLINE_JOBS === "true";
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const jobsQuery = useQuery({
    queryKey: qk.jobs(20),
    queryFn: async () => (await atlasApi.jobs(20)).data,
    refetchInterval: 4_000,
  });

  const demoBacktestMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runBacktest({
          symbol: "NIFTY500",
          timeframe: "1d",
          strategy_template: "trend_breakout",
          params: {
            trend_period: 200,
            breakout_lookback: 20,
            atr_stop_mult: 2.0,
            atr_trail_mult: 2.0,
          },
          config: {},
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Demo backtest queued");
      router.push("/strategy-lab");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue demo backtest");
    },
  });

  const demoWalkforwardMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runWalkForward({
          symbol: "NIFTY500",
          timeframe: "1d",
          strategy_template: "trend_breakout",
          config: { trials: 20, sampler: "tpe", pruner: "median" },
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Walk-forward queued");
      router.push("/walk-forward");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue walk-forward run");
    },
  });

  const demoResearchMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.runResearch({
          timeframes: ["1d"],
          strategy_templates: ["trend_breakout", "pullback_trend", "squeeze_breakout"],
          symbol_scope: "liquid",
          config: {
            trials_per_strategy: 6,
            max_symbols: 1,
            max_evaluations: 3,
            min_trades: 5,
            stress_pass_rate_threshold: 0.3,
            sampler: "random",
            pruner: "none",
            seed: 11,
            force_inline: forceInlineJobs,
          },
        })
      ).data,
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      queryClient.invalidateQueries({ queryKey: qk.jobs(20) });
      toast.success("Auto Research queued");
      router.push("/auto-research");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not queue Auto Research run");
    },
  });

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
      if (event.key === "Escape") {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const commands = useMemo<CommandItem[]>(
    () => [
      { id: "nav:dashboard", label: "Go to Dashboard", keywords: ["home"] },
      { id: "nav:universe", label: "Go to Universe & Data", keywords: ["import", "dataset"] },
      { id: "nav:lab", label: "Go to Strategy Lab", keywords: ["backtest"] },
      { id: "nav:walkforward", label: "Go to Walk-Forward", keywords: ["oos", "optuna"] },
      { id: "nav:research", label: "Go to Auto Research", keywords: ["policy", "candidates"] },
      { id: "nav:paper", label: "Go to Paper Trading", keywords: ["orders", "positions"] },
      { id: "nav:settings", label: "Go to Settings", keywords: ["risk", "costs"] },
      { id: "run:demo-backtest", label: "Run demo backtest", subtitle: "NIFTY500 Trend Breakout" },
      { id: "run:walkforward", label: "Run walk-forward", subtitle: "NIFTY500 Trend Breakout" },
      {
        id: "run:auto-research",
        label: "Run auto research",
        subtitle: "Multi-template robust scan",
      },
      { id: "job:open-latest", label: "Open latest job drawer", keywords: ["sse", "progress"] },
    ],
    [],
  );

  const runCommand = (id: string) => {
    setPaletteOpen(false);
    if (id === "nav:dashboard") return router.push("/");
    if (id === "nav:universe") return router.push("/universe-data");
    if (id === "nav:lab") return router.push("/strategy-lab");
    if (id === "nav:walkforward") return router.push("/walk-forward");
    if (id === "nav:research") return router.push("/auto-research");
    if (id === "nav:paper") return router.push("/paper-trading");
    if (id === "nav:settings") return router.push("/settings");
    if (id === "run:demo-backtest") return demoBacktestMutation.mutate();
    if (id === "run:walkforward") return demoWalkforwardMutation.mutate();
    if (id === "run:auto-research") return demoResearchMutation.mutate();
    if (id === "job:open-latest") {
      const jobs = jobsQuery.data ?? [];
      const active =
        jobs.find((job) => job.status === "RUNNING" || job.status === "QUEUED") ?? jobs[0];
      if (!active) {
        toast.error("No recent jobs available");
        return;
      }
      setActiveJobId(active.id);
    }
  };

  return (
    <div className="min-h-screen text-ink">
      <CommandPalette
        open={paletteOpen}
        commands={commands}
        onClose={() => setPaletteOpen(false)}
        onSelect={runCommand}
      />
      <JobDrawer jobId={activeJobId} onClose={() => setActiveJobId(null)} title="Job Progress" />

      <div className="mx-auto flex w-full max-w-[1440px] gap-4 p-4 md:p-6">
        <aside className="card hidden w-64 shrink-0 p-4 md:block" aria-label="Sidebar">
          <div className="mb-6">
            <p className="text-xs uppercase tracking-[0.14em] text-muted">Atlas</p>
            <h1 className="mt-1 text-xl font-semibold">Adaptive Swing Lab</h1>
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href as Route}
                  className={`focus-ring block rounded-xl px-3 py-2 text-sm transition ${
                    active
                      ? "bg-accent/10 text-accent"
                      : "text-muted hover:bg-surface hover:text-ink"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </aside>

        <div className="min-w-0 flex-1">
          <header className="card mb-4 flex items-center justify-between px-4 py-3 md:px-5">
            <div>
              <p className="text-xs uppercase tracking-[0.1em] text-muted">
                Research + Paper Trading
              </p>
              <p className="text-lg font-semibold">Local-first NIFTY 500 platform</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPaletteOpen(true)}
                className="focus-ring rounded-xl border border-border px-3 py-1.5 text-xs font-semibold text-muted"
              >
                Command (Ctrl/Cmd+K)
              </button>
              <ThemeToggle />
            </div>
          </header>

          <motion.main
            key={pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="space-y-5"
          >
            {children}
            <p className="px-1 pb-2 text-xs text-muted">{disclaimer}</p>
          </motion.main>
        </div>
      </div>
    </div>
  );
}
