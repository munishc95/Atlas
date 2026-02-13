"use client";

import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { atlasApi } from "@/src/lib/api/endpoints";
import { qk } from "@/src/lib/query/keys";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({
    queryKey: qk.settings,
    queryFn: async () => (await atlasApi.settings()).data,
  });

  const [form, setForm] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!settingsQuery.data) {
      return;
    }
    const nextForm: Record<string, string> = {};
    for (const [key, value] of Object.entries(settingsQuery.data)) {
      nextForm[key] = String(value ?? "");
    }
    setForm(nextForm);
  }, [settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const payload = {
        risk_per_trade: Number(form.risk_per_trade),
        max_positions: Number(form.max_positions),
        kill_switch_dd: Number(form.kill_switch_dd),
        cooldown_days: Number(form.cooldown_days),
        commission_bps: Number(form.commission_bps),
        slippage_base_bps: Number(form.slippage_base_bps),
        slippage_vol_factor: Number(form.slippage_vol_factor),
        cost_model_enabled: form.cost_model_enabled === "true",
        cost_mode: form.cost_mode,
        brokerage_bps: Number(form.brokerage_bps),
        stt_delivery_buy_bps: Number(form.stt_delivery_buy_bps),
        stt_delivery_sell_bps: Number(form.stt_delivery_sell_bps),
        stt_intraday_buy_bps: Number(form.stt_intraday_buy_bps),
        stt_intraday_sell_bps: Number(form.stt_intraday_sell_bps),
        exchange_txn_bps: Number(form.exchange_txn_bps),
        sebi_bps: Number(form.sebi_bps),
        stamp_delivery_buy_bps: Number(form.stamp_delivery_buy_bps),
        stamp_intraday_buy_bps: Number(form.stamp_intraday_buy_bps),
        gst_rate: Number(form.gst_rate),
        max_position_value_pct_adv: Number(form.max_position_value_pct_adv),
        diversification_corr_threshold: Number(form.diversification_corr_threshold),
        allowed_sides: String(form.allowed_sides ?? "")
          .split(",")
          .map((value) => value.trim().toUpperCase())
          .filter(Boolean),
        paper_short_squareoff_time: form.paper_short_squareoff_time,
        autopilot_max_symbols_scan: Number(form.autopilot_max_symbols_scan),
        autopilot_max_runtime_seconds: Number(form.autopilot_max_runtime_seconds),
        reports_auto_generate_daily: form.reports_auto_generate_daily === "true",
        health_window_days_short: Number(form.health_window_days_short),
        health_window_days_long: Number(form.health_window_days_long),
        drift_maxdd_multiplier: Number(form.drift_maxdd_multiplier),
        drift_negative_return_cost_ratio_threshold: Number(
          form.drift_negative_return_cost_ratio_threshold,
        ),
        drift_win_rate_drop_pct: Number(form.drift_win_rate_drop_pct),
        drift_return_delta_threshold: Number(form.drift_return_delta_threshold),
        drift_warning_risk_scale: Number(form.drift_warning_risk_scale),
        drift_degraded_risk_scale: Number(form.drift_degraded_risk_scale),
        drift_degraded_action: form.drift_degraded_action,
        futures_brokerage_bps: Number(form.futures_brokerage_bps),
        futures_stt_sell_bps: Number(form.futures_stt_sell_bps),
        futures_exchange_txn_bps: Number(form.futures_exchange_txn_bps),
        futures_stamp_buy_bps: Number(form.futures_stamp_buy_bps),
        futures_initial_margin_pct: Number(form.futures_initial_margin_pct),
        futures_symbol_mapping_strategy: form.futures_symbol_mapping_strategy,
        paper_use_simulator_engine: form.paper_use_simulator_engine === "true",
        four_hour_bars: form.four_hour_bars,
      };
      return (await atlasApi.updateSettings(payload)).data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: qk.settings });
      queryClient.invalidateQueries({ queryKey: qk.paperState });
      toast.success("Settings saved");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not save settings");
    },
  });

  const fields = useMemo(
    () => [
      { key: "risk_per_trade", label: "Risk per trade" },
      { key: "max_positions", label: "Max concurrent positions" },
      { key: "kill_switch_dd", label: "Kill-switch drawdown" },
      { key: "cooldown_days", label: "Cooldown (trading days)" },
      { key: "commission_bps", label: "Commission (bps)" },
      { key: "slippage_base_bps", label: "Base slippage (bps)" },
      { key: "slippage_vol_factor", label: "Slippage vol factor" },
      { key: "cost_model_enabled", label: "Use India cost model (true/false)" },
      { key: "cost_mode", label: "Cost mode (delivery/intraday)" },
      { key: "brokerage_bps", label: "Brokerage (bps)" },
      { key: "stt_delivery_buy_bps", label: "STT delivery buy (bps)" },
      { key: "stt_delivery_sell_bps", label: "STT delivery sell (bps)" },
      { key: "stt_intraday_buy_bps", label: "STT intraday buy (bps)" },
      { key: "stt_intraday_sell_bps", label: "STT intraday sell (bps)" },
      { key: "exchange_txn_bps", label: "Exchange txn charges (bps)" },
      { key: "sebi_bps", label: "SEBI charges (bps)" },
      { key: "stamp_delivery_buy_bps", label: "Stamp duty delivery buy (bps)" },
      { key: "stamp_intraday_buy_bps", label: "Stamp duty intraday buy (bps)" },
      { key: "gst_rate", label: "GST rate" },
      { key: "max_position_value_pct_adv", label: "Max position value / ADV" },
      { key: "diversification_corr_threshold", label: "Diversification correlation threshold" },
      { key: "allowed_sides", label: "Allowed sides (comma-separated BUY,SELL)" },
      { key: "paper_short_squareoff_time", label: "Cash short EOD square-off cutoff (HH:mm)" },
      { key: "autopilot_max_symbols_scan", label: "Autopilot max symbols scan (hard cap)" },
      { key: "autopilot_max_runtime_seconds", label: "Autopilot max runtime seconds" },
      { key: "reports_auto_generate_daily", label: "Auto-generate daily reports (true/false)" },
      { key: "health_window_days_short", label: "Policy health short window (days)" },
      { key: "health_window_days_long", label: "Policy health long window (days)" },
      { key: "drift_maxdd_multiplier", label: "Drift maxDD multiplier" },
      {
        key: "drift_negative_return_cost_ratio_threshold",
        label: "Drift cost ratio threshold (60d negative return)",
      },
      { key: "drift_win_rate_drop_pct", label: "Drift win-rate drop threshold" },
      { key: "drift_return_delta_threshold", label: "Drift return delta threshold" },
      { key: "drift_warning_risk_scale", label: "WARNING risk scale override" },
      { key: "drift_degraded_risk_scale", label: "DEGRADED risk scale override" },
      { key: "drift_degraded_action", label: "DEGRADED action (PAUSE/RETIRED/NONE)" },
      { key: "futures_brokerage_bps", label: "Futures brokerage (bps)" },
      { key: "futures_stt_sell_bps", label: "Futures STT sell (bps)" },
      { key: "futures_exchange_txn_bps", label: "Futures exchange txn charges (bps)" },
      { key: "futures_stamp_buy_bps", label: "Futures stamp duty buy (bps)" },
      { key: "futures_initial_margin_pct", label: "Futures initial margin %" },
      { key: "futures_symbol_mapping_strategy", label: "Futures symbol mapping strategy" },
      {
        key: "paper_use_simulator_engine",
        label: "Use unified simulator engine for paper execution (true/false)",
      },
      { key: "four_hour_bars", label: "Session bars" },
    ],
    [],
  );

  return (
    <div className="space-y-5">
      <section className="card p-4">
        <h2 className="text-xl font-semibold">Settings</h2>
        <p className="mt-1 text-sm text-muted">
          Risk controls, costs/slippage, market session, and scheduling.
        </p>

        {settingsQuery.isLoading ? (
          <LoadingState label="Loading runtime settings" />
        ) : settingsQuery.isError ? (
          <ErrorState
            title="Could not load settings"
            action="Check backend connectivity and retry."
            onRetry={() => void settingsQuery.refetch()}
          />
        ) : Object.keys(form).length === 0 ? (
          <EmptyState
            title="No settings payload"
            action="Refresh once backend settings endpoint is reachable."
          />
        ) : (
          <>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {fields.map((field) => (
                <label key={field.key} className="text-sm text-muted md:col-span-1">
                  {field.label}
                  <input
                    className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                    value={form[field.key] ?? ""}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        [field.key]: event.target.value,
                      }))
                    }
                  />
                </label>
              ))}
            </div>

            <button
              type="button"
              onClick={() => saveMutation.mutate()}
              className="focus-ring mt-4 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
              disabled={saveMutation.isPending}
            >
              {saveMutation.isPending ? "Saving..." : "Save settings"}
            </button>
            <p className="mt-2 text-xs text-muted">
              Unified simulator is default for paper execution; disable this only as a temporary legacy fallback.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
