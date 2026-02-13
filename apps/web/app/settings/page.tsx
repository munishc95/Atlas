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
          </>
        )}
      </section>
    </div>
  );
}
