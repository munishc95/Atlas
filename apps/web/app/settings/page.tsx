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
  const operateStatusQuery = useQuery({
    queryKey: qk.operateStatus,
    queryFn: async () => (await atlasApi.operateStatus()).data,
  });
  const activeEnsembleId =
    typeof operateStatusQuery.data?.active_ensemble_id === "number"
      ? operateStatusQuery.data.active_ensemble_id
      : null;
  const activeEnsembleQuery = useQuery({
    queryKey: qk.ensemble(activeEnsembleId),
    queryFn: async () => (await atlasApi.ensembleById(Number(activeEnsembleId))).data,
    enabled: activeEnsembleId !== null,
  });
  const upstoxStatusQuery = useQuery({
    queryKey: qk.upstoxTokenStatus,
    queryFn: async () => (await atlasApi.upstoxTokenStatus()).data,
    refetchInterval: 15_000,
  });
  const upstoxTokenRequestLatestQuery = useQuery({
    queryKey: qk.upstoxTokenRequestLatest,
    queryFn: async () => {
      try {
        return (await atlasApi.upstoxTokenRequestLatest()).data;
      } catch {
        return null;
      }
    },
    refetchInterval: 15_000,
  });
  const upstoxTokenRequestHistoryQuery = useQuery({
    queryKey: qk.upstoxTokenRequestHistory(1, 10),
    queryFn: async () => (await atlasApi.upstoxTokenRequestHistory(1, 10)).data,
    refetchInterval: 15_000,
  });

  const [form, setForm] = useState<Record<string, string>>({});
  const [regimeWeightsText, setRegimeWeightsText] = useState("{}");

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

  useEffect(() => {
    if (!activeEnsembleQuery.data) {
      setRegimeWeightsText("{}");
      return;
    }
    const payload = activeEnsembleQuery.data.regime_weights ?? {};
    setRegimeWeightsText(JSON.stringify(payload, null, 2));
  }, [activeEnsembleQuery.data]);

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
        trading_calendar_segment: form.trading_calendar_segment,
        operate_mode: form.operate_mode,
        data_quality_stale_severity: form.data_quality_stale_severity,
        data_quality_max_stale_minutes_1d: Number(form.data_quality_max_stale_minutes_1d),
        data_quality_max_stale_minutes_intraday: Number(form.data_quality_max_stale_minutes_intraday),
        operate_auto_run_enabled: form.operate_auto_run_enabled === "true",
        operate_auto_run_time_ist: form.operate_auto_run_time_ist,
        operate_auto_run_include_data_updates: form.operate_auto_run_include_data_updates === "true",
        operate_auto_eval_enabled: form.operate_auto_eval_enabled === "true",
        operate_auto_eval_frequency: form.operate_auto_eval_frequency,
        operate_auto_eval_day_of_week: Number(form.operate_auto_eval_day_of_week),
        operate_auto_eval_time_ist: form.operate_auto_eval_time_ist,
        operate_auto_eval_lookback_trading_days: Number(form.operate_auto_eval_lookback_trading_days),
        operate_auto_eval_min_trades: Number(form.operate_auto_eval_min_trades),
        operate_auto_eval_cooldown_trading_days: Number(form.operate_auto_eval_cooldown_trading_days),
        operate_auto_eval_max_switches_per_30d: Number(form.operate_auto_eval_max_switches_per_30d),
        operate_auto_eval_auto_switch: form.operate_auto_eval_auto_switch === "true",
        operate_auto_eval_shadow_only_gate: form.operate_auto_eval_shadow_only_gate === "true",
        data_updates_inbox_enabled: form.data_updates_inbox_enabled === "true",
        data_updates_max_files_per_run: Number(form.data_updates_max_files_per_run),
        data_updates_provider_enabled: form.data_updates_provider_enabled === "true",
        data_updates_provider_kind: form.data_updates_provider_kind,
        data_updates_provider_max_symbols_per_run: Number(
          form.data_updates_provider_max_symbols_per_run,
        ),
        data_updates_provider_max_calls_per_run: Number(form.data_updates_provider_max_calls_per_run),
        data_updates_provider_timeframe_enabled: form.data_updates_provider_timeframe_enabled,
        data_updates_provider_timeframes: String(form.data_updates_provider_timeframes ?? "")
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
        data_updates_provider_repair_last_n_trading_days: Number(
          form.data_updates_provider_repair_last_n_trading_days,
        ),
        data_updates_provider_backfill_max_days: Number(form.data_updates_provider_backfill_max_days),
        data_updates_provider_allow_partial_4h_ish:
          form.data_updates_provider_allow_partial_4h_ish === "true",
        upstox_persist_env_fallback: form.upstox_persist_env_fallback === "true",
        upstox_auto_renew_enabled: form.upstox_auto_renew_enabled === "true",
        upstox_auto_renew_time_ist: form.upstox_auto_renew_time_ist,
        upstox_auto_renew_if_expires_within_hours: Number(
          form.upstox_auto_renew_if_expires_within_hours,
        ),
        upstox_auto_renew_only_when_provider_enabled:
          form.upstox_auto_renew_only_when_provider_enabled === "true",
        coverage_missing_latest_warn_pct: Number(form.coverage_missing_latest_warn_pct),
        coverage_missing_latest_fail_pct: Number(form.coverage_missing_latest_fail_pct),
        coverage_inactive_after_missing_days: Number(form.coverage_inactive_after_missing_days),
        risk_overlay_enabled: form.risk_overlay_enabled === "true",
        risk_overlay_target_vol_annual: Number(form.risk_overlay_target_vol_annual),
        risk_overlay_lookback_days: Number(form.risk_overlay_lookback_days),
        risk_overlay_min_scale: Number(form.risk_overlay_min_scale),
        risk_overlay_max_scale: Number(form.risk_overlay_max_scale),
        risk_overlay_max_gross_exposure_pct: Number(form.risk_overlay_max_gross_exposure_pct),
        risk_overlay_max_single_name_exposure_pct: Number(form.risk_overlay_max_single_name_exposure_pct),
        risk_overlay_max_sector_exposure_pct: Number(form.risk_overlay_max_sector_exposure_pct),
        risk_overlay_corr_clamp_enabled: form.risk_overlay_corr_clamp_enabled === "true",
        risk_overlay_corr_threshold: Number(form.risk_overlay_corr_threshold),
        risk_overlay_corr_reduce_factor: Number(form.risk_overlay_corr_reduce_factor),
        no_trade_enabled: form.no_trade_enabled === "true",
        no_trade_regimes: String(form.no_trade_regimes ?? "")
          .split(",")
          .map((value) => value.trim().toUpperCase())
          .filter(Boolean),
        no_trade_max_realized_vol_annual: Number(form.no_trade_max_realized_vol_annual),
        no_trade_min_breadth_pct: Number(form.no_trade_min_breadth_pct),
        no_trade_min_trend_strength: Number(form.no_trade_min_trend_strength),
        no_trade_cooldown_trading_days: Number(form.no_trade_cooldown_trading_days),
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
  const saveRegimeWeightsMutation = useMutation({
    mutationFn: async () => {
      if (activeEnsembleId === null) {
        throw new Error("No active ensemble selected.");
      }
      let parsed: Record<string, Record<string, number>> = {};
      try {
        const candidate = JSON.parse(regimeWeightsText);
        if (candidate && typeof candidate === "object") {
          parsed = candidate as Record<string, Record<string, number>>;
        }
      } catch (error) {
        throw new Error("Regime weights must be valid JSON.");
      }
      return (await atlasApi.putEnsembleRegimeWeights(activeEnsembleId, parsed)).data;
    },
    onSuccess: () => {
      toast.success("Regime weights saved");
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      if (activeEnsembleId !== null) {
        queryClient.invalidateQueries({ queryKey: qk.ensemble(activeEnsembleId) });
        queryClient.invalidateQueries({ queryKey: qk.ensembles(1, 100, null) });
      }
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not save regime weights");
    },
  });
  const connectUpstoxMutation = useMutation({
    mutationFn: async () => {
      if (typeof window === "undefined") {
        throw new Error("Connect action requires browser context.");
      }
      const redirectUri = `${window.location.origin}/providers/upstox/callback`;
      return (await atlasApi.upstoxAuthUrl({ redirect_uri: redirectUri })).data;
    },
    onSuccess: (data) => {
      if (typeof window !== "undefined") {
        window.location.href = data.auth_url;
      }
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not start Upstox connect flow");
    },
  });
  const verifyUpstoxMutation = useMutation({
    mutationFn: async () => (await atlasApi.upstoxTokenVerify()).data,
    onSuccess: () => {
      toast.success("Upstox token verified");
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Upstox token verification failed");
    },
  });
  const disconnectUpstoxMutation = useMutation({
    mutationFn: async () => (await atlasApi.upstoxDisconnect()).data,
    onSuccess: () => {
      toast.success("Upstox disconnected");
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not disconnect Upstox");
    },
  });
  const requestUpstoxTokenMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.upstoxTokenRequest({
          source: "settings_manual",
        })
      ).data,
    onSuccess: (payload) => {
      const dedupeTag = payload.deduplicated ? " (reused pending request)" : "";
      toast.success(`Upstox token request submitted${dedupeTag}`);
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenStatus });
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenRequestLatest });
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenRequestHistory(1, 10) });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not request Upstox token");
    },
  });
  const saveUpstoxAutoRenewMutation = useMutation({
    mutationFn: async () =>
      (
        await atlasApi.updateSettings({
          upstox_auto_renew_enabled: form.upstox_auto_renew_enabled === "true",
          upstox_auto_renew_time_ist: String(form.upstox_auto_renew_time_ist || "06:30"),
          upstox_auto_renew_if_expires_within_hours: Number(
            form.upstox_auto_renew_if_expires_within_hours || 12,
          ),
          upstox_auto_renew_only_when_provider_enabled:
            form.upstox_auto_renew_only_when_provider_enabled === "true",
        })
      ).data,
    onSuccess: () => {
      toast.success("Upstox auto-renew settings saved");
      queryClient.invalidateQueries({ queryKey: qk.settings });
      queryClient.invalidateQueries({ queryKey: qk.upstoxTokenStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateStatus });
      queryClient.invalidateQueries({ queryKey: qk.operateHealth(null, null) });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Could not save Upstox auto-renew settings");
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
      { key: "trading_calendar_segment", label: "Trading calendar segment (EQUITIES)" },
      { key: "operate_mode", label: "Operate mode (offline/live)" },
      { key: "data_quality_stale_severity", label: "Stale-data severity (WARN/FAIL)" },
      { key: "data_quality_max_stale_minutes_1d", label: "Stale limit 1D (minutes)" },
      { key: "data_quality_max_stale_minutes_intraday", label: "Stale limit intraday (minutes)" },
      { key: "operate_auto_run_enabled", label: "Auto-run scheduler enabled (true/false)" },
      { key: "operate_auto_run_time_ist", label: "Auto-run time IST (HH:mm)" },
      {
        key: "operate_auto_run_include_data_updates",
        label: "Auto-run includes data updates (true/false)",
      },
      { key: "operate_auto_eval_enabled", label: "Auto-evaluation enabled (true/false)" },
      { key: "operate_auto_eval_frequency", label: "Auto-eval frequency (WEEKLY/DAILY)" },
      { key: "operate_auto_eval_day_of_week", label: "Auto-eval day-of-week (0=Mon ... 6=Sun)" },
      { key: "operate_auto_eval_time_ist", label: "Auto-eval time IST (HH:mm)" },
      { key: "operate_auto_eval_lookback_trading_days", label: "Auto-eval lookback trading days" },
      { key: "operate_auto_eval_min_trades", label: "Auto-eval minimum trades gate" },
      { key: "operate_auto_eval_cooldown_trading_days", label: "Auto-eval switch cooldown (trading days)" },
      { key: "operate_auto_eval_max_switches_per_30d", label: "Auto-eval max switches per 30d" },
      { key: "operate_auto_eval_auto_switch", label: "Auto-eval auto-switch (true/false)" },
      { key: "operate_auto_eval_shadow_only_gate", label: "Auto-eval shadow-only safety gate (true/false)" },
      { key: "data_updates_inbox_enabled", label: "Data inbox updates enabled (true/false)" },
      { key: "data_updates_max_files_per_run", label: "Data updates max files per run" },
      { key: "data_updates_provider_enabled", label: "Provider updates enabled (true/false)" },
      { key: "data_updates_provider_kind", label: "Provider kind (UPSTOX/MOCK)" },
      {
        key: "data_updates_provider_max_symbols_per_run",
        label: "Provider updates max symbols per run",
      },
      {
        key: "data_updates_provider_max_calls_per_run",
        label: "Provider updates max API calls per run",
      },
      {
        key: "data_updates_provider_timeframe_enabled",
        label: "Provider updates enabled timeframes (comma-separated)",
      },
      {
        key: "data_updates_provider_timeframes",
        label: "Provider update timeframes list (comma-separated, e.g. 1d,4h_ish)",
      },
      {
        key: "data_updates_provider_repair_last_n_trading_days",
        label: "Provider repair last N trading days",
      },
      {
        key: "data_updates_provider_backfill_max_days",
        label: "Provider backfill max trading days",
      },
      {
        key: "data_updates_provider_allow_partial_4h_ish",
        label: "Allow partial 4h_ish bars (true/false)",
      },
      {
        key: "upstox_persist_env_fallback",
        label: "Upstox fallback: persist token to .env (true/false)",
      },
      { key: "upstox_auto_renew_enabled", label: "Upstox auto-renew enabled (true/false)" },
      { key: "upstox_auto_renew_time_ist", label: "Upstox auto-renew time IST (HH:mm)" },
      {
        key: "upstox_auto_renew_if_expires_within_hours",
        label: "Upstox auto-renew if expires within hours",
      },
      {
        key: "upstox_auto_renew_only_when_provider_enabled",
        label: "Upstox auto-renew only when provider updates enabled (true/false)",
      },
      { key: "coverage_missing_latest_warn_pct", label: "Coverage warning threshold (%)" },
      { key: "coverage_missing_latest_fail_pct", label: "Coverage fail threshold (%)" },
      {
        key: "coverage_inactive_after_missing_days",
        label: "Mark symbol inactive after missing trading days",
      },
      { key: "risk_overlay_enabled", label: "Risk overlay enabled (true/false)" },
      { key: "risk_overlay_target_vol_annual", label: "Risk overlay target annual vol" },
      { key: "risk_overlay_lookback_days", label: "Risk overlay lookback days" },
      { key: "risk_overlay_min_scale", label: "Risk overlay minimum scale" },
      { key: "risk_overlay_max_scale", label: "Risk overlay maximum scale" },
      { key: "risk_overlay_max_gross_exposure_pct", label: "Risk overlay max gross exposure (%)" },
      { key: "risk_overlay_max_single_name_exposure_pct", label: "Risk overlay max single-name exposure (%)" },
      { key: "risk_overlay_max_sector_exposure_pct", label: "Risk overlay max sector exposure (%)" },
      { key: "risk_overlay_corr_clamp_enabled", label: "Risk overlay correlation clamp enabled (true/false)" },
      { key: "risk_overlay_corr_threshold", label: "Risk overlay correlation threshold" },
      { key: "risk_overlay_corr_reduce_factor", label: "Risk overlay correlation reduce factor" },
      { key: "no_trade_enabled", label: "No-trade gate enabled (true/false)" },
      { key: "no_trade_regimes", label: "No-trade blocked regimes (comma-separated)" },
      { key: "no_trade_max_realized_vol_annual", label: "No-trade max realized annual vol" },
      { key: "no_trade_min_breadth_pct", label: "No-trade min breadth (%)" },
      { key: "no_trade_min_trend_strength", label: "No-trade min trend strength" },
      { key: "no_trade_cooldown_trading_days", label: "No-trade cooldown (trading days)" },
      { key: "four_hour_bars", label: "Session bars" },
    ],
    [],
  );

  const upstoxExpiryLabel = useMemo(() => {
    const expiresAt = upstoxStatusQuery.data?.expires_at;
    if (!expiresAt) {
      return "-";
    }
    return expiresAt;
  }, [upstoxStatusQuery.data?.expires_at]);
  const upstoxExpiresInLabel = useMemo(() => {
    const expiresAt = upstoxStatusQuery.data?.expires_at;
    if (!expiresAt) {
      return "-";
    }
    const ms = new Date(expiresAt).getTime() - Date.now();
    if (!Number.isFinite(ms)) {
      return "-";
    }
    if (ms <= 0) {
      return "expired";
    }
    const totalMinutes = Math.floor(ms / 60_000);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  }, [upstoxStatusQuery.data?.expires_at]);
  const requestHistory = upstoxTokenRequestHistoryQuery.data ?? [];
  const latestRequestRun = upstoxTokenRequestLatestQuery.data ?? null;

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
      <section className="card p-4">
        <h3 className="text-base font-semibold">Regime Ensemble Weights</h3>
        <p className="mt-1 text-sm text-muted">
          Configure regime-specific weights for the active ensemble.
        </p>
        <p className="mt-2 text-xs text-muted">
          Active ensemble:{" "}
          {activeEnsembleQuery.data?.name
            ? `${activeEnsembleQuery.data.name} (#${activeEnsembleQuery.data.id})`
            : "-"}
        </p>
        <label className="mt-3 block text-sm text-muted">
          Regime weights JSON
          <textarea
            className="focus-ring mt-1 min-h-[180px] w-full rounded-xl border border-border px-3 py-2 font-mono text-xs"
            value={regimeWeightsText}
            onChange={(event) => setRegimeWeightsText(event.target.value)}
          />
        </label>
        <button
          type="button"
          onClick={() => saveRegimeWeightsMutation.mutate()}
          className="focus-ring mt-3 rounded-xl border border-border px-4 py-2 text-sm text-muted"
          disabled={saveRegimeWeightsMutation.isPending || activeEnsembleId === null}
        >
          {saveRegimeWeightsMutation.isPending ? "Saving..." : "Save Regime Weights"}
        </button>
      </section>
      <section className="card p-4">
        <h3 className="text-base font-semibold">Providers - Upstox</h3>
        <p className="mt-1 text-sm text-muted">
          Connect Upstox for automated provider updates. Token is stored encrypted locally.
        </p>
        {upstoxStatusQuery.isLoading ? (
          <div className="mt-3">
            <LoadingState label="Loading Upstox status" />
          </div>
        ) : upstoxStatusQuery.isError ? (
          <div className="mt-3">
            <ErrorState
              title="Could not load Upstox status"
              action="Retry to fetch token status."
              onRetry={() => void upstoxStatusQuery.refetch()}
            />
          </div>
        ) : (
          <>
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Status:{" "}
                <span
                  className={`badge ${
                    upstoxStatusQuery.data?.connected
                      ? "bg-success/15 text-success"
                      : "bg-danger/15 text-danger"
                  }`}
                >
                  {upstoxStatusQuery.data?.connected ? "Connected" : "Disconnected"}
                </span>
              </p>
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Expires at: {upstoxExpiryLabel}
              </p>
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Expires in: {upstoxExpiresInLabel}
              </p>
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Last verified: {upstoxStatusQuery.data?.last_verified_at ?? "-"}
              </p>
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Last request status: {latestRequestRun?.status ?? "-"}
              </p>
              <p className="rounded-xl border border-border px-3 py-2 text-sm">
                Request auth window: {latestRequestRun?.authorization_expiry ?? "-"}
              </p>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => connectUpstoxMutation.mutate()}
                disabled={connectUpstoxMutation.isPending}
                className="focus-ring rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
              >
                {connectUpstoxMutation.isPending ? "Redirecting..." : "Connect"}
              </button>
              <button
                type="button"
                onClick={() => verifyUpstoxMutation.mutate()}
                disabled={verifyUpstoxMutation.isPending || !upstoxStatusQuery.data?.connected}
                className="focus-ring rounded-xl border border-border px-4 py-2 text-sm text-muted"
              >
                {verifyUpstoxMutation.isPending ? "Verifying..." : "Verify"}
              </button>
              <button
                type="button"
                onClick={() => disconnectUpstoxMutation.mutate()}
                disabled={disconnectUpstoxMutation.isPending || !upstoxStatusQuery.data?.connected}
                className="focus-ring rounded-xl border border-border px-4 py-2 text-sm text-muted"
              >
                {disconnectUpstoxMutation.isPending ? "Disconnecting..." : "Disconnect"}
              </button>
            </div>
            <div className="mt-4 rounded-xl border border-border p-3">
              <h4 className="text-sm font-semibold">Auto-Renew</h4>
              <p className="mt-1 text-xs text-muted">
                Atlas can request a new Upstox token approval before expiry. Approval still happens in Upstox.
              </p>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <label className="text-sm text-muted">
                  Enable auto-renew (true/false)
                  <input
                    className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                    value={form.upstox_auto_renew_enabled ?? "false"}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        upstox_auto_renew_enabled: event.target.value,
                      }))
                    }
                  />
                </label>
                <label className="text-sm text-muted">
                  Auto-renew time IST (HH:mm)
                  <input
                    className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                    value={form.upstox_auto_renew_time_ist ?? "06:30"}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        upstox_auto_renew_time_ist: event.target.value,
                      }))
                    }
                  />
                </label>
                <label className="text-sm text-muted">
                  Request when expiring within hours
                  <input
                    className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                    value={form.upstox_auto_renew_if_expires_within_hours ?? "12"}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        upstox_auto_renew_if_expires_within_hours: event.target.value,
                      }))
                    }
                  />
                </label>
                <label className="text-sm text-muted">
                  Only when provider updates enabled (true/false)
                  <input
                    className="focus-ring mt-1 w-full rounded-xl border border-border px-3 py-2"
                    value={form.upstox_auto_renew_only_when_provider_enabled ?? "true"}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        upstox_auto_renew_only_when_provider_enabled: event.target.value,
                      }))
                    }
                  />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => saveUpstoxAutoRenewMutation.mutate()}
                  disabled={saveUpstoxAutoRenewMutation.isPending}
                  className="focus-ring rounded-xl border border-border px-4 py-2 text-sm text-muted"
                >
                  {saveUpstoxAutoRenewMutation.isPending ? "Saving..." : "Save Auto-Renew"}
                </button>
                <button
                  type="button"
                  onClick={() => requestUpstoxTokenMutation.mutate()}
                  disabled={requestUpstoxTokenMutation.isPending}
                  className="focus-ring rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white"
                >
                  {requestUpstoxTokenMutation.isPending ? "Requesting..." : "Request token now"}
                </button>
              </div>
              <details className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
                <summary className="cursor-pointer font-medium text-foreground">How it works</summary>
                <p className="mt-2">1. Atlas requests approval from Upstox.</p>
                <p className="mt-1">2. You approve the request in Upstox.</p>
                <p className="mt-1">3. Atlas receives the token via notifier webhook.</p>
              </details>
              <div className="mt-3 rounded-xl border border-border px-3 py-2 text-xs text-muted">
                <p>Latest notifier endpoint:</p>
                <p className="mt-1 break-all">
                  {latestRequestRun?.notifier_url ??
                    "Run \"Request token now\" to generate a nonce-specific notifier URL."}
                </p>
              </div>
            </div>
            <div className="mt-4 rounded-xl border border-border p-3">
              <h4 className="text-sm font-semibold">Token Request History</h4>
              {upstoxTokenRequestHistoryQuery.isLoading ? (
                <div className="mt-2">
                  <LoadingState label="Loading token request history" />
                </div>
              ) : upstoxTokenRequestHistoryQuery.isError ? (
                <div className="mt-2">
                  <ErrorState
                    title="Could not load token request history"
                    action="Retry to refresh request runs."
                    onRetry={() => void upstoxTokenRequestHistoryQuery.refetch()}
                  />
                </div>
              ) : requestHistory.length === 0 ? (
                <div className="mt-2">
                  <EmptyState
                    title="No token requests yet"
                    action='Click "Request token now" to start approval.'
                  />
                </div>
              ) : (
                <div className="mt-2 overflow-hidden rounded-xl border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-surface text-left text-muted">
                      <tr>
                        <th className="px-3 py-2">Status</th>
                        <th className="px-3 py-2">Requested</th>
                        <th className="px-3 py-2">Auth expiry</th>
                      </tr>
                    </thead>
                    <tbody>
                      {requestHistory.map((run) => (
                        <tr key={run.id} className="border-t border-border">
                          <td className="px-3 py-2">{run.status}</td>
                          <td className="px-3 py-2">{run.requested_at ?? "-"}</td>
                          <td className="px-3 py-2">{run.authorization_expiry ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="mt-2 text-xs text-muted">
                Pending approval: approve in Upstox app and keep your notifier tunnel running.
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
