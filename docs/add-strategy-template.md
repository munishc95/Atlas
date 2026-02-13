# Add a Strategy Template

1. Add signal logic in `apps/api/app/strategies/templates.py`.
2. Register the template in `list_templates()` with:
   - `key`
   - `name`
   - `description`
   - `default_params`
   - `param_ranges`
   - `signal_fn`
3. Keep signal output as `pd.Series[bool]` aligned with OHLCV rows.
4. Ensure ATR parameters (`atr_stop_mult`, `atr_trail_mult`) exist in defaults/ranges if the strategy uses default risk controls.
5. Add/adjust tests in `apps/api/tests/test_strategies.py`.
6. Expose strategy in UI template cards (`apps/web/app/strategy-lab/page.tsx`) if needed.

Notes:

- Keep strategy logic pure and deterministic.
- Do not add future-leaking features (signals must use current/past bars only).
