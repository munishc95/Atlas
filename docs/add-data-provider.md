# Add a Data Provider

The provider interface is defined in `apps/api/app/providers/base.py`.

Required methods:

- `get_symbols()`
- `get_ohlcv(symbol, timeframe, start, end)`
- `get_corporate_actions(symbol)`

## Steps

1. Create a provider in `apps/api/app/providers/` implementing `DataProvider`.
2. Return normalized OHLCV columns:
   - `datetime`, `open`, `high`, `low`, `close`, `volume`
3. Register provider selection in import/data services.
4. Ensure time is timezone-aware UTC.
5. Add tests for provider behavior.

## Current providers

- `CSVProvider`: parquet-backed local data.
- `MockProvider`: synthetic deterministic test data.
