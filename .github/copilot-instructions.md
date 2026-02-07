# Copilot Instructions for marketIntelligence

## Big picture
- Data pipeline: Alpha Vantage fetcher `infrastructure/alpha_adapter/alphavantage_adapter.py` (AlphaLoader) → transforms in `transform_utils` → Postgres via `CompanyDataManager` + ORM models in `company_table_objects.py` → tables consumed by `symbol.Symbol` and Dash UI.
- ETL orchestrators live in `infrastructure/databases/company/*.py` (build/drop/init load). Debug artifacts live under `logs/management/debug_data`.
- Dash UI (`infrastructure/ui/dash/app.py`) pulls data via `data_service.load_symbol_data` (wraps `Symbol`), renders Mantine/Plotly tabs for company base, price indicators, earnings, income, balance sheet, etc.

## Setup & run
- Python project with `requirements.txt`; load env from `.env` (`DB_HOST/PORT/NAME/USER/PASSWORD`, `ALPHA_API_KEY`, optional `ALPHA_LOCAL_STORE_PATH`, `CHAT_GPT_KEY`).
- Build schema: run `python -m infrastructure.databases.company.build_db` (drops+creates by default). Drop-only/combined scripts exist beside it.
- Initial data load: `python -m infrastructure.databases.company.initial_load` (samples symbols from `configs/nasdaq_screener.csv`) or call `download_stock_data(["MSFT", ...])` directly.
- Run Dash: `python -m infrastructure.ui.dash.app` (uses default theme, suppress_callback_exceptions=True).

## ETL patterns
- `download_stock_data` clears `logs/management/debug_data`, then runs a fixed function list on AlphaLoader (daily, company base, income/balance/cashflow/earnings, insider, splits, dividends) with verbose logging.
- AlphaLoader uses `_call_api_with_backoff` for rate limits; set `last_df` / `last_df_quarterly` before risky ops so `log_exception` can persist CSV + API JSON into debug_data. Keep symbols uppercase.
- Transform helpers (`preprocess_*` in `transform_utils`) normalize columns/types before inserting; prefer reusing them instead of ad-hoc munging.
- DB inserts go through `CompanyDataManager.insert_new_data`/`insert_update_one`; use ORM tables from `company_table_objects.py` or `table_name_to_class` mapping.

## Data access & indicators
- `symbol.Symbol` loads all tables for a symbol via `CompanyDataManager`, optionally triggers Alpha ETL if missing, checks freshness vs last workday, adjusts prices for splits, and can refresh in background.
- When `add_price_indicators=True` (default), it decorates price data using functions in `analyst/quantitative_analyst/add_indicators_to_price_data.py` (SMA/EMA/RSI/MACD/Bollinger/VWAP/Stochastic/OBV/ADX); keep column naming patterns (`sma_win_len_{window}`, etc.).

## Dash integration
- `data_service.load_symbol_data` returns JSON-serializable records plus `status_message` and default 6M date range; callbacks expect keys matching `Ids` constants.
- Tabs and loaders defined via `Ids`/`Tabs` enums (`ids.py`), `components.tab_panel`, and `panel_builders`/`plot_utils`. When adding tabs, update `Ids`, `Tabs`, and register in `app_callbacks.register_callbacks`.

## Logging, paths, conventions
- Use `utils.logger.get_logger` for management scripts (rotates `*.log` to `*.prev.log` in `logs/management`). Avoid hardcoded absolute paths; anchor to repo root like current AlphaLoader/initial_load.
- Table naming consistency matters (e.g., `cash_flow_quarterly` vs historical `cash_flow_statement_quarterly`). Use `table_name_to_class` to avoid typos.
- Prefer singleton DB adapter via `get_company_data_handler()` when reusing connections; otherwise `CompanyDataManager()` is fine per operation.
- Dataframes generally expect columns `date`, `open/high/low/close/volume`, `symbol`; dividends and splits have date-like fields—coerce to datetime/NaT instead of sentinel dates.

## Handy references
- Schema models: `infrastructure/databases/company/postgre_manager/company_table_objects.py`.
- ETL orchestrator: `infrastructure/databases/company/initial_load.py`.
- UI entry: `infrastructure/ui/dash/app.py`; callbacks in `app_callbacks.py`.
- Utility helpers: `utils/utils.py` (symbols CSV loader), `utils/logger.py` (logging), configs in `configs/config.py`.
