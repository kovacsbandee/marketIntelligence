"""
This script defines a utility class `DBSymbolStorage` for loading and managing database tables 
as pandas DataFrames, filtered by a specific symbol if applicable. The script is designed to 
work with a PostgreSQL database using an adapter and ORM classes.

Classes:
    - DBSymbolStorage: A class that loads database tables into DataFrame attributes, 
      optionally filtering rows by a given symbol.

Functions:
    - print_loaded_tables: Prints the names and row counts of all tables stored in the given DBSymbolStorage.

Usage:
    - The script initializes a `PostgresAdapter` instance and uses it to load tables 
      into the `DBSymbolStorage` class. The tables are then accessible as attributes 
      of the `DBSymbolStorage` instance.

Dependencies:
    - logging: For logging warnings and information.
    - pandas: For handling data as DataFrames.
    - PostgresAdapter: A custom adapter for interacting with the PostgreSQL database.
    - postgre_objects: A module containing ORM classes for database tables.

Example:
    To run the script, execute it as a standalone program. It will load tables for 
    the symbol "IBM" (or another symbol if modified) and print the loaded table names 
    and their row counts.
    ```
    python [db_load_from_db_runner.py](http://_vscodecontentref_/0)
    ```
"""

import threading
import pandas as pd
from infrastructure.databases.company.postgre_manager.company_data_manager import CompanyDataManager
from infrastructure.databases.company.postgre_manager.company_table_objects import table_name_to_class
from infrastructure.databases.company.initial_load import download_stock_data

from analyst.quantitative_analyst.add_indicators_to_price_data import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_vwap,
    calculate_stochastic,
    calculate_obv,
    calculate_adx,
)

from analyst.financial_analyst.add_financial_metrics import (
    calculate_profitability_margins,
    calculate_revenue_growth,
    calculate_liquidity_ratios,
    calculate_leverage_ratios,
    calculate_cashflow_metrics,
    calculate_earnings_metrics,
)

from utils.logger import get_logger  # <-- Import your project logger

# Set up project-standard logger
logger = get_logger("db_load_from_db_runner")


class Symbol:
    """
    Stores each database table as a DataFrame attribute, filtered by symbol if applicable.
    Handles missing symbols by optionally triggering ETL.

    Attributes:
        _adapter (PostgresAdapter): The database adapter.
        _symbol (str): The stock symbol to filter tables by.
        status_message (str): Status message about the data loading process.
    """

    # per-symbol refresh guard (in-memory) to avoid hammering the API
    _refresh_registry = {}
    _refresh_registry_lock = threading.Lock()

    def __init__(self, adapter: CompanyDataManager, 
                       symbol: str, 
                       auto_load_if_missing: bool = True,
                       add_price_indicators: bool = True,
                       add_financial_metrics: bool = True):
        """
        Initialize the DBSymbolStorage.

        Args:
            adapter (PostgresAdapter): The database adapter.
            symbol (str): The stock symbol to filter tables by.
            auto_load_if_missing (bool): Whether to attempt to load data if the symbol is missing.
            add_price_indicators (bool): Whether to compute technical indicators on daily_timeseries.
            add_financial_metrics (bool): Whether to compute derived financial metrics on quarterly tables.
        """
        self._adapter = adapter
        self._symbol = symbol
        self.status_message = ""
        self.refreshing = False
        self.last_refresh_completed_at = None
        self._refresh_thread = None
        self.add_price_indicators = add_price_indicators
        self.add_financial_metrics = add_financial_metrics
        self._load_tables(auto_load_if_missing)

    def _symbol_exists(self) -> bool:
        """
        Check if the symbol exists in the company_fundamentals table.

        Returns:
            bool: True if the symbol exists, False otherwise.
        """
        orm_company_fundamentals = table_name_to_class["company_fundamentals"]
        rows = self._adapter.load_filtered_with_matching_values(orm_company_fundamentals, {"symbol": self._symbol})
        return bool(rows)
    
    def _check_for_latest_date(self) -> tuple[bool, dict | None]:
        """
        Check if the latest date in the daily_timeseries table for the symbol is up to date.

        Returns:
            bool: True if the latest date is the last workday's date, False otherwise.
        """
        date_filter_values = None
        try:
            daily_timeseries_df = getattr(self, "daily_timeseries", None)
            if daily_timeseries_df is None or daily_timeseries_df.empty:
                logger.warning("No daily_timeseries data loaded; cannot check latest date.")
                return False, None

            latest_date_in_db = pd.to_datetime(daily_timeseries_df['date']).max()
            today = pd.Timestamp.today().normalize()
            last_workday = today - pd.offsets.BDay(1)
            date_filter_values = dict()
            date_filter_values["latest_date_in_db"] = latest_date_in_db
            date_filter_values["last_workday"] = last_workday
            logger.info("Latest date in DB: %s, Last workday: %s", latest_date_in_db, last_workday)
            if latest_date_in_db >= last_workday:
                return True, date_filter_values
            else:
                logger.info("Latest date in DB (%s) is not up to date (last workday: %s).", latest_date_in_db, last_workday)
                return False, date_filter_values

        except Exception as e:
            logger.error("Error checking for latest date: %s", str(e))
            return False, date_filter_values

    def _mark_refresh_attempt(self):
        with self._refresh_registry_lock:
            self._refresh_registry[self._symbol] = pd.Timestamp.utcnow()

    def _should_skip_refresh(self) -> bool:
        with self._refresh_registry_lock:
            ts = self._refresh_registry.get(self._symbol)
        if ts is None:
            return False
        return (pd.Timestamp.utcnow() - ts) < pd.Timedelta(hours=6)

    def _refresh_in_background(self):
        """Run full Alpha loader in a background thread, then reload tables into memory."""
        try:
            self.refreshing = True
            self.status_message = (
                f"Refreshing data for symbol '{self._symbol}' in background..."
            )
            logger.info(self.status_message)
            self._mark_refresh_attempt()

            # Run full pipeline (reuses existing implementation)
            download_stock_data([self._symbol])
            logger.info("Background refresh completed Alpha load for %s", self._symbol)

            # Reload tables from DB into this instance
            self._load_all_tables_from_db()

            self.status_message = (
                f"Data refreshed for symbol '{self._symbol}' (background)."
            )
            self.last_refresh_completed_at = pd.Timestamp.utcnow()
            logger.info(self.status_message)
        except Exception as e:
            logger.error("Background refresh failed for %s: %s", self._symbol, e)
            self.status_message = f"Background refresh failed for '{self._symbol}'."
        finally:
            self.refreshing = False
            self._refresh_thread = None

    def _update_price_data_with_splits(self):
        """
        Adjusts the daily_timeseries DataFrame for stock splits using the stock_splits DataFrame.
        Only updates the DataFrame in memory.
        """
        try:
            price_df = getattr(self, "daily_timeseries", None)
            stock_splits_df = getattr(self, "stock_splits", None)
            if price_df is None or stock_splits_df is None:
                logger.warning("Price data or stock_splits data not loaded; skipping split adjustment.")
                return

            price_df = price_df.copy()
            price_df["date"] = pd.to_datetime(price_df["date"])
            stock_splits_df = stock_splits_df[stock_splits_df["symbol"] == self._symbol].copy()
            stock_splits_df["effective_date"] = pd.to_datetime(stock_splits_df["effective_date"])
            stock_splits_df = stock_splits_df.sort_values("effective_date")

            price_df = price_df.sort_values("date")
            price_df["adj_factor"] = 1.0

            for _, split in stock_splits_df.iterrows():
                mask = price_df["date"] < split["effective_date"]
                price_df.loc[mask, "adj_factor"] *= float(split["split_factor"])

            for col in ["open", "high", "low", "close"]:
                if col in price_df.columns:
                    price_df[col] = price_df[col] / price_df["adj_factor"]

            if "volume" in price_df.columns:
                price_df["volume"] = price_df["volume"] * price_df["adj_factor"]

            price_df = price_df.drop(columns=["adj_factor"])
            setattr(self, "daily_timeseries", price_df)
            logger.info("Adjusted daily_timeseries for splits.")

        except Exception as e:
            logger.error("Error adjusting prices for splits: %s", str(e))

    def _load_all_tables_from_db(self):
        """Load all tables for the symbol from DB into DataFrame attributes."""
        table_names = self._adapter.list_tables()
        for table_name in table_names:
            orm_class = table_name_to_class.get(table_name)
            if orm_class is None:
                logger.warning("ORM class for table '%s' not found. Skipping.", table_name)
                continue
            if hasattr(orm_class, "symbol"):
                rows = self._adapter.load_filtered_with_matching_values(orm_class, {"symbol": self._symbol})
            else:
                rows = self._adapter.load_all(orm_class)
            loaded_df = pd.DataFrame(rows)
            setattr(self, table_name, loaded_df)
            logger.info("Loaded table '%s' with %d rows.", table_name, len(loaded_df))

        # Adjust price data for splits after all tables are loaded
        self._update_price_data_with_splits()

        if self.add_price_indicators:
            self.add_all_price_indicators()

        if self.add_financial_metrics:
            self.add_all_financial_metrics()

    def _load_tables(self, auto_load_if_missing: bool):
        """
        Load all tables for the given symbol. If the symbol is missing and auto_load_if_missing is True,
        attempt to load the data using the ETL utility.

        Args:
            auto_load_if_missing (bool): Whether to attempt to load data if the symbol is missing.

        TODO: I need to update the logic in the _load_tables() function! 
        Right now if the symbol exists in the DB it loads it into the application from the state,
         when it was downloaded from alpha vantage. 
         In this way in the dashboard the prices and other tables are not the freshest available 
         when the dashboard is used. I need you to update the functions logic, in such a way, 
         that if the symbol exists in the db, the _load_tables() checks if the last entry for the symbol is the
         latest possible, it needs to fit to the last workday's date.
        """
        # 1) Ensure symbol exists (load if missing)
        if not self._symbol_exists():
            self.status_message = (
                f"Symbol '{self._symbol}' not found in database. Attempting to load..."
            )
            logger.warning(self.status_message)
            if auto_load_if_missing:
                download_stock_data([self._symbol])
                logger.info("Called load_stock_data for symbol: %s", self._symbol)
                # Check again after ETL
                if not self._symbol_exists():
                    self.status_message = (
                        f"Failed to load data for symbol '{self._symbol}'."
                    )
                    logger.error(self.status_message)
                    return
                else:
                    self.status_message = (
                        f"Data loaded for symbol '{self._symbol}'."
                    )
            else:
                return

        # 2) Load current tables from DB
        self._load_all_tables_from_db()

        # 3) Freshness check on daily_timeseries
        up_to_date, dates = self._check_for_latest_date()
        if up_to_date:
            self.status_message = f"Data loaded for symbol '{self._symbol}'."
            logger.info(self.status_message)
            return

        # 4) If stale, decide whether to refresh (with 6h guard)
        if self._should_skip_refresh():
            self.status_message = (
                f"Data for '{self._symbol}' may be stale; recent refresh attempt within 6h."
            )
            logger.info(self.status_message)
            return

        self.status_message = (
            f"Data for symbol '{self._symbol}' is outdated (latest: {dates.get('latest_date_in_db') if dates else 'unknown'}). "
            "Refreshing in background..."
        )
        logger.info(self.status_message)

        # Kick off background refresh
        self._refresh_thread = threading.Thread(target=self._refresh_in_background, daemon=True)
        self._refresh_thread.start()


    def get_table(self, table_name: str) -> pd.DataFrame:
        """
        Retrieve a table as a pandas DataFrame by its name.

        Args:
            table_name (str): The name of the table to retrieve.

        Returns:
            pd.DataFrame: The table corresponding to the given name, or None if the table does not exist.
        """
        return getattr(self, table_name, None)

    def list_tables(self):
        """
        Lists all attributes of the instance that are pandas DataFrame objects.

        Returns:
            list: A list of attribute names (strings) that are pandas DataFrame objects.
        """
        return [attr for attr in self.__dict__ if isinstance(getattr(self, attr), pd.DataFrame)]

    def add_all_price_indicators(self):
        """
        This method adds all relevant price indicators to the symbol's daily_timeseries DataFrame.
        """
        try:
            price_df = getattr(self, "daily_timeseries", None)
            if price_df is None or price_df.empty:
                logger.warning("No daily_timeseries data loaded; cannot compute indicators.")
                return

            # Add indicators in-place, chaining each function
            price_df = calculate_sma(price_df, window=20)  # Example window
            price_df = calculate_ema(price_df, window=20)  # Example window
            price_df = calculate_rsi(price_df, window=14)
            price_df = calculate_macd(price_df)
            price_df = calculate_bollinger_bands(price_df, window=20, num_std=2)
            price_df = calculate_vwap(price_df)
            price_df = calculate_stochastic(price_df, k_window=14, d_window=3)
            price_df = calculate_obv(price_df)
            price_df = calculate_adx(price_df, window=14)

            setattr(self, "daily_timeseries", price_df)
            logger.info("All price indicators added to daily_timeseries.")
        except Exception as e:
            logger.error("Error adding price indicators: %s", str(e))
            

    def add_all_financial_metrics(self):
        """
        Enrich quarterly financial statement DataFrames with derived metrics
        (margins, ratios, growth rates, etc.).

        Mirrors ``add_all_price_indicators`` for fundamental data.  Each
        quarterly table is passed through the corresponding pure-function
        calculator defined in ``analyst/financial_analyst/add_financial_metrics.py``.
        The enriched DataFrame replaces the original attribute on this instance.
        """
        # --- Income statement quarterly: profitability margins & revenue growth ---
        income_df = getattr(self, "income_statement_quarterly", None)
        if income_df is not None and not income_df.empty:
            try:
                income_df = calculate_profitability_margins(income_df)
                income_df = calculate_revenue_growth(income_df)
                setattr(self, "income_statement_quarterly", income_df)
                logger.info("Financial metrics added to income_statement_quarterly.")
            except Exception as e:
                logger.error("Error adding income-statement metrics: %s", str(e))

        # --- Balance sheet quarterly: liquidity & leverage ratios ---
        balance_df = getattr(self, "balance_sheet_quarterly", None)
        if balance_df is not None and not balance_df.empty:
            try:
                balance_df = calculate_liquidity_ratios(balance_df)
                balance_df = calculate_leverage_ratios(balance_df)
                setattr(self, "balance_sheet_quarterly", balance_df)
                logger.info("Financial metrics added to balance_sheet_quarterly.")
            except Exception as e:
                logger.error("Error adding balance-sheet metrics: %s", str(e))

        # --- Cash flow quarterly: FCF, CF quality ---
        cashflow_df = getattr(self, "cash_flow_quarterly", None)
        if cashflow_df is not None and not cashflow_df.empty:
            try:
                cashflow_df = calculate_cashflow_metrics(cashflow_df)
                setattr(self, "cash_flow_quarterly", cashflow_df)
                logger.info("Financial metrics added to cash_flow_quarterly.")
            except Exception as e:
                logger.error("Error adding cash-flow metrics: %s", str(e))

        # --- Earnings quarterly: EPS growth, beat flag ---
        earnings_df = getattr(self, "earnings_quarterly", None)
        if earnings_df is not None and not earnings_df.empty:
            try:
                earnings_df = calculate_earnings_metrics(earnings_df)
                setattr(self, "earnings_quarterly", earnings_df)
                logger.info("Financial metrics added to earnings_quarterly.")
            except Exception as e:
                logger.error("Error adding earnings metrics: %s", str(e))

    def analyse_symbol_with_all_the_analysts(self):
        """
        This is a placeholder for analysing the symbol with all available analysts.
        """
        try:
            raise NotImplementedError("The analyse_symbol_with_all_the_analysts method is not yet implemented.")
        except NotImplementedError as e:
            logger.error("An error occurred: %s", str(e))

    def hand_over_to_brokerage(self):
        """
        This is a placeholder for handing over the data to the brokerage module.
        """
        try:
            raise NotImplementedError("The hand_over_to_brokerage method is not yet implemented.")
        except NotImplementedError as e:
            logger.error("An error occurred: %s", str(e))
            

def print_loaded_tables(storage: Symbol):
    """
    Prints the names and row counts of all tables stored in the given DBSymbolStorage.

    Args:
        storage (DBSymbolStorage): The storage object containing the tables to be listed and printed.
    """
    logger.info("Loaded tables: %s", storage.list_tables())
    for name in storage.list_tables():
        df = storage.get_table(name)
        logger.info("%s: %d rows", name, df.shape[0])
        print(f"{name}: {df.shape} rows")

# if __name__ == "__main__":
#     """
#     Main entry point for loading and displaying tables for a given symbol.
#     """
#     symbol = "MSFT"  # Replace with user input as needed
#     logger.info("Starting DB load for symbol: %s", symbol)
#     adapter = CompanyDataManager()
#     storage = Symbol(adapter, symbol)

#     if storage.status_message.startswith("Failed"):
#         print(storage.status_message)
#     else:
#         print_loaded_tables(storage)