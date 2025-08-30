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

import pandas as pd
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyDataManager
from infrastructure.databases.company.postgre_manager.postgre_objects import table_name_to_class
from data_manager.src_postgre_db.db_etl_jobs.db_initial_load_runner import download_stock_data

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

from utils.logger import get_logger  # <-- Import your project logger

#from analyst.analyst import Analyst, FinancialAnalyst, NewsAnalyst, QuantitativeAnalyst, LLMAnalyst

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

    def __init__(self, adapter: CompanyDataManager, 
                       symbol: str, 
                       auto_load_if_missing: bool = True,
                       add_price_indicators: bool = True):
        """
        Initialize the DBSymbolStorage.

        Args:
            adapter (PostgresAdapter): The database adapter.
            symbol (str): The stock symbol to filter tables by.
            auto_load_if_missing (bool): Whether to attempt to load data if the symbol is missing.
        """
        self._adapter = adapter
        self._symbol = symbol
        self.status_message = ""
        self.add_price_indicators = add_price_indicators
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
        # Check if symbol exists in company_fundamentals
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
        else:
            self.status_message = f"Data loaded for symbol '{self._symbol}'."

        # Proceed to load all tables
        table_names = self._adapter.list_tables()
        print("Debug: Loading tables:", table_names)
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
            if table_name == "cashflow_statement_quarterly":
                print("Debug: Loaded cash flow data:", loaded_df)
            setattr(self, table_name, loaded_df)
            logger.info("Loaded table '%s' with %d rows.", table_name, len(loaded_df))

        # Adjust price data for splits after all tables are loaded
        self._update_price_data_with_splits()

        if self.add_price_indicators:
            self.add_all_price_indicators()


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

if __name__ == "__main__":
    """
    Main entry point for loading and displaying tables for a given symbol.
    """
    symbol = "MSFT"  # Replace with user input as needed
    logger.info("Starting DB load for symbol: %s", symbol)
    adapter = CompanyDataManager()
    storage = Symbol(adapter, symbol)

    if storage.status_message.startswith("Failed"):
        print(storage.status_message)
    else:
        print_loaded_tables(storage)