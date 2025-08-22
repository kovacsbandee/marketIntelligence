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
Notes:
    - Ensure that the `PostgresAdapter` and `postgre_objects` modules are correctly 
      implemented and available in the specified paths.
    - Replace the hardcoded symbol "IBM" with user input or another mechanism as needed.

"""

import pandas as pd
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyDataManager
#from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter

from infrastructure.databases.company.postgre_manager.postgre_objects import table_name_to_class

#import data_manager.src_postgre_db.db_infrastructure.postgre_objects as orm_module
from data_manager.src_postgre_db.db_etl_jobs.db_initial_load_runner import download_stock_data
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

    def __init__(self, adapter: CompanyDataManager, symbol: str, auto_load_if_missing: bool = True):
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
        Update the price data with stock splits if applicable.
        This method is a placeholder and should be implemented with actual logic.
        """
        try:
            raise NotImplementedError("The _update_price_data_with_splits method is not yet implemented.")
        except NotImplementedError as e:
            logger.error("An error occurred: %s", str(e))


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

        # TODO: Implement logic to update price data with splits if applicable


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
    
    def do_financial_analysis(self):
        """
        This is a placeholder for applying the financial analysis logic implemented in the analyst module.
        """
        try:
            raise NotImplementedError("The financial analysis logic is not yet implemented.")
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