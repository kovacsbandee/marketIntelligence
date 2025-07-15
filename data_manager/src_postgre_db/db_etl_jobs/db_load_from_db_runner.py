"""
This script defines a utility class `DBSymbolStorage` for loading and managing database tables 
as pandas DataFrames, filtered by a specific symbol if applicable. The script is designed to 
work with a PostgreSQL database using an adapter and ORM classes.

Classes:
    - DBSymbolStorage: A class that loads database tables into DataFrame attributes, 
      optionally filtering rows by a given symbol.

Functions:
    - main: Demonstrates the usage of the `DBSymbolStorage` class by loading tables 
      for a specific symbol and printing their details.

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
    python db_load_from_db_runner.py
    ```
Notes:
    - Ensure that the `PostgresAdapter` and `postgre_objects` modules are correctly 
      implemented and available in the specified paths.
    - Replace the hardcoded symbol "IBM" with user input or another mechanism as needed.

"""

import pandas as pd
from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
import data_manager.src_postgre_db.db_infrastructure.postgre_objects as orm_module
from data_manager.src_postgre_db.db_etl_jobs.db_initial_load_runner import load_stock_data
from utils.logger import get_logger  # <-- Import your project logger

# Set up project-standard logger
logger = get_logger("db_load_from_db_runner")
class DBSymbolStorage:
    """
    Stores each database table as a DataFrame attribute, filtered by symbol if applicable.
    """

    def __init__(self, adapter: PostgresAdapter, symbol: str):
        self._adapter = adapter
        self._symbol = symbol
        self._load_tables()

    def _load_tables(self):
        """
        Loads data from database tables into pandas DataFrames and assigns them as attributes of the instance.

        This method retrieves a list of table names from the adapter, dynamically identifies the corresponding
        ORM class for each table, and loads the data into a pandas DataFrame. If a table has a 'symbol' column,
        the data is filtered by the instance's `_symbol` attribute. Otherwise, all rows are loaded. The resulting
        DataFrame is then assigned as an attribute of the instance, named after the table.

        Logs warnings for tables without a corresponding ORM class and logs the number of rows loaded for each table.

        Raises:
            AttributeError: If the adapter does not have the required methods or attributes.

        Notes:
            - The ORM classes are expected to be accessible as attributes of `orm_module`.
            - The adapter must implement `list_tables`, `load_filtered_with_matching_values`, and `load_all` methods.

        """
        table_names = self._adapter.list_tables()
        for table_name in table_names:
            orm_class = orm_module.table_name_to_class.get(table_name)
            if orm_class is None:
                logger.warning("ORM class for table '%s' not found. Skipping.", table_name)
                continue
            # If table has a 'symbol' column, filter by symbol
            if hasattr(orm_class, "symbol"):
                rows = self._adapter.load_filtered_with_matching_values(orm_class, {"symbol": self._symbol})
            else:
                rows = self._adapter.load_all(orm_class)
            loaded_df = pd.DataFrame(rows)
            setattr(self, table_name, loaded_df)
            logger.info("Loaded table '%s' with %d rows.", table_name, len(loaded_df))

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

        This method iterates through the instance's attributes and checks if each
        attribute is an instance of `pandas.DataFrame`. It returns a list of attribute
        names that meet this condition.

        Returns:
            list: A list of attribute names (strings) that are pandas DataFrame objects.
        """
        # Only return attributes that are DataFrames
        return [attr for attr in self.__dict__ if isinstance(getattr(self, attr), pd.DataFrame)]

def print_loaded_tables(storage: DBSymbolStorage):
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
    symbol = "MSFT"  # Replace with user input as needed
    logger.info("Starting DB load for symbol: %s", symbol)
    adapter = PostgresAdapter()
    # Directly check if symbol exists in company_fundamentals
    orm_company_fundamentals = orm_module.table_name_to_class["company_fundamentals"]
    rows = adapter.load_filtered_with_matching_values(orm_company_fundamentals, {"symbol": symbol})

    if not rows:
        logger.warning("Symbol '%s' not found in company_fundamentals. Attempting to load...", symbol)
        print(f"Symbol '{symbol}' not found in company_fundamentals. Attempting to load...")
        load_stock_data([symbol])
        logger.info("Called load_stock_data for symbol: %s", symbol)

    # Now load all tables for the symbol (whether it was present or just loaded)
    storage = DBSymbolStorage(adapter, symbol)
    has_data = any(
        isinstance(getattr(storage, attr), pd.DataFrame) and not getattr(storage, attr).empty
        for attr in storage.__dict__
    )
    if not has_data:
        logger.error("Failed to load data for symbol '%s' after ETL.", symbol)
        print(f"Failed to load data for symbol '{symbol}'.")
    else:
        print_loaded_tables(storage)