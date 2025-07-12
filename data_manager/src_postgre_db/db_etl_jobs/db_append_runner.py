"""
Appends new stock symbols' data to existing database tables using AlphaLoader.

This script:
    - Loads all available symbols from a CSV (e.g., NASDAQ screener).
    - Checks which symbols are already present in the database (using CompanyFundamentalsTable).
    - Selects only new symbols not yet present.
    - Loads and appends data for these new symbols using the same logic as the initial load runner.
    - Supports random sampling for batch processing.
    - Logs all actions and results.

Usage:
    $ python db_append_runner.py

Dependencies:
    - Requires AlphaLoader and PostgresAdapter to be implemented and accessible.
    - Database configuration and Alpha Vantage credentials must be set in the environment or code.
    - The initial load runner's logic is reused for consistency.
"""

import random

from utils.logger import get_logger
from utils.utils import get_symbols_from_csv
from data_manager.src_postgre_db.db_etl_jobs.db_initial_load_runner import load_stock_data
from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
from data_manager.src_postgre_db.db_infrastructure.postgre_objects import CompanyFundamentalsTable

def get_new_symbols(all_symbols):
    """
    Returns a list of symbols that are not yet present in the database.

    Args:
        all_symbols (list[str]): List of all available symbols (e.g., from CSV).

    Returns:
        list[str]: Symbols not already present in the CompanyFundamentalsTable.
    """
    adapter = PostgresAdapter()
    # Query all symbols already in the DB (change table/class as needed)
    existing = set(row['symbol'] for row in adapter.load_all(CompanyFundamentalsTable))
    # Only keep symbols not already present
    return [s for s in all_symbols if s not in existing]

def main():
    """
    Main entry point for appending new stock symbols' data to the database.

    Loads all symbols from CSV, filters out those already present in the database,
    randomly samples a batch, and loads data for the new symbols using the
    initial load logic.
    """
    logger = get_logger("db_append_runner")
    all_symbols = get_symbols_from_csv(csv_path="configs/nasdaq_screener.csv")
    new_symbols = get_new_symbols(all_symbols)
    num_to_load = 5  # Or all, or as needed
    random.seed(42)
    symbols = random.sample(new_symbols, min(num_to_load, len(new_symbols)))
    if not symbols:
        logger.info("No new symbols to append.")
        return
    load_stock_data(symbols)

if __name__ == "__main__":
    main()
