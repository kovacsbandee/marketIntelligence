"""
Performs an initial load of stock data for a given list of symbols using AlphaLoader.

This script will:
    - For each provided symbol, fetch and load multiple datasets from Alpha Vantage
    - Store the data in the connected database (via AlphaLoader, configured for DB mode)
    - Log all processing and errors using structured logging

Usage:
    Run this script directly, or extend it to pull symbols from a CSV/file/CLI.
    Example:
        $ python initial_load_runner.py

Dependencies:
    - Requires AlphaLoader to be implemented and accessible.
    - Database configuration and Alpha Vantage credentials must be set in the environment or code.
"""

import random
import shutil
from pathlib import Path
import os

from utils.logger import get_logger
from utils.utils import get_symbols_from_csv
from infrastructure.alpha_adapter.alphavantage_adapter import AlphaLoader

# Use the same debug data root as AlphaLoader
DEBUG_DATA_ROOT = Path(__file__).resolve().parents[3] / "logs" / "management" / "debug_data"
INPUT_DIR = DEBUG_DATA_ROOT / "input"
OUTPUT_DIR = DEBUG_DATA_ROOT / "output"

def clear_debug_data():
    """
    Delete all files in logs/management/debug_data before each run.
    Also ensures 'input' and 'output' subdirectories exist for debug data.
    """
    if DEBUG_DATA_ROOT.exists() and DEBUG_DATA_ROOT.is_dir():
        for item in DEBUG_DATA_ROOT.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    # Ensure input and output subfolders exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_stock_data(symbols):
    """
    Load stock data for the provided list of symbols.

    For each symbol, loads:
        - Daily time series
        - Company base information
        - Income statement, balance sheet, cash flow, and earnings data
        - Insider transactions, stock splits, and dividends

    Args:
        symbols (list[str]): List of stock ticker symbols.
    """
    clear_debug_data()
    logger = get_logger("db_initial_load_runner")
    logger.info("Starting initial load for %d symbols...", len(symbols))

    loader_functions = [
        ("get_daily_timeseries", lambda loader: loader.get_daily_timeseries()),
        ("get_company_base", lambda loader: loader.get_company_base()),
        ("get_financials:INCOME_STATEMENT", lambda loader: loader.get_financials(function="INCOME_STATEMENT")),
        ("get_financials:BALANCE_SHEET", lambda loader: loader.get_financials(function="BALANCE_SHEET")),
        ("get_financials:CASH_FLOW", lambda loader: loader.get_financials(function="CASH_FLOW")),
        ("get_financials:EARNINGS", lambda loader: loader.get_financials(function="EARNINGS")),
        ("get_insider_transactions", lambda loader: loader.get_insider_transactions()),
        ("get_stock_splits", lambda loader: loader.get_stock_splits()),
        ("get_dividends", lambda loader: loader.get_dividends()),
    ]

    for symbol in symbols:
        loader = AlphaLoader(symbol=symbol, db_mode=True, local_store_mode=False, verbose_data_logging=True)
        logger.info("Processing symbol: %s", symbol)
        for func_name, func in loader_functions:
            try:
                logger.debug("Calling %s for %s", func_name, symbol)
                func(loader)
            except Exception as e:
                logger.error(
                    "Error in %s for %s: %s", func_name, symbol, str(e), exc_info=True
                )
                # If loader has a last_df or last_row attribute, log it:
                if hasattr(loader, "last_df") and loader.last_df is not None:
                    logger.error("Failing DataFrame for %s: %s", func_name, loader.last_df)
                if hasattr(loader, "last_row") and loader.last_row is not None:
                    logger.error("Failing row for %s: %s", func_name, loader.last_row)

    logger.info("Initial loader finished its running.")


def main():
    """
    Example entry point for initial stock data load.

    Defines a static list of symbols and triggers the initial load.
    """
    all_symbols = get_symbols_from_csv(csv_path="configs/nasdaq_screener.csv")
    num_to_load = 42  # Change as needed
    random.seed(42)
    symbols = random.sample(all_symbols, min(num_to_load, len(all_symbols)))
    download_stock_data(symbols)


if __name__ == "__main__":
    main()
