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

from utils.logger import get_logger
from data_manager.etl_jobs.alphavantage_adapter import AlphaLoader


def load_initial_stocks(symbols):
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
    logger = get_logger("run_initial_load")

    logger.info("Starting initial load for %d symbols...", len(symbols))

    loader_functions = [
        ("get_daily_timeseries", lambda loader: loader.get_daily_timeseries()),
        ("get_company_base", lambda loader: loader.get_company_base()),
        (
            "get_financials:INCOME_STATEMENT",
            lambda loader: loader.get_financials(function="INCOME_STATEMENT"),
        ),
        (
            "get_financials:BALANCE_SHEET",
            lambda loader: loader.get_financials(function="BALANCE_SHEET"),
        ),
        (
            "get_financials:CASH_FLOW",
            lambda loader: loader.get_financials(function="CASH_FLOW"),
        ),
        (
            "get_financials:EARNINGS",
            lambda loader: loader.get_financials(function="EARNINGS"),
        ),
        ("get_insider_transactions", lambda loader: loader.get_insider_transactions()),
        ("get_stock_splits", lambda loader: loader.get_stock_splits()),
        ("get_dividends", lambda loader: loader.get_dividends()),
    ]

    for symbol in symbols:
        loader = AlphaLoader(symbol=symbol, db_mode=True, local_store_mode=False)
        logger.info("Processing symbol: %s", symbol)
        for func_name, func in loader_functions:
            try:
                logger.debug("Calling %s for %s", func_name, symbol)
                func(loader)
            except Exception:
                logger.error("❌ Error in %s for %s", func_name, symbol, exc_info=True)

    logger.info("✅ Initial loader finished its running.")


def main():
    """
    Example entry point for initial stock data load.

    Defines a static list of symbols and triggers the initial load.
    """
    symbols = ["MSFT", "AAPL", "TSLA", "NVDA", "AA"]
    load_initial_stocks(symbols)


if __name__ == "__main__":
    main()
