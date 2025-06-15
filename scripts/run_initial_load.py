"""
Module to perform the initial data load from Alpha Vantage for a list of stock symbols.

This script retrieves multiple datasets including time series, company info,
financials, insider transactions, splits, and dividends. It stores the results
in a PostgreSQL database through the AlphaLoader interface.
"""

import logging
import sys
from utils.utils import get_symbols_from_csv
from data_manager.etl_jobs.alphavantage_adapter import AlphaLoader


def load_initial_stocks(symbols):
    """
    Load stock data for a list of symbols from Alpha Vantage and store in the database.

    Args:
        symbols (list[str]): List of stock ticker symbols to process.

    This function fetches daily time series, company fundamentals, various financial
    statements, insider transactions, stock splits, and dividend data for each symbol.
    """
    logger = logging.getLogger("InitialLoader")
    logger.info("Starting initial load for %d symbols...", len(symbols))

    for symbol in symbols:
        try:
            loader = AlphaLoader(
                symbol=symbol, db_mode=True, local_store_mode=False
            )
            loader.get_daily_timeseries()
            loader.get_company_base()
            loader.get_financials(function="INCOME_STATEMENT")
            loader.get_financials(function="BALANCE_SHEET")
            loader.get_financials(function="CASH_FLOW")
            loader.get_financials(function="EARNINGS")
            loader.get_insider_transactions()
            loader.get_stock_splits()
            loader.get_dividends()
        except Exception as exc:
            logger.exception("❌ Error processing %s: %s", symbol, exc)

    logger.info("✅ Initial loader finished running.")



def get_symbols():
    """
    Get a limited list of stock symbols for initial loading.

    Returns:
        list[str]: A list of stock ticker symbols.
    
    Note:
        The limit is intended for development/testing. Remove or adjust for production use.
    """
    return get_symbols_from_csv(limit=50)


def main():
    """
    Main entry point for the initial load script.

    Sets up logging, retrieves the list of stock symbols, and initiates the data load.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("initial_load.log", mode="w"),
        ],
    )

    symbols = get_symbols()
    load_initial_stocks(symbols)


if __name__ == "__main__":
    main()
