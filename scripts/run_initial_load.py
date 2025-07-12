"""
Module to perform the initial data load from Alpha Vantage for a list of stock symbols.

This script retrieves multiple datasets including time series, company info,
financials, insider transactions, splits, and dividends. It stores the results
in a PostgreSQL database through the AlphaLoader interface.
"""

from utils.logger import get_logger
from utils.utils import get_symbols_from_csv
from data_manager.db_etl_scripts.alphavantage_adapter import AlphaLoader


def load_initial_stocks(symbols):
    """
    Load stock data for a list of symbols from Alpha Vantage and store in the database.

    Args:
        symbols (list[str]): List of stock ticker symbols to process.

    This function fetches daily time series, company fundamentals, various financial
    statements, insider transactions, stock splits, and dividend data for each symbol.
    """
    logger = get_logger("run_initial_load")
    logger.info("Starting initial load for %d symbols...", len(symbols))

    for symbol in symbols:
        try:
            logger.info("Processing symbol: %s", symbol)
            loader = AlphaLoader(symbol=symbol, db_mode=True, local_store_mode=False)
            for func in [
                loader.get_daily_timeseries,
                loader.get_company_base,
                lambda: loader.get_financials(function="INCOME_STATEMENT"),
                lambda: loader.get_financials(function="BALANCE_SHEET"),
                lambda: loader.get_financials(function="CASH_FLOW"),
                lambda: loader.get_financials(function="EARNINGS"),
                loader.get_insider_transactions,
                loader.get_stock_splits,
                loader.get_dividends,
            ]:
                logger.debug("Calling %s for %s", func.__name__, symbol)
                func()
            logger.info("Finished processing symbol: %s", symbol)
        except Exception:
            logger.error("❌ Error processing %s", symbol, exc_info=True)

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
    logger = get_logger("run_initial_load")

    symbols = get_symbols()
    load_initial_stocks(symbols)


if __name__ == "__main__":
    main()
