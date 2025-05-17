import logging
from data_manager.etl_jobs.alphavantage_adapter import AlphaLoader

def load_initial_stocks(symbols):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("InitialLoader")

    logger.info(f"Starting initial load for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            loader = AlphaLoader(symbol=symbol, db_mode=True, local_store_mode=False)
            loader.get_daily_timeseries()
            loader.get_company_base()
            loader.get_financials(function='INCOME_STATEMENT')
            loader.get_financials(function='BALANCE_SHEET')
            loader.get_financials(function="CASH_FLOW")
            loader.get_financials(function="EARNINGS")
            loader.get_insider_transactions()
            loader.get_stock_splits()
            loader.get_dividends()
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")

    logger.info("✅ initial loader finished its running.")

def main():
    # List of symbols for initial load (example)
    symbols = ["MSFT", "AAPL", "TSLA", "NVDA", "AA"]

    # You could extend this by reading from a file or CLI args
    load_initial_stocks(symbols)

if __name__ == "__main__":
    main()

