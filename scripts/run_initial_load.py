import logging
import sys
from utils.utils import get_symbols_from_csv
from data_manager.etl_jobs.alphavantage_adapter import AlphaLoader

def load_initial_stocks(symbols):
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
            logger.exception(f"❌ Error processing {symbol}")

    logger.info("✅ initial loader finished its running.")

def get_symbols():
    return get_symbols_from_csv(limit=50)  # Remove limit for prod, keep for tests/dev!

def main():
    # Logging config (file + console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("initial_load.log", mode='w')
        ]
    )

    symbols = get_symbols()
    load_initial_stocks(symbols)

if __name__ == "__main__":
    main()
