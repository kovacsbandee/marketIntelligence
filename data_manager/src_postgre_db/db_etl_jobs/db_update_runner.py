import logging
from datetime import datetime
from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
from data_manager.src_postgre_db.db_infrastructure.postgre_objects import (
    CompanyFundamentalsTable, DailyTimeSeries, AnnualIncomeStatement, QuarterlyIncomeStatement,
    AnnualBalanceSheetTable, QuarterlyBalanceSheetTable, AnnualCashFlowTable, QuarterlyCashFlowTable,
    AnnualEarningsTable, QuarterlyEarningsTable, InsiderTransactionTable, StockSplit, DividendsTable
)
from data_manager.src_postgre_db.db_etl_scripts.alphavantage_adapter import AlphaLoader

def update_company_data(symbol):
    logger = logging.getLogger("db_update_runner")
    adapter = PostgresAdapter()
    symbol = symbol.upper()
    loader = AlphaLoader(symbol, db_mode=True)

    # 1. Update Daily Time Series
    try:
        logger.info(f"Updating daily timeseries for {symbol}")
        loader.get_daily_timeseries()
    except Exception as e:
        logger.error(f"Failed to update daily timeseries for {symbol}: {e}")

    # 2. Update Company Fundamentals
    try:
        logger.info(f"Updating company fundamentals for {symbol}")
        loader.get_company_base()
    except Exception as e:
        logger.error(f"Failed to update company fundamentals for {symbol}: {e}")

    # 3. Update Financials (Income Statement, Balance Sheet, Cash Flow, Earnings)
    for function in ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS"]:
        try:
            logger.info(f"Updating {function} for {symbol}")
            loader.get_financials(function)
        except Exception as e:
            logger.error(f"Failed to update {function} for {symbol}: {e}")

    # 4. Update Insider Transactions
    try:
        logger.info(f"Updating insider transactions for {symbol}")
        loader.get_insider_transactions()
    except Exception as e:
        logger.error(f"Failed to update insider transactions for {symbol}: {e}")

    # 5. Update Stock Splits
    try:
        logger.info(f"Updating stock splits for {symbol}")
        loader.get_stock_splits()
    except Exception as e:
        logger.error(f"Failed to update stock splits for {symbol}: {e}")

    # 6. Update Dividends
    try:
        logger.info(f"Updating dividends for {symbol}")
        loader.get_dividends()
    except Exception as e:
        logger.error(f"Failed to update dividends for {symbol}: {e}")

    # 7. Mark company as up-to-date
    try:
        logger.info(f"Marking {symbol} as up-to-date in company_fundamentals")
        with adapter.session_scope() as session:
            company = session.query(CompanyFundamentalsTable).filter_by(symbol=symbol).first()
            if company:
                company.data_state = "up_to_date"
                company.last_update = datetime.now()
                session.commit()
                logger.info(f"{symbol} marked as up-to-date.")
            else:
                logger.warning(f"{symbol} not found in company_fundamentals.")
    except Exception as e:
        logger.error(f"Failed to update data_state for {symbol}: {e}")

def main(symbol):
    logging.basicConfig(level=logging.INFO)
    update_company_data(symbol)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python db_update_runner.py SYMBOL")
    else:
        main(sys.argv[1])