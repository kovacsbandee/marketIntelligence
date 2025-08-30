import pandas as pd
from symbol.symbol import Symbol
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyDataManager
from infrastructure.ui.dash.app_util import get_last_6_months_range

def load_symbol_data(symbol: str):
    """
    Load all relevant financial data for a given stock symbol.

    This function initializes the database adapter and uses the Symbol class to retrieve
    all necessary tables for the specified symbol. It returns a dictionary containing
    the price data, dividends, company fundamentals, annual and quarterly balance sheets,
    and a default date range (last 6 months) for the price data.

    Args:
        symbol (str): The stock symbol to load data for.

    Returns:
        dict: A dictionary with the following keys:
            - status_message (str): Status or error message from the Symbol loader.
            - daily_timeseries (pd.DataFrame or None): Daily price time series.
            - dividends (pd.DataFrame or None): Dividend data.
            - company_fundamentals (pd.DataFrame or None): Company fundamentals.
            - annual_balance_sheet (pd.DataFrame or None): Annual balance sheet.
            - quarterly_balance_sheet (pd.DataFrame or None): Quarterly balance sheet.
            - start_date (str): Start date for the default 6-month range.
            - end_date (str): End date for the default 6-month range.
    """
    adapter = CompanyDataManager()
    storage = Symbol(adapter, symbol.upper(), add_price_indicators=True)
    daily_timeseries = storage.get_table("daily_timeseries")
    if daily_timeseries is None or daily_timeseries.empty:
        return {
            "status_message": storage.status_message,
            "daily_timeseries": None,
            "dividends": None,
            "company_fundamentals": None,
            "annual_balance_sheet": None,
            "balance_sheet_quarterly": None,
        }
    dividends = storage.get_table("dividends")
    company_fundamentals = storage.get_table("company_fundamentals")
    balance_sheet_annual = storage.get_table("balance_sheet_annual")
    balance_sheet_quarterly = storage.get_table("balance_sheet_quarterly")
    earnings = storage.get_table("earnings_quarterly")
    income_statement_quarterly = storage.get_table("income_statement_quarterly")
    cashflow_statement_quarterly = storage.get_table("cash_flow_quarterly")
    print("Debug: Loaded cash flow data:", cashflow_statement_quarterly)
    start_date, end_date = get_last_6_months_range(daily_timeseries)
    return {
        "status_message": storage.status_message,
        "daily_timeseries": daily_timeseries,
        "dividends": dividends,
        "company_fundamentals": company_fundamentals,
        "annual_balance_sheet": balance_sheet_annual,
        "balance_sheet_quarterly": balance_sheet_quarterly,
        "earnings": earnings,
        "income_statement_quarterly": income_statement_quarterly,
        "cashflow_statement_quarterly": cashflow_statement_quarterly,
        "start_date": start_date,
        "end_date": end_date,
    }

