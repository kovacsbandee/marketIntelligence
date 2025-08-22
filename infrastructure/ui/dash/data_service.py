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
            - price_df (pd.DataFrame or None): Daily price time series.
            - dividends_df (pd.DataFrame or None): Dividend data.
            - company_base_df (pd.DataFrame or None): Company fundamentals.
            - annual_balance_sheet_df (pd.DataFrame or None): Annual balance sheet.
            - quarterly_balance_sheet_df (pd.DataFrame or None): Quarterly balance sheet.
            - start_date (str): Start date for the default 6-month range.
            - end_date (str): End date for the default 6-month range.
    """
    adapter = CompanyDataManager()
    storage = Symbol(adapter, symbol.upper())
    price_df = storage.get_table("daily_timeseries")
    if price_df is None or price_df.empty:
        return {
            "status_message": storage.status_message,
            "price_df": None,
            "dividends_df": None,
            "company_base_df": None,
            "annual_balance_sheet_df": None,
            "quarterly_balance_sheet_df": None,
        }
    dividends_df = storage.get_table("dividends")
    company_base_df = storage.get_table("company_fundamentals")
    annual_balance_sheet_df = storage.get_table("balance_sheet_annual")
    quarterly_balance_sheet_df = storage.get_table("balance_sheet_quarterly")
    start_date, end_date = get_last_6_months_range(price_df)
    return {
        "status_message": storage.status_message,
        "price_df": price_df,
        "dividends_df": dividends_df,
        "company_base_df": company_base_df,
        "annual_balance_sheet_df": annual_balance_sheet_df,
        "quarterly_balance_sheet_df": quarterly_balance_sheet_df,
        "start_date": start_date,
        "end_date": end_date,
    }