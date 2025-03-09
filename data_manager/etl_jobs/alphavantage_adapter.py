import os
import requests
from dotenv import load_dotenv
import pandas as pd

from data_manager.build_db.postgre_adapter import PostgresAdapter
from data_manager.build_db.postgre_objects import CandlestickTable

load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY") 
SYMBOL = "TSLA"
#symbol = SYMBOL

def get_daily_candlestick(symbol: str = SYMBOL, 
                          db_mode: bool = False,
                          local_store_mode: bool = False):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    data_df = pd.DataFrame.from_dict(data=data["Time Series (Daily)"], orient='index')
    data_df.columns = ['open', 'high', 'low', 'close', 'volume']
    data_df.index = pd.to_datetime(data_df.index)
    data_df.sort_index(ascending=True, axis=0, inplace=True)
    data_df.reset_index(inplace=True, names=["date"])
    data_df["symbol"] = symbol
    print(data_df.columns)
    data_df = data_df[["date","symbol", "open", "high", "low", "close", "volume"]]
    if db_mode:
        adapter = PostgresAdapter()
        data_df_rows = data_df.to_dict(orient="records")
        print(data_df_rows)
        adapter.insert_new_data(table=CandlestickTable, rows=data_df_rows)
        print(f'Candlestick data for {symbol} is loaded into the database.')
    if local_store_mode:
        data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_daily_time_series.csv", index=False)
    if not db_mode and not local_store_mode:
        print('Chose a place where you want to store the data from the API!')


def get_company_base(symbol: str = SYMBOL, 
                     db_mode: bool = False,
                     local_store_mode: bool = False):
    """
    This could be updated when a report is arriving for a company...
    """
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    data_df = pd.DataFrame([data])
    latest_quarter = data_df.LatestQuarter.item()
    if db_mode:
        adapter = PostgresAdapter()
        data_df_rows = data_df.to_dict(orient="records")
        print(data_df_rows)
        adapter.insert_new_data(table=CandlestickTable, rows=data_df_rows)
        print(f'Candlestick data for {symbol} is loaded into the database.')
    if local_store_mode:
        data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_daily_time_series.csv", index=False)
    if not db_mode and not local_store_mode:
        print('Chose a place where you want to store the data from the API!')
    data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_company_fundamentals_lat_quart_{latest_quarter}.csv", index=False)




def corp_actions(symbol: str = SYMBOL, function: str = 'DIVIDENDS'):
    """
    function can be DIVIDENDS or SPLITS
    """
    function = 'SPLITS'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    data_df = pd.DataFrame(data=data["data"])
    data_df["symbol"] = data["symbol"]
    if function == 'DIVIDENDS':
        for c in data_df.columns:
            if "date" in c:
                data_df[c] = pd.to_datetime(data_df[c], errors="coerce")
        data_df.sort_values("ex_dividend_date", inplace=True, ascending=True)
        data_df = data_df[['symbol', 'amount', 'ex_dividend_date', 'declaration_date', 'record_date', 'payment_date',]]
    if function == "SPLITS":
        data_df = data_df[['symbol', 'effective_date', 'split_factor']]
    data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}.csv", index=False)


def get_time_series_intraday(month: str, symbol: str = SYMBOL, interval: str='1min'):
    """    month is in YYYY-MM format   """

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&month={month}&outputsize=full&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    data_df = pd.DataFrame.from_dict(data=data["Time Series (1min)"], orient='index')
    data_df.columns = ['open', 'high', 'low', 'close', 'volume']
    data_df.index = pd.to_datetime(data_df.index)
    data_df.sort_index(ascending=True, axis=0, inplace=True)
    data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_intraday_{month}.csv", index=True)
    print(f"{symbol} for {month} was successfully written out to trash_data!")


def financials(function: str, 
               symbol: str = SYMBOL,
               db_mode: bool = False,
               local_store_mode: bool = False):
    """
    function can be INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW
    """
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    annual_df = pd.DataFrame(data=data["annualReports"])
    annual_df["symbol"] = data["symbol"]
    annual_df["fiscalDateEnding"] = pd.to_datetime(annual_df["fiscalDateEnding"])
    annual_df.sort_values("fiscalDateEnding", inplace=True, ascending=True)
    annual_df = annual_df[['symbol'] + [col for col in annual_df.columns if col != 'symbol']]
    
    quaterly_df = pd.DataFrame(data=data["quarterlyReports"])
    quaterly_df["symbol"] = data["symbol"]
    quaterly_df["fiscalDateEnding"] = pd.to_datetime(quaterly_df["fiscalDateEnding"])
    quaterly_df.sort_values("fiscalDateEnding", inplace=True, ascending=True)
    quaterly_df = quaterly_df[['symbol'] + [col for col in quaterly_df.columns if col != 'symbol']]
    if db_mode:
        adapter = PostgresAdapter()
        annual_data_df_rows = annual_df.to_dict(orient="records")
        if function == 'INCOME_STATEMENT':

            from data_manager.build_db.postgre_objects import AnnualIncomeStatement as AIS
            adapter.insert_new_data(table=AIS, rows=annual_data_df_rows)

            from data_manager.build_db.postgre_objects import QuarterlyIncomeStatement as QIS
            adapter.insert_new_data(table=QIS, rows=annual_data_df_rows)

        if function == 'BALANCE_SHEET':

            from data_manager.build_db.postgre_objects import AnnualBalanceSheetTable as ABS
            adapter.insert_new_data(table=ABS, rows=annual_data_df_rows)

            from data_manager.build_db.postgre_objects import QuarterlyBalanceSheetTable as QBS
            adapter.insert_new_data(table=QBS, rows=annual_data_df_rows)

        if function == 'CASH_FLOW':

            from data_manager.build_db.postgre_objects import AnnualCashFlowTable as ACF
            adapter.insert_new_data(table=ACF, rows=annual_data_df_rows)

            from data_manager.build_db.postgre_objects import QuarterlyCashFlowTable as QCF
            adapter.insert_new_data(table=QCF, rows=annual_data_df_rows)

        print(f'Candlestick data for {symbol} is loaded into the database.')
    if local_store_mode:
        annual_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_annual.csv", index=False)
        quaterly_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_quaterly.csv", index=False)
    if not db_mode and not local_store_mode:
        print('Chose a place where you want to store the data from the API!')


def earnings(function: str, symbol: str = SYMBOL):
    """
    function can be EARNINGS
    """
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    annual_df = pd.DataFrame(data=data["annualEarnings"])
    annual_df["symbol"] = data["symbol"]
    annual_df["fiscalDateEnding"] = pd.to_datetime(annual_df["fiscalDateEnding"])
    annual_df.sort_values("fiscalDateEnding", inplace=True, ascending=True)
    annual_df = annual_df[['symbol'] + [col for col in annual_df.columns if col != 'symbol']]
    annual_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_annual.csv", index=False)

    quaterly_df = pd.DataFrame(data=data["quarterlyEarnings"])
    quaterly_df["symbol"] = data["symbol"]
    quaterly_df["fiscalDateEnding"] = pd.to_datetime(quaterly_df["fiscalDateEnding"])
    quaterly_df.sort_values("fiscalDateEnding", inplace=True, ascending=True)
    quaterly_df = quaterly_df[['symbol'] + [col for col in quaterly_df.columns if col != 'symbol']]
    quaterly_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_quaterly.csv", index=False)


def get_insider_transactions(symbol: str = SYMBOL):
    symbol = SYMBOL
    url = f'https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    insider_df = pd.DataFrame(data=data["data"])
    insider_df = insider_df[['ticker', 'transaction_date', 'executive', 'executive_title', 'security_type', 'acquisition_or_disposal', 'shares', 'share_price']]
    insider_df.rename(columns={'ticker': 'symbol'}, inplace=True)
    insider_df["executive"] = insider_df["executive"].str.replace(',', '')
    insider_df["executive_title"] = insider_df["executive_title"].str.replace(',', '')
    insider_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_insider_transactions.csv", index=False)
