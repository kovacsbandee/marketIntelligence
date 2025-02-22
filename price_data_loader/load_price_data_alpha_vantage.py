import os
import requests
from dotenv import load_dotenv


import pandas as pd
load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY") 
SYMBOL = "IBM"
symbol = SYMBOL


def get_company_base(symbol: str = SYMBOL):
    """
    This could be updated when a report is arriving for a company...
    """
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()
    data_df = pd.DataFrame([data])
    latest_quarter = data_df.LatestQuarter.item()
    data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_company_fundamentals_lat_quart_{latest_quarter}.csv", index=False)


def get_daily_prices(symbol: str = SYMBOL):
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
    data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_daily_time_series.csv", index=False)


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


def income_statements(function: str, symbol: str = SYMBOL):
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
    annual_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_annual.csv", index=False)
    quaterly_df = pd.DataFrame(data=data["quarterlyReports"])
    quaterly_df["symbol"] = data["symbol"]
    quaterly_df["fiscalDateEnding"] = pd.to_datetime(quaterly_df["fiscalDateEnding"])
    quaterly_df.sort_values("fiscalDateEnding", inplace=True, ascending=True)
    quaterly_df = quaterly_df[['symbol'] + [col for col in quaterly_df.columns if col != 'symbol']]
    quaterly_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_{function.lower()}_quaterly.csv", index=False)


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

import os
import pandas as pd
from typing import List, Optional

class LoadExampleData:
    def __init__(self,
                 data_path: str = "/home/bandee/projects/marketIntelligence/dev_data",
                 company_fundamental: bool = False,
                 load_daily_time_series: bool = False,
                 load_dividends: bool = False,
                 load_splits: bool = False,
                 load_insider_transactions: bool = False,
                 load_income_statement_quarterly: bool = False,
                 load_income_statement_annual: bool = False,
                 load_earnings_quarterly: bool = False,
                 load_earnings_annual: bool = False,
                 load_balance_sheet_quarterly: bool = False,
                 load_balance_sheet_annual: bool = False,
                 load_cash_flow_quarterly: bool = False,
                 load_cash_flow_annual: bool = False) -> None:
        
        self.data_path: str = data_path
        self.files: List[str] = os.listdir(data_path),
        self.files: List[str] = os.listdir(self.data_path)
        self.company_fundamental = company_fundamental
        self.load_daily_time_series = load_daily_time_series
        self.load_dividends = load_dividends
        self.load_splits = load_splits
        self.load_insider_transactions = load_insider_transactions
        self.load_income_statement_quarterly = load_income_statement_quarterly
        self.load_income_statement_annual = load_income_statement_annual
        self.load_earnings_quarterly = load_earnings_quarterly
        self.load_earnings_annual = load_earnings_annual
        self.load_balance_sheet_quarterly = load_balance_sheet_quarterly
        self.load_balance_sheet_annual = load_balance_sheet_annual
        self.load_cash_flow_quarterly = load_cash_flow_quarterly
        self.load_cash_flow_annual = load_cash_flow_annual

    def toggle_all_flags(self) -> None:
        for attr_name in vars(self):
            if isinstance(getattr(self, attr_name), bool):
                setattr(self, attr_name, not getattr(self, attr_name))
    
    def load(self) -> None:
        """Loads CSV files from the data directory into corresponding attributes."""
        for file in self.files:
            file_path = f"{self.data_path}/{file}"
            
            if self.company_fundamental and 'company_fundamental':
                self.fundamentals = pd.read_csv(file_path)
            
            if self.load_daily_time_series and "daily_time_series" in file:
                self.daily_time_series = pd.read_csv(file_path, parse_dates=["date"])
            
            if self.load_dividends and "dividends" in file:
                self.dividends = pd.read_csv(file_path, parse_dates=["ex_dividend_date", "declaration_date", "record_date", "payment_date"])
            
            if self.load_splits and "splits" in file:
                self.splits = pd.read_csv(file_path)
            
            if self.load_insider_transactions and "insider_transactions" in file:
                self.insider_transactions = pd.read_csv(file_path)
            
            if self.load_income_statement_quarterly and "income_statement_quaterly" in file:
                self.income_statement_quarterly = pd.read_csv(file_path)

            if self.load_income_statement_annual and "income_statement_annual" in file:
                self.income_statement_annual = pd.read_csv(file_path)
            
            if self.load_earnings_quarterly and "earnings_quaterly" in file:
                self.earnings_quarterly = pd.read_csv(file_path)
            
            if self.load_earnings_annual and "earnings_annual" in file:
                self.earnings_annual = pd.read_csv(file_path)
            
            if self.load_balance_sheet_quarterly and "balance_sheet_quaterly" in file:
                self.balance_sheet_quaterly = pd.read_csv(file_path)
                
            if self.load_balance_sheet_annual and "balance_sheet_annual" in file:
                self.balance_sheet_annual = pd.read_csv(file_path)
            
            if self.load_cash_flow_quarterly and "cash_flow_quaterly" in file:
                self.cash_flow_quarterly = pd.read_csv(file_path)
            
            if self.load_cash_flow_annual and "cash_flow_annual" in file:
                self.cash_flow_annual = pd.read_csv(file_path)

# class LoadExampleData:

#     def __init__(self, data_path="/home/bandee/projects/marketIntelligence/dev_data"):
#         self.data_path = data_path
#         self.files = os.listdir(self.data_path)
#         self.company_fundamental = False

#     def load(self):
#         for file in self.files:
#             if self.company_fundamental:
#                 self.fundamentals = pd.read_csv(f'{self.data_path}/{file}')
#             if 'daily_time_series' in file:
#                 self.daily_time_series = pd.read_csv(f'{self.data_path}/{file}',
#                                                      parse_dates=['date'])
#             if 'dividends' in file:
#                 self.dividends = pd.read_csv(f'{self.data_path}/{file}',
#                                              parse_dates=['ex_dividend_date', 'declaration_date', 'record_date', 'payment_date'])
#             if 'splits' in file:
#                 self.splits = pd.read_csv(f'{self.data_path}/{file}')
#             if 'insider_transactions' in file:
#                 self.insider_transactions = pd.read_csv(f'{self.data_path}/{file}')#parse_dates=["transaction_date"])

#             if 'income_statement_quaterly' in file:
#                 self.income_statement_quaterly = pd.read_csv(f'{self.data_path}/{file}')
#             if 'income_statement_annual' in file:
#                 self.income_statement_annual = pd.read_csv(f'{self.data_path}/{file}')

#             if 'earnings_quaterly' in file:
#                 self.earnings_quaterly = pd.read_csv(f'{self.data_path}/{file}')
#             if 'earnings_annual' in file:
#                 self.earnings_annual = pd.read_csv(f'{self.data_path}/{file}')
            
#             if 'balance_sheet_quaterly' in file:
#                 self.balance_sheet_quaterly = pd.read_csv(f'{self.data_path}/{file}')
#             if 'balance_sheet_annual' in file:
#                 self.balance_sheet_annual = pd.read_csv(f'{self.data_path}/{file}')
            
#             if 'cash_flow_quaterly' in file:
#                 self.cash_flow_quaterly = pd.read_csv(f'{self.data_path}/{file}')
#             if 'cash_flow_annual' in file:
#                 self.cash_flow_annual = pd.read_csv(f'{self.data_path}/{file}')


def create_desc_json_base(data_path = "/home/bandee/projects/stockAnalyzer/dev_data"):
    column_desc = dict()
    total_keys = 0
    for file in os.listdir(data_path):
        # Strip prefix and suffix to generate dictionary key
        key = file.lstrip("IBM_").rstrip(".csv")
        print(key)
        
        # Process each file based on its type
        if 'company_fundamental' in file:
            fundamentals = pd.read_csv(f'{data_path}/{file}')
            meaning = ""  # Adding `meaning` as in the first row
            column_desc[key] = [{c: meaning} for c in fundamentals.columns if c != 'Symbol']
            total_keys += len(column_desc[key])
        
        elif 'daily_time_series' in file:
            daily_time_series = pd.read_csv(f'{data_path}/{file}', parse_dates=['date'])
            meaning = ""
            column_desc[key] = [{c: meaning} for c in daily_time_series.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'dividends' in file:
            dividends = pd.read_csv(f'{data_path}/{file}', parse_dates=['ex_dividend_date', 'declaration_date', 'record_date', 'payment_date'])
            meaning = ""
            column_desc[key] = [{c: meaning} for c in dividends.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'splits' in file:
            splits = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in splits.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'insider_transactions' in file:
            insider_transactions = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in insider_transactions.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'income_statement_quaterly' in file:
            income_statement_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in income_statement_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'income_statement_annual' in file:
            income_statement_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in income_statement_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'earnings_quaterly' in file:
            earnings_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in earnings_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'earnings_annual' in file:
            earnings_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in earnings_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'balance_sheet_quaterly' in file:
            balance_sheet_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in balance_sheet_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'balance_sheet_annual' in file:
            balance_sheet_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in balance_sheet_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'cash_flow_quaterly' in file:
            cash_flow_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in cash_flow_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'cash_flow_annual' in file:
            cash_flow_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning} for c in cash_flow_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])
    # Print the resulting dictionary
    print(column_desc)

    import json
    output_file = os.path.join(data_path, "alpha_vantage_column_description.json")
    with open(output_file, "w") as f:
        json.dump(column_desc, f, indent=4)
