import os
from typing import List
import pandas as pd


class LoadExampleData:
    def __init__(self,
                 data_path: str = "/home/bandee/projects/marketIntelligence/dev_data/csvs",
                 load_company_fundamental: bool = False,
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
        self.load_company_fundamental = load_company_fundamental
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

            if self.load_company_fundamental and 'company_fundamental' in file:
                self.fundamentals = pd.read_csv(file_path)

            if self.load_daily_time_series and "daily_time_series" in file:
                self.daily_time_series = pd.read_csv(
                    file_path, parse_dates=["date"])

            if self.load_dividends and "dividends" in file:
                # , parse_dates=["ex_dividend_date", "declaration_date", "record_date", "payment_date"])
                self.dividends = pd.read_csv(file_path)

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


def create_desc_json_base(data_path="/home/bandee/projects/stockAnalyzer/dev_data",
                          out_file_name="alpha_vantage_column_description.json"):
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
            column_desc[key] = [{c: meaning}
                                for c in fundamentals.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'daily_time_series' in file:
            daily_time_series = pd.read_csv(
                f'{data_path}/{file}', parse_dates=['date'])
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in daily_time_series.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'dividends' in file:
            dividends = pd.read_csv(f'{data_path}/{file}', parse_dates=[
                                    'ex_dividend_date', 'declaration_date', 'record_date', 'payment_date'])
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in dividends.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'splits' in file:
            splits = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in splits.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'insider_transactions' in file:
            insider_transactions = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in insider_transactions.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'income_statement_quaterly' in file:
            income_statement_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [
                {c: meaning} for c in income_statement_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'income_statement_annual' in file:
            income_statement_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in income_statement_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'earnings_quaterly' in file:
            earnings_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in earnings_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'earnings_annual' in file:
            earnings_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in earnings_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'balance_sheet_quaterly' in file:
            balance_sheet_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in balance_sheet_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'balance_sheet_annual' in file:
            balance_sheet_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in balance_sheet_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'cash_flow_quaterly' in file:
            cash_flow_quaterly = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in cash_flow_quaterly.columns if c != 'Symbol']
            total_keys += len(column_desc[key])

        elif 'cash_flow_annual' in file:
            cash_flow_annual = pd.read_csv(f'{data_path}/{file}')
            meaning = ""
            column_desc[key] = [{c: meaning}
                                for c in cash_flow_annual.columns if c != 'Symbol']
            total_keys += len(column_desc[key])
    # Print the resulting dictionary
    print(column_desc)

    import json
    output_file = os.path.join(data_path, out_file_name)
    with open(output_file, "w") as f:
        json.dump(column_desc, f, indent=4)
