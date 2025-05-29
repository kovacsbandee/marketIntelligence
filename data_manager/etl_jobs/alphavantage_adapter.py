
import requests
import pandas as pd

from configs.config import ALPHA_API_KEY
from data_manager.db_builders.postgre_adapter import PostgresAdapter
from data_manager.db_builders.postgre_objects import DailyTimeSeries, CompanyFundamentalsTable

# TODO: ADD  ROW FOR EACH DATA HANDLER TO CHECK FOR DATABASE COMPLIANCE AND DATA QUALITY
# e.g. missing values, and add a logger for these stuff!

class AlphaLoader:

    def __init__(self, symbol: str, db_mode: bool = False, local_store_mode: bool = False):
        self.symbol = symbol.upper()
        self.db_mode = db_mode
        self.local_store_mode = local_store_mode
        self.base_url = "https://www.alphavantage.co/query?function="
        self.local_store_path = "/home/bandee/projects/marketIntelligence/dev_data/jsons"


    def get_daily_timeseries(self):
        url = f"{self.base_url}TIME_SERIES_DAILY&outputsize=full&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        print(f"Fetching data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            #data = json.load(open(f'{self.local_store_path}/{self.symbol}_daily_time_series.json'))
            # Handle API limit or error response
            if "Time Series (Daily)" not in data:
                print(f"❌ Error in API response for {self.symbol}: {data.get('Note') or data}")
                return

            data_df = pd.DataFrame.from_dict(data=data["Time Series (Daily)"], orient='index')
            data_df.columns = ['open', 'high', 'low', 'close', 'volume']
            data_df.index = pd.to_datetime(data_df.index)
            data_df.sort_index(ascending=True, axis=0, inplace=True)
            data_df.dropna(inplace=True)
            data_df.reset_index(inplace=True, names=["date"])
            data_df["symbol"] = self.symbol

            # Reorder columns
            data_df = data_df[["date", "symbol", "open", "high", "low", "close", "volume"]]

            if self.db_mode:
                adapter = PostgresAdapter()
                data_df_rows = data_df.to_dict(orient="records")
                adapter.insert_new_data(table=DailyTimeSeries, rows=data_df_rows)
                print(f"✅ Candlestick data for {self.symbol} loaded into the database.")

            if self.local_store_mode:
                output_path = f"{self.local_store_path}/{self.symbol}_daily_time_series.csv"
                data_df.to_csv(output_path, index=False)
                print(f"✅ Data saved locally: {output_path}")

            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Chose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")


    def get_company_base(self):
        """
        Fetch company base information from Alpha Vantage API and store it.
        This could be updated when a report is arriving for a company...
        """
        url = f'{self.base_url}OVERVIEW&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        print(f"Fetching company base data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            #file = [f for f in os.listdir(self.local_store_path) if 'company_fundamentals' in f][0]
            #data = json.load(open(f'{self.local_store_path}/{file}'))

            # Handle error in API response
            if "Error Message" in data or "Note" in data:
                print(f"❌ Error in API response for {self.symbol}: {data.get('Note') or data}")
                return
            # Convert the response into a DataFrame
            data_df = pd.DataFrame([data])
            
            # converting the datatypes:
            data_df.replace("None", pd.NA, inplace=True)
            data_df = data_df.apply(pd.to_numeric, errors='ignore')

            from data_manager.etl_jobs.transform_utils import standardize_company_fundamentals_columns
            data_df = standardize_company_fundamentals_columns(data_df)

            # Save data into the database if db_mode is enabled
            if self.db_mode:
                adapter = PostgresAdapter()
                data_df_rows = data_df.to_dict(orient="records")
                for d in data_df_rows:
                    for k, v in d.items():
                        if pd.isna(v):
                            d[k] = None
                adapter.insert_new_data(table=CompanyFundamentalsTable, rows=data_df_rows)
                print(f"✅ Company base data for {self.symbol} loaded into the database.")

            # Save data locally as a CSV if local_store_mode is enabled
            if self.local_store_mode:
                output_path = f"{self.local_store_path}/{self.symbol}_company_fundamentals_lat_quart_{latest_quarter}.csv"
                data_df.to_csv(output_path, index=False)
                print(f"✅ Company base data saved locally: {output_path}")

            # If neither db_mode nor local_store_mode is enabled, prompt the user
            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Chose a place where you want to store the company base data!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")


    def get_financials(self, function: str):
        """
        Fetch financial data (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, or EARNINGS) for the given symbol.
        """
        url = f'{self.base_url}{function}&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        print(f"Fetching {function} data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()

            # === 1. Build DataFrames ===
            if function == 'EARNINGS':
                # EARNINGS uses different top-level keys!
                annual_key = "annualEarnings"
                quarterly_key = "quarterlyEarnings"
                if annual_key not in data or quarterly_key not in data:
                    print(f"❌ Error in API response for {self.symbol}: {data.get('Note') or data}")
                    return
                annual_df = pd.DataFrame(data[annual_key])
                quarterly_df = pd.DataFrame(data[quarterly_key])
                # Attach symbol if not present
                annual_df["symbol"] = data["symbol"]
                quarterly_df["symbol"] = data["symbol"]
            else:
                if "annualReports" not in data or "quarterlyReports" not in data:
                    print(f"❌ Error in API response for {self.symbol}: {data.get('Note') or data}")
                    return
                annual_df = pd.DataFrame(data["annualReports"])
                annual_df["symbol"] = data["symbol"]

                quarterly_df = pd.DataFrame(data["quarterlyReports"])
                quarterly_df["symbol"] = data["symbol"]

            # === 2. Standardize and type-cast columns ===
            if function == 'INCOME_STATEMENT':
                from data_manager.etl_jobs.transform_utils import (
                    standardize_annual_income_statement_columns, standardize_quarterly_income_statement_columns
                )
                annual_df = standardize_annual_income_statement_columns(annual_df)
                quarterly_df = standardize_quarterly_income_statement_columns(quarterly_df)
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                from data_manager.db_builders.postgre_objects import (
                    AnnualIncomeStatement as AnnualTable, QuarterlyIncomeStatement as QuarterlyTable
                )
            elif function == 'BALANCE_SHEET':
                from data_manager.etl_jobs.transform_utils import (
                    standardize_annual_balance_sheet_columns, standardize_quarterly_balance_sheet_columns
                )
                annual_df = standardize_annual_balance_sheet_columns(annual_df)
                quarterly_df = standardize_quarterly_balance_sheet_columns(quarterly_df)
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                from data_manager.db_builders.postgre_objects import (
                    AnnualBalanceSheetTable as AnnualTable, QuarterlyBalanceSheetTable as QuarterlyTable
                )
            elif function == 'CASH_FLOW':
                from data_manager.etl_jobs.transform_utils import (
                    standardize_annual_cash_flow_columns, standardize_quarterly_cash_flow_columns
                )
                annual_df = standardize_annual_cash_flow_columns(annual_df)
                quarterly_df = standardize_quarterly_cash_flow_columns(quarterly_df)
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                from data_manager.db_builders.postgre_objects import (
                    AnnualCashFlowTable as AnnualTable, QuarterlyCashFlowTable as QuarterlyTable
                )
            elif function == 'EARNINGS':
                from data_manager.etl_jobs.transform_utils import (
                    standardize_annual_earnings_columns, standardize_quarterly_earnings_columns
                )
                annual_df = standardize_annual_earnings_columns(annual_df)
                quarterly_df = standardize_quarterly_earnings_columns(quarterly_df)
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                from data_manager.db_builders.postgre_objects import (
                    AnnualEarningsTable as AnnualTable, QuarterlyEarningsTable as QuarterlyTable
                )
            else:
                print(f"❌ Unknown function '{function}'")
                return

            # === 3. Final cleanup (drop rows with null PKs, reindex) ===
            for df, pk in [(annual_df, annual_pk), (quarterly_df, quarterly_pk)]:
                if pk in df.columns:
                    df.dropna(subset=[pk], inplace=True)

            # === 4. Save to DB if enabled ===
            if self.db_mode:
                adapter = PostgresAdapter()
                adapter.insert_new_data(table=AnnualTable, rows=annual_df.to_dict(orient="records"))
                adapter.insert_new_data(table=QuarterlyTable, rows=quarterly_df.to_dict(orient="records"))
                print(f"✅ {function} data for {self.symbol} loaded into the database.")

            # === 5. Save as CSV locally if enabled ===
            if self.local_store_mode:
                annual_df.to_csv(f"{self.local_store_path}/{self.symbol}_{function.lower()}_annual.csv", index=False)
                quarterly_df.to_csv(f"{self.local_store_path}/{self.symbol}_{function.lower()}_quaterly.csv", index=False)
                print(f"✅ {function} data saved locally for {self.symbol}.")

            # === 6. Warn if nothing is enabled ===
            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

    def get_insider_transactions(self):
        """
        Fetch and store insider transactions for the given symbol.
        """
        url = f"{self.base_url}INSIDER_TRANSACTIONS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        print(f"Fetching INSIDER_TRANSACTIONS data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            df = pd.DataFrame(data)
            df["symbol"] = self.symbol

            from data_manager.etl_jobs.transform_utils import standardize_insider_transaction_columns
            df = standardize_insider_transaction_columns(df)
            if "transaction_date" not in df.columns:
                print("❌ 'transaction_date' column missing after standardization. Columns are:", df.columns)
                return
            df.dropna(subset=["transaction_date", "symbol"], inplace=True)

            if self.db_mode:
                from data_manager.db_builders.postgre_objects import InsiderTransactionTable
                adapter = PostgresAdapter()
                adapter.insert_new_data(table=InsiderTransactionTable, rows=df.to_dict(orient="records"))
                print(f"✅ Insider transaction data for {self.symbol} loaded into the database.")

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_insider_transactions.csv", index=False)
                print(f"✅ Insider transaction data saved locally for {self.symbol}.")

            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
    
    def get_stock_splits(self):
        """
        Fetch and store stock split data for the given symbol.
        """
        url = f"{self.base_url}SPLITS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        print(f"Fetching STOCK_SPLITS data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            print(data)
            # If the response is a dict with the splits in a key
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            # Attach the symbol to each row if missing
            for row in data:
                if "symbol" not in row:
                    row["symbol"] = self.symbol

            df = pd.DataFrame(data)
            from data_manager.etl_jobs.transform_utils import standardize_stock_split_columns
            df = standardize_stock_split_columns(df)
            df.dropna(subset=["symbol", "effective_date"], inplace=True)

            if self.db_mode:
                from data_manager.db_builders.postgre_objects import StockSplit
                adapter = PostgresAdapter()
                adapter.insert_new_data(table=StockSplit, rows=df.to_dict(orient="records"))
                print(f"✅ Stock split data for {self.symbol} loaded into the database.")

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_stock_splits.csv", index=False)
                print(f"✅ Stock split data saved locally for {self.symbol}.")

            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

    def get_dividends(self):
        """
        Fetch and store dividend data for the given symbol.
        """
        url = f"{self.base_url}DIVIDENDS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        print(f"Fetching DIVIDENDS data for {self.symbol}...")

        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()

            # If the response is a dict with dividends in a key
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            # Attach symbol if missing
            for row in data:
                if "symbol" not in row:
                    row["symbol"] = self.symbol

            df = pd.DataFrame(data)
            from data_manager.etl_jobs.transform_utils import standardize_dividends_columns
            df = standardize_dividends_columns(df)
            df.dropna(subset=["symbol", "ex_dividend_date"], inplace=True)

            if self.db_mode:
                from data_manager.db_builders.postgre_objects import DividendsTable
                adapter = PostgresAdapter()
                adapter.insert_new_data(table=DividendsTable, rows=df.to_dict(orient="records"))
                print(f"✅ Dividend data for {self.symbol} loaded into the database.")

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_dividends.csv", index=False)
                print(f"✅ Dividend data saved locally for {self.symbol}.")

            if not self.db_mode and not self.local_store_mode:
                print('⚠️  Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {self.symbol}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")





# def get_time_series_intraday(month: str, symbol: str = SYMBOL, interval: str='1min'):
#     """    
#         month is in YYYY-MM format   
#     """

#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&month={month}&outputsize=full&apikey={ALPHA_API_KEY}'
#     r = requests.get(url)
#     data = r.json()
#     data_df = pd.DataFrame.from_dict(data=data["Time Series (1min)"], orient='index')
#     data_df.columns = ['open', 'high', 'low', 'close', 'volume']
#     data_df.index = pd.to_datetime(data_df.index)
#     data_df.sort_index(ascending=True, axis=0, inplace=True)
#     data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_intraday_{month}.csv", index=True)
#     print(f"{symbol} for {month} was successfully written out to trash_data!")