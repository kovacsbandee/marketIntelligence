import pandas as pd

def get_last_6_months_range(price_df: pd.DataFrame):
    end_date_val = price_df["date"].max()
    start_date_val = end_date_val - pd.Timedelta(days=182)
    start_date = start_date_val.strftime("%Y-%m-%d")
    end_date = end_date_val.strftime("%Y-%m-%d")
    return start_date, end_date
