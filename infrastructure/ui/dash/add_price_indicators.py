import pandas as pd


class AddPriceIndicators:

    def __init__(self,
                 table: pd.DataFrame = None,
                 price_col: str = "close"):

        self.table = table
        self.price_col = price_col
        self.rsi_period = None

    def add_macd(self,
                 macd_short_period=12,
                 macd_long_period=26,
                 macd_signal_period=9,
                 clean_intermediates=True):
        """
            Adds MACD (Moving Average Convergence Divergence) to the table.

            Parameters:
            macd_short_period (int): Short-term EMA period for MACD calculation (default is 12).
            macd_long_period (int): Long-term EMA period for MACD calculation (default is 26).
            macd_signal_period (int): Signal line EMA period (default is 9).
            clean_intermediates (bool): Whether to drop intermediate columns after calculation.

            Returns:
            pd.DataFrame: The updated table with MACD and optional intermediate columns.
        """
        # Compute EMAs
        self.table[f"EMA_{macd_short_period}"] = (
            self.table[self.price_col].ewm(
                span=macd_short_period, adjust=False).mean()
        )
        self.table[f"EMA_{macd_long_period}"] = (
            self.table[self.price_col].ewm(
                span=macd_long_period, adjust=False).mean()
        )

        # Compute MACD Line
        self.table["MACD_line"] = (
            self.table[f"EMA_{macd_short_period}"]
            - self.table[f"EMA_{macd_long_period}"]
        )

        # Compute Signal Line
        self.table["signal_line"] = (
            self.table["MACD_line"].ewm(
                span=macd_signal_period, adjust=False).mean()
        )

        # Compute MACD Histogram
        self.table["MACD_histogram"] = (
            self.table["MACD_line"] - self.table["signal_line"]
        )

        # Log parameters
        print("MACD parameters:")
        print(f"Short-term EMA: {macd_short_period}")
        print(f"Long-term EMA: {macd_long_period}")
        print(f"Signal line EMA: {macd_signal_period}")
        print(
            f"Intermediate columns {'dropped' if clean_intermediates else 'kept'}.")

        # Optionally clean intermediate columns
        if clean_intermediates:
            self.table.drop(
                columns=[f"EMA_{macd_short_period}",
                         f"EMA_{macd_long_period}"],
                inplace=True,
            )

        return self.table

    def add_rsi(self,
                rsi_period=14,
                clean_intermediates=True):
        """
        Adds RSI (Relative Strength Index) to the table.

        Parameters:
        rsi_period (int): Period for RSI calculation (default is 14).
        clean_intermediates (bool): Whether to drop intermediate columns after calculation.

        Returns:
        pd.DataFrame: The updated table with RSI and optional intermediate columns.
        """
        # Compute price differences
        self.table["price_diff"] = self.table[self.price_col].diff()

        # Compute gains and losses
        self.table["gain"] = self.table["price_diff"].where(
            self.table["price_diff"] > 0, 0
        )
        self.table["loss"] = -self.table["price_diff"].where(
            self.table["price_diff"] < 0, 0
        )

        # Compute average gain and average loss
        self.table["avg_gain"] = (
            self.table["gain"].ewm(span=rsi_period, adjust=False).mean()
        )
        self.table["avg_loss"] = (
            self.table["loss"].ewm(span=rsi_period, adjust=False).mean()
        )

        # Compute Relative Strength (RS)
        self.table["RS"] = self.table["avg_gain"] / self.table["avg_loss"]

        # Compute RSI
        self.table["RSI"] = 100 - (100 / (1 + self.table["RS"]))

        # Log parameters
        print("RSI parameters:")
        print(f"RSI period: {rsi_period}")
        print(
            f"Intermediate columns {'dropped' if clean_intermediates else 'kept'}.")

        # Optionally clean intermediate columns
        if clean_intermediates:
            self.table.drop(
                columns=["price_diff", "gain", "loss",
                         "avg_gain", "avg_loss", "RS"],
                inplace=True,
            )

        return self.table

    def add_indicator(self,
                      macd=True):
        if macd:
            self.add_macd()
        print("Adding indicators was successfull!")
