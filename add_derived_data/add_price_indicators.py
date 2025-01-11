import os
from dotenv import load_dotenv

import pandas as pd
from pandas import DataFrame

class addPriceIndicators:

    def __init__(self, table: DataFrame = None):
        self.table = table
        self.macd_short_period = None
        self.macd_long_period = None
        self.rsi_period = None
        pass

    def macd(self):
        pass

    def rsi(self):
        pass

    def add_indicator(self):
        pass