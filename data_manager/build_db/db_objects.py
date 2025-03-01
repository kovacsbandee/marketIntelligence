import uuid
from typing import TypedDict
from sqlalchemy import Column, Integer, String, Float, Date, DATE
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class CandlestickTable(Base):
    """
    ORM class representing daily candlestick data for stock symbols.

    Primary Key:
        - date (DATE): The trading date of the candlestick.
        - symbol (String): The stock ticker symbol.

    Fields:
        - open (Float, optional): The opening price of the stock on the given date.
        - high (Float, optional): The highest price of the stock during the trading session.
        - low (Float, optional): The lowest price of the stock during the trading session.
        - close (Float, optional): The closing price of the stock on the given date.
        - volume (Integer, optional): The total number of shares traded during the session.

    Notes:
        - The `date` and `symbol` combination ensures unique records for each stock per day.
        - `open`, `high`, `low`, `close`, and `volume` values can be `NULL` if data is unavailable.
    """
    __tablename__ = 'daily_candlestick'
    date = Column(DATE, nullable=False, primary_key=True)
    symbol = Column(String, nullable=False, primary_key=True)
    open = Column(Float(precision=2), nullable=True)
    high = Column(Float(precision=2), nullable=True)
    low = Column(Float(precision=2), nullable=True)
    close = Column(Float(precision=2), nullable=True)
    volume = Column(Integer, nullable=True)


class DividendTable(Base):
    """
    ORM class representing dividend payments for stocks.

    Primary Key:
        - symbol
        - ex_dividend_date (since each symbol can have multiple dividend payments)

    Fields:
        - symbol (str): The stock ticker symbol.
        - amount (float): The dividend amount.
        - ex_dividend_date (Date): The date when the stock starts trading without the dividend.
        - declaration_date (Date, optional): The date when the dividend was declared.
        - record_date (Date, optional): The date when shareholders must own the stock to receive the dividend.
        - payment_date (Date, optional): The date when the dividend is paid out.
    """
    __tablename__ = 'dividends'

    symbol = Column(String, primary_key=True, nullable=False)
    ex_dividend_date = Column(Date, primary_key=True, nullable=False)  # Unique per symbol
    amount = Column(Float, nullable=False)
    declaration_date = Column(Date, nullable=True)
    record_date = Column(Date, nullable=True)
    payment_date = Column(Date, nullable=True)
