from sqlalchemy import Column, Integer, BigInteger, String, Float, Date, Sequence, DateTime, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

# --- Quantitative Analyst Tables ---
class AnalystQuantitativeBase(Base):
    """
    ORM osztály a kvantitatív elemzői összesített eredmények tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - last_date (DateTime, nullable=True): Az utolsó adat dátuma.
        - num_data_points (Integer, nullable=True): Felhasznált adatpontok száma.
        - obv_used (Integer, nullable=False): OBV indikátor használva (1/0).
        - vwap_used (Integer, nullable=False): VWAP indikátor használva (1/0).
        - macd_used (Integer, nullable=False): MACD indikátor használva (1/0).
        - rsi_used (Integer, nullable=False): RSI indikátor használva (1/0).
        - stochastic_oscillator_used (Integer, nullable=False): Stochastic Oscillator indikátor használva (1/0).
        - bollinger_bands_used (Integer, nullable=False): Bollinger Bands indikátor használva (1/0).
        - sma_used (Integer, nullable=False): SMA indikátor használva (1/0).
        - ema_used (Integer, nullable=False): EMA indikátor használva (1/0).
        - analysis_run_timestamp (BigInteger, nullable=True): Elemzés futtatásának időbélyege.
        - analysis_run_datetime (DateTime, nullable=True): Elemzés futtatásának dátuma.
        - warnings (String, nullable=True): Figyelmeztetések (JSON string vagy vesszővel elválasztott).
        - errors (String, nullable=True): Hibák (JSON string vagy vesszővel elválasztott).
        - aggregate_score (Float, nullable=True): Az összesített kvantitatív pontszám.
    """
    __tablename__ = 'analyst_quantitative_base'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    last_date = Column(DateTime, nullable=True)
    num_data_points = Column(Integer, nullable=True)
    obv_used = Column(Integer, nullable=False, default=0)
    vwap_used = Column(Integer, nullable=False, default=0)
    macd_used = Column(Integer, nullable=False, default=0)
    rsi_used = Column(Integer, nullable=False, default=0)
    stochastic_oscillator_used = Column(Integer, nullable=False, default=0)
    bollinger_bands_used = Column(Integer, nullable=False, default=0)
    sma_used = Column(Integer, nullable=False, default=0)
    ema_used = Column(Integer, nullable=False, default=0)
    analysis_run_timestamp = Column(BigInteger, nullable=True)
    analysis_run_datetime = Column(DateTime, nullable=True)
    warnings = Column(String, nullable=True)
    errors = Column(String, nullable=True)
    aggregate_score = Column(Float, nullable=True)

class AnalystQuantitativeScore(Base):
    """
    ORM osztály a kvantitatív elemzői indikátor pontszámok tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - aggregate_score (Float, nullable=True): Az összesített kvantitatív pontszám.
        - obv (Float, nullable=True): On-Balance Volume pontszám.
        - vwap (Float, nullable=True): Volume-Weighted Average Price pontszám.
        - macd (Float, nullable=True): MACD pontszám.
        - rsi (Float, nullable=True): RSI pontszám.
        - stochastic_oscillator (Float, nullable=True): Stochastic Oscillator pontszám.
        - bollinger_bands (Float, nullable=True): Bollinger Bands pontszám.
        - sma (Float, nullable=True): Simple Moving Average pontszám.
        - ema (Float, nullable=True): Exponential Moving Average pontszám.
    """
    __tablename__ = 'analyst_quantitative_score'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    aggregate_score = Column(Float, nullable=True)
    obv = Column(Float, nullable=True)
    vwap = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    stochastic_oscillator = Column(Float, nullable=True)
    bollinger_bands = Column(Float, nullable=True)
    sma = Column(Float, nullable=True)
    ema = Column(Float, nullable=True)

class AnalystQuantitativeExplanation(Base):
    """
    ORM osztály a kvantitatív elemzői indikátor magyarázatok tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - obv (String, nullable=True): On-Balance Volume magyarázat.
        - vwap (String, nullable=True): Volume-Weighted Average Price magyarázat.
        - macd (String, nullable=True): MACD magyarázat.
        - rsi (String, nullable=True): RSI magyarázat.
        - stochastic_oscillator (String, nullable=True): Stochastic Oscillator magyarázat.
        - bollinger_bands (String, nullable=True): Bollinger Bands magyarázat.
        - sma (String, nullable=True): Simple Moving Average magyarázat.
        - ema (String, nullable=True): Exponential Moving Average magyarázat.
    """
    __tablename__ = 'analyst_quantitative_explanation'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    obv = Column(String, nullable=True)
    vwap = Column(String, nullable=True)
    macd = Column(String, nullable=True)
    rsi = Column(String, nullable=True)
    stochastic_oscillator = Column(String, nullable=True)
    bollinger_bands = Column(String, nullable=True)
    sma = Column(String, nullable=True)
    ema = Column(String, nullable=True)

analyst_table_name_to_class = {
    "analyst_quantitative_base": AnalystQuantitativeBase,
    "analyst_quantitative_score": AnalystQuantitativeScore,
    "analyst_quantitative_explanation": AnalystQuantitativeExplanation,
}