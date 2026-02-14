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
    adx_used = Column(Integer, nullable=False, default=0)
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
    adx = Column(Float, nullable=True)

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
    adx = Column(String, nullable=True)

# --- Financial Analyst Tables ---
class AnalystFinancialBase(Base):
    """
    ORM osztály a pénzügyi elemzői összesített eredmények tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - last_date (DateTime, nullable=True): Az utolsó adat dátuma.
        - num_quarters (Integer, nullable=True): Felhasznált negyedévek száma.
        - profitability_used (Integer, nullable=False): Jövedelmezőség kategória használva (1/0).
        - revenue_growth_used (Integer, nullable=False): Bevételi növekedés kategória használva (1/0).
        - earnings_used (Integer, nullable=False): Nyereség kategória használva (1/0).
        - liquidity_used (Integer, nullable=False): Likviditás kategória használva (1/0).
        - leverage_used (Integer, nullable=False): Tőkeáttétel kategória használva (1/0).
        - cash_flow_health_used (Integer, nullable=False): Pénzforgalmi egészség kategória használva (1/0).
        - analysis_run_timestamp (BigInteger, nullable=True): Elemzés futtatásának időbélyege.
        - analysis_run_datetime (DateTime, nullable=True): Elemzés futtatásának dátuma.
        - warnings (String, nullable=True): Figyelmeztetések (JSON string vagy vesszővel elválasztott).
        - errors (String, nullable=True): Hibák (JSON string vagy vesszővel elválasztott).
        - aggregate_score (Float, nullable=True): Az összesített pénzügyi pontszám.
    """
    __tablename__ = 'analyst_financial_base'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    last_date = Column(DateTime, nullable=True)
    num_quarters = Column(Integer, nullable=True)
    profitability_used = Column(Integer, nullable=False, default=0)
    revenue_growth_used = Column(Integer, nullable=False, default=0)
    earnings_used = Column(Integer, nullable=False, default=0)
    liquidity_used = Column(Integer, nullable=False, default=0)
    leverage_used = Column(Integer, nullable=False, default=0)
    cash_flow_health_used = Column(Integer, nullable=False, default=0)
    analysis_run_timestamp = Column(BigInteger, nullable=True)
    analysis_run_datetime = Column(DateTime, nullable=True)
    warnings = Column(String, nullable=True)
    errors = Column(String, nullable=True)
    aggregate_score = Column(Float, nullable=True)


class AnalystFinancialScore(Base):
    """
    ORM osztály a pénzügyi elemzői kategória pontszámok tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - aggregate_score (Float, nullable=True): Az összesített pénzügyi pontszám.
        - profitability (Float, nullable=True): Jövedelmezőség pontszám.
        - revenue_growth (Float, nullable=True): Bevételi növekedés pontszám.
        - earnings (Float, nullable=True): Nyereség pontszám.
        - liquidity (Float, nullable=True): Likviditás pontszám.
        - leverage (Float, nullable=True): Tőkeáttétel pontszám.
        - cash_flow_health (Float, nullable=True): Pénzforgalmi egészség pontszám.
    """
    __tablename__ = 'analyst_financial_score'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    aggregate_score = Column(Float, nullable=True)
    profitability = Column(Float, nullable=True)
    revenue_growth = Column(Float, nullable=True)
    earnings = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    leverage = Column(Float, nullable=True)
    cash_flow_health = Column(Float, nullable=True)


class AnalystFinancialExplanation(Base):
    """
    ORM osztály a pénzügyi elemzői kategória magyarázatok tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése.
        - epoch (BigInteger): Az elemzés időpontja (Unix epoch).

    Mezők:
        - profitability (String, nullable=True): Jövedelmezőség magyarázat.
        - revenue_growth (String, nullable=True): Bevételi növekedés magyarázat.
        - earnings (String, nullable=True): Nyereség magyarázat.
        - liquidity (String, nullable=True): Likviditás magyarázat.
        - leverage (String, nullable=True): Tőkeáttétel magyarázat.
        - cash_flow_health (String, nullable=True): Pénzforgalmi egészség magyarázat.
    """
    __tablename__ = 'analyst_financial_explanation'
    symbol = Column(String, primary_key=True, nullable=False)
    epoch = Column(BigInteger, primary_key=True, nullable=False)
    profitability = Column(String, nullable=True)
    revenue_growth = Column(String, nullable=True)
    earnings = Column(String, nullable=True)
    liquidity = Column(String, nullable=True)
    leverage = Column(String, nullable=True)
    cash_flow_health = Column(String, nullable=True)


analyst_table_name_to_class = {
    "analyst_quantitative_base": AnalystQuantitativeBase,
    "analyst_quantitative_score": AnalystQuantitativeScore,
    "analyst_quantitative_explanation": AnalystQuantitativeExplanation,
    "analyst_financial_base": AnalystFinancialBase,
    "analyst_financial_score": AnalystFinancialScore,
    "analyst_financial_explanation": AnalystFinancialExplanation,
}