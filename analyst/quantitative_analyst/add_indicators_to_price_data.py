
import pandas as pd


def calculate_sma(data, window):
    """
    Calculate the Simple Moving Average (SMA) of a price series.

    This function computes the simple moving average over a specified window length. The SMA is the unweighted arithmetic mean of the price values over the window period. In other words, it sums up the prices of the last N periods and divides by N to produce each point. This smooths out short-term fluctuations and highlights longer-term price trends:contentReference[oaicite:15]{index=15}. For example, a 50-day SMA at time T is the average of the closing prices from the past 50 days up to T.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least a 'close' column.
    - window (int): Number of periods over which to calculate the moving average.

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'sma' column.
    """
    df = data.copy()

    prices = df['close']
    sma = prices.rolling(window=window, min_periods=window).mean()
    df[f'sma_win_len_{window}'] = sma
    return df


def calculate_ema(data, window):
    """
    Calculate the Exponential Moving Average (EMA) of a price series.

    This function computes the exponential moving average over a specified window. The EMA assigns greater weight to more recent prices and less weight to older prices, making it more responsive to recent changes than the simple moving average:contentReference[oaicite:16]{index=16}. The smoothing factor α is given by 2/(window+1). At each time step, EMA_t = α * price_t + (1 - α) * EMA_{t-1}. Because of this weighting scheme, the EMA will react faster to price moves and is often used to capture short-term trends.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least a 'close' column.
    - window (int): Number of periods for the EMA calculation.

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'ema' column.
    """
    df = data.copy()

    prices = df['close']
    ema = prices.ewm(span=window, adjust=False).mean()
    df[f'ema_win_len_{window}'] = ema
    return df


def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) and add it as a new column to the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least a 'close' column.
    - window (int): Look-back period for RSI calculation (default 14).

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'rsi' column.
    """
    df = data.copy()

    prices = df['close']
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df[f'rsi_win_len_{window}'] = rsi
    return df


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    This function computes MACD values for the given price series using the standard parameters. MACD is defined as the difference between a fast EMA and a slow EMA of the price (by default, 12-day EMA minus 26-day EMA). It also computes the signal line, which is an EMA of the MACD line (default 9-day EMA), and the histogram, which is the difference between MACD and the signal line. MACD is both a momentum and trend-following indicator; when the MACD values are above zero it means the short-term average is above the long-term average (indicating an upward bias in trend), and values below zero indicate the short-term average is below the long-term average (downward bias):contentReference[oaicite:19]{index=19}. Traders often watch for crossovers between the MACD line and the signal line as signals (MACD crossing below the signal can indicate a bearish turn, while crossing above can indicate a bullish turn):contentReference[oaicite:20]{index=20}.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least a 'close' column.
    - fast (int): Period for the fast EMA (default 12).
    - slow (int): Period for the slow EMA (default 26).
    - signal (int): Period for the signal line EMA (default 9).

    Returns:
    - pd.DataFrame: The input DataFrame with added 'macd', 'macd_signal', and 'macd_hist' columns.
    """
    df = data.copy()

    prices = df['close']
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    df[f'macd_f_{fast}_s_{slow}'] = macd
    df[f'macd_signal_{signal}'] = macd_signal
    df[f'macd_hist_{signal}'] = macd_hist
    return df


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands for a price series.

    This function computes the Bollinger Bands, which include:
    - Middle Band: a moving average of the price (typically a simple moving average over the given window).
    - Upper Band: the middle band plus (num_std * standard deviation of price over the window).
    - Lower Band: the middle band minus (num_std * standard deviation of price over the window).

    By default, a 20-period SMA is used for the middle band and bands are placed ±2 standard deviations from it, which encompasses roughly 95% of price movements under normal distribution assumptions:contentReference[oaicite:21]{index=21}. Bollinger Bands adjust to volatility; they widen during periods of high volatility and contract during periods of low volatility. Traders use them to identify potential overbought or oversold conditions, as price touching the upper band may suggest an overbought level and price touching the lower band may suggest an oversold level.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least a 'close' column.
    - window (int): Period for the moving average and standard deviation (default 20).
    - num_std (int or float): Number of standard deviations for the band offset (default 2).

    Returns:
    - pd.DataFrame: The input DataFrame with added 'bb_middle', 'bb_upper', and 'bb_lower' columns.
    """
    df = data.copy()

    prices = df['close']
    middle = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    df[f'bollinger_bands_middle_win_{window}_std_{num_std}'] = middle
    df[f'bollinger_bands_upper_win_{window}_std_{num_std}'] = upper
    df[f'bollinger_bands_lower_win_{window}_std_{num_std}'] = lower
    return df


def calculate_vwap(data):
    """
    Calculate the Volume-Weighted Average Price (VWAP) for daily OHLCV data.

    Parameters:
        data (pd.DataFrame): DataFrame with columns 'high', 'low', 'close', and 'volume'.
            The DataFrame may contain additional columns, which are preserved.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'vwap' column.
    """

    df = data.copy()
    # Accept both upper and lower case column names
    col_map = {c.lower(): c for c in df.columns}
    required = ['high', 'low', 'close', 'volume']
    if not all(k in col_map for k in required):
        raise ValueError(f"Input DataFrame must contain columns: {required}")

    high = df[col_map['high']]
    low = df[col_map['low']]
    close = df[col_map['close']]
    volume = df[col_map['volume']]

    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    vwap = pv.cumsum() / volume.cumsum()
    df['vwap'] = vwap
    return df


def calculate_stochastic(data, k_window=14, d_window=3):
    """
    Calculate the Stochastic Oscillator (%K and %D values).

    This function computes the stochastic oscillator for the given price data. %K is calculated as: 
    %K = (Current Close - Lowest Low of last N periods) / (Highest High of last N periods - Lowest Low of last N periods) * 100, 
    where N = k_window. %D is then the simple moving average of %K over d_window periods (a 3-period SMA by default). The oscillator outputs values from 0 to 100 representing the position of the close within the recent high-low range.

    The stochastic oscillator is a momentum indicator; readings close to 100 indicate the price is near its high of the recent range (strong upward momentum), while readings close to 0 indicate the price is near its low of the range (strong downward momentum):contentReference[oaicite:24]{index=24}. In practice, values above 80 are often deemed overbought and values below 20 oversold. Traders watch for the %K line crossing the %D line as potential buy or sell signals, especially when these lines are in the overbought/oversold regions.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns 'high', 'low', 'close'.
    - k_window (int): Look-back period for %K (default 14).
    - d_window (int): Period for the %D moving average (default 3).

    Returns:
    - pd.DataFrame: The input DataFrame with added '%K' and '%D' columns.
    """
    df = data.copy()

    high = df['high']
    low = df['low']
    close = df['close']
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_window, min_periods=d_window).mean()
    df[f'stochastic_oscillator_%K_kwin_{k_window}'] = percent_k
    df[f'stochastic_oscillator_%D_dwin_{d_window}'] = percent_d
    return df


def calculate_obv(data):
    """
    Calculate On-Balance Volume (OBV) and add it as a new column to the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least 'close' and 'volume' columns, sorted by date.

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'obv' column.
    """

    df = data.copy()
    close = df['close']
    volume = df['volume']
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df


def calculate_adx(data, window=14):
    """
    Calculate the Average Directional Index (ADX) and add it, along with +DI and -DI, as new columns to the input DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns 'high', 'low', 'close'.
    - window (int): The number of periods to use for smoothing and calculating ADX (default 14).

    Returns:
    - pd.DataFrame: The input DataFrame with added 'adx', 'plus_di', and 'minus_di' columns.
    """

    df = data.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=window, min_periods=window).mean()
    df[f'adx_win_{window}'] = adx
    df[f'plus_di_win_{window}'] = plus_di
    df[f'minus_di_win_{window}'] = minus_di
    return df
