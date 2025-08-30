
def add_bollinger_bands_to_candlestick(data):
    """
    Plots a candlestick chart with overlaid Bollinger Bands (upper, middle, lower) using existing columns.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # Try to find columns for Bollinger Bands
    bb_middle = [col for col in df.columns if 'bb_middle' in col][0] if any('bb_middle' in col for col in df.columns) else None
    bb_upper = [col for col in df.columns if 'bb_upper' in col][0] if any('bb_upper' in col for col in df.columns) else None
    bb_lower = [col for col in df.columns if 'bb_lower' in col][0] if any('bb_lower' in col for col in df.columns) else None
    if not (bb_middle and bb_upper and bb_lower):
        return plot_candlestick_chart(df)
    fig = plot_candlestick_chart(df)
    x = df["date"] if "date" in df.columns else df.index
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[bb_upper],
            mode="lines",
            name="BB Upper",
            line=dict(color="#636EFA", width=1, dash="dot"),
            hovertemplate=f"<b>BB Upper</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[bb_middle],
            mode="lines",
            name="BB Middle",
            line=dict(color="#00CC96", width=1, dash="dash"),
            hovertemplate=f"<b>BB Middle</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[bb_lower],
            mode="lines",
            name="BB Lower",
            line=dict(color="#EF553B", width=1, dash="dot"),
            hovertemplate=f"<b>BB Lower</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_layout(
        title={
            "text": f"Candlestick Chart with Bollinger Bands",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        }
    )
    return fig

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from infrastructure.ui.dash.add_price_indicators import AddPriceIndicators
from infrastructure.ui.dash.plot_utils import DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT, _get_column_descriptions


def plot_candlestick_chart(data):
    """
    Plots a candlestick chart for price data (OHLC), optionally with volume bars.

    This function creates a candlestick chart from OHLC (open-high-low-close) data for a given stock or financial instrument over time.
    Each candlestick represents one period (e.g. one day) and shows the open, high, low, and closing prices for that interval.
    The candlestick chart provides a visual representation of price movement and volatility, reflecting changes in investor sentiment over time.
    The trading volume can be displayed (typically as a bar plot below the candlesticks) to give context on the strength of price moves,
    since higher volume often signifies greater conviction behind a price change.

    Implementation details:
        - The function is structured to allow easy addition of overlays (e.g. SMA, EMA, ADX) to the price chart in the future.
        - Uses Plotly subplots to display volume below the price chart if available.
        - All overlays should be added as additional traces to row=1, col=1.

    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data with columns:
            - date (datetime or str)
            - open
            - high
            - low
            - close
            - volume (optional)
    Returns:
        go.Figure: Plotly Figure object containing the candlestick chart and volume bars.
    """
    descriptions = _get_column_descriptions("daily_timeseries")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.empty or not all(col in data.columns for col in ["open", "high", "low", "close"]):
        return go.Figure()
    df = data.copy()
    # Try to parse date column if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        x = df["date"]
    else:
        x = df.index

    # Create subplots: candlestick + volume if available
    has_volume = "volume" in df.columns and df["volume"].notnull().any()
    row_heights = [0.75, 0.25] if has_volume else [1.0]
    specs = [[{"secondary_y": False}]]
    if has_volume:
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[descriptions.get("candlestick", "Price (OHLC)")] + ([descriptions.get("volume", "Volume")] if has_volume else [])
    )

    # Candlestick trace
    # Prepare hovertext for each candlestick
    hovertext = [
        f"<b>{descriptions.get('date', 'Date')}</b>: {d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d}<br>"
        f"{descriptions.get('open', 'Open')}: {o:.2f}<br>"
        f"{descriptions.get('high', 'High')}: {h:.2f}<br>"
        f"{descriptions.get('low', 'Low')}: {l:.2f}<br>"
        f"{descriptions.get('close', 'Close')}: {c:.2f}"
        for d, o, h, l, c in zip(x, df["open"], df["high"], df["low"], df["close"])
    ]
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=descriptions.get("candlestick", "Price (OHLC)"),
            increasing_line_color="#21ba45",
            decreasing_line_color="#fa5252",
            showlegend=False,
            hovertext=hovertext,
        ),
        row=1, col=1
    )

    # Volume bars (if available)
    if has_volume:
        # Color volume bars green if close > open, red otherwise
        volume_colors = [
            "#21ba45" if c > o else "#fa5252" for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["volume"],
                name=descriptions.get("volume", "Volume"),
                marker_color=volume_colors,
                opacity=0.5,
                showlegend=False,
                hovertemplate=f"<b>{descriptions.get('date', 'Date')}</b>: %{{x|%Y-%m-%d}}<br>"
                              f"{descriptions.get('volume', 'Volume')}: %{{y:.0f}}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text=descriptions.get("volume", "Volume"), row=2, col=1)

    # Layout
    fig.update_layout(
        title={
            "text": descriptions.get("candlestick_title", "Candlestick Chart"),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title=descriptions.get("date", "Date"),
        yaxis_title=descriptions.get("price", "Price"),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    fig.update_yaxes(title_text=descriptions.get("price", "Price"), row=1, col=1)
    # For overlays: future indicators can be added as additional traces to row=1, col=1
    return fig


def add_moving_averages_to_candlestick(data):
    """
    Plots a candlestick chart with overlaid moving average lines (e.g. 50-day and 200-day SMA).

    This function overlays one or more moving average lines on a candlestick price chart to visualize the underlying trend.
    Traders often use a medium-term and a long-term moving average (for example, 50-day and 200-day) to identify trend direction;
    a crossover where a shorter-term moving average rises above a longer-term moving average is viewed as a bullish signal indicating upward momentum:contentReference[oaicite:1]{index=1}.
    The moving average lines help smooth out short-term price fluctuations, making the overall trend clearer.

    The function calls the `calculate_sma` (or `calculate_ema` if specified) helper for each window period to compute the moving averages, then plots these lines on top of the candlestick chart.

    Implementation details:
        - All columns in the input DataFrame whose names contain 'sma' (case-insensitive) are overlaid as unique colored lines.
        - Each SMA line is given a unique color for clarity (using Plotly's qualitative palettes).
        - Uses plot_candlestick_chart as the base chart, overlays all SMA lines as traces on row=1, col=1.
        - If no SMA columns are found, returns the base candlestick chart.

    Args:
        data (pd.DataFrame): DataFrame with OHLC price data and one or more SMA columns (e.g. 'sma_win_len_50').
    Returns:
        go.Figure: Plotly Figure object with candlestick and all SMA overlays.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # Find all SMA columns (case-insensitive)
    sma_cols = [col for col in df.columns if 'sma' in col.lower()]
    if not sma_cols:
        # No SMA columns, just return the candlestick chart
        return plot_candlestick_chart(df)

    # Use plot_candlestick_chart for the base
    fig = plot_candlestick_chart(df)
    # Overlay each SMA as a line (unique color)
    color_cycle = qualitative.Plotly + qualitative.D3 + qualitative.Set1 + qualitative.Set2
    x = df["date"] if "date" in df.columns else df.index
    for i, col in enumerate(sma_cols):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color=color_cycle[i % len(color_cycle)], width=2, dash="solid"),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>SMA: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
    fig.update_layout(
        title={
            "text": "Candlestick Chart with SMA Overlays",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        }
    )
    return fig


def plot_candlestick_with_bollinger_bands(data, window=20, num_std=2):
    """
    Plots a candlestick chart with Bollinger Bands overlaid.

    Bollinger Bands consist of a middle band (a moving average of the price, typically a 20-period SMA)
    and an upper and lower band offset by a certain number of standard deviations (usually 2) above and below that middle band:contentReference[oaicite:2]{index=2}.
    These bands expand and contract with market volatility and help identify potential overbought or oversold conditions:
    for example, when price touches or exceeds the upper band, it may be considered relatively high (overbought), and when it touches or falls below the lower band,
    it may be relatively low (oversold):contentReference[oaicite:3]{index=3}.

    This function uses the `calculate_bollinger_bands` helper to compute the middle, upper, and lower band values from the price data.
    It then plots the candlestick chart with the Bollinger Bands lines to visualize volatility and price extremes.

    Parameters:
        data (pd.DataFrame): DataFrame with OHLC price data (columns "Open", "High", "Low", "Close").
        window (int): Look-back period for the moving average (middle band) and standard deviation (default 20).
        num_std (int or float): Number of standard deviations for the band offset (default 2).

    Returns:
        Plotly.Figure: The Figure object containing the candlestick chart with upper and lower Bollinger Bands.

    Example:
        >>> fig = plot_candlestick_with_bollinger_bands(df, window=20, num_std=2)
        >>> fig.show()
    """
    pass


def plot_candlestick_with_vwap(data):
    """
    Plots a candlestick chart with a Volume-Weighted Average Price (VWAP) line overlay.

    This function generates a candlestick chart for intraday price data and overlays the VWAP line on the chart. VWAP represents the average price of the security for the day weighted by trading volume. It is calculated cumulatively from the start of the trading session, giving traders a benchmark of the average price paid per share. Prices above the VWAP line indicate the asset is trading above the day's average cost (often viewed as bullish intraday sentiment), while prices below VWAP indicate trading below the average cost (bearish sentiment).

    The function uses the VWAP column if present, otherwise returns the base candlestick chart.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    vwap_col = [col for col in df.columns if 'vwap' in col.lower()]
    if not vwap_col:
        return plot_candlestick_chart(df)
    vwap_col = vwap_col[0]
    fig = plot_candlestick_chart(df)
    x = df["date"] if "date" in df.columns else df.index
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[vwap_col],
            mode="lines",
            name="VWAP",
            line=dict(color="purple", width=2, dash="solid"),
            hovertemplate=f"<b>VWAP</b><br>Date: %{{x|%Y-%m-%d}}<br>VWAP: %{{y:.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_layout(
        title={
            "text": "Candlestick Chart with VWAP Overlay",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        }
    )
    return fig


def plot_rsi(data, window=14):
    """
    Plots the Relative Strength Index (RSI) as a line chart.

    The Relative Strength Index is a momentum oscillator that measures the magnitude of recent price gains versus losses to assess the speed of price movements:contentReference[oaicite:5]{index=5}. It oscillates between 0 and 100. In general, an RSI value above 70 suggests the asset may be overbought (price has risen quickly in the period, potentially due for a pullback), while an RSI below 30 suggests it may be oversold (price has fallen quickly, potentially due for a rebound):contentReference[oaicite:6]{index=6}.

    This function uses the `calculate_rsi` helper to compute the RSI values (typically using a 14-period window by default) from the input price data (usually closing prices). It then plots the RSI line, often including horizontal reference lines at 30 and 70 to denote the common oversold/overbought thresholds.

    Parameters:
        data (pd.DataFrame or pd.Series): Price data to compute RSI from (if DataFrame, the "Close" column is used).
        window (int): The number of periods for RSI calculation (default 14).

    Returns:
        Plotly.Figure: The Figure object containing the RSI chart, which can be shown below a price chart for context.

    Example:
        >>> fig = plot_rsi(df, window=14)
        >>> fig.show()
    """
    """
    Plots the Relative Strength Index (RSI) as a line chart.

    The Relative Strength Index is a momentum oscillator that measures the magnitude of recent price gains versus losses to assess the speed of price movements. It oscillates between 0 and 100. In general, an RSI value above 70 suggests the asset may be overbought (price has risen quickly in the period, potentially due for a pullback), while an RSI below 30 suggests it may be oversold (price has fallen quickly, potentially due for a rebound).

    This function uses the `calculate_rsi` helper to compute the RSI values (typically using a 14-period window by default) from the input price data (usually closing prices). It then plots the RSI line, often including horizontal reference lines at 30 and 70 to denote the common oversold/overbought thresholds.

    Parameters:
        data (pd.DataFrame or pd.Series): Price data to compute RSI from (if DataFrame, the "Close" column is used).
        window (int): The number of periods for RSI calculation (default 14).

    Returns:
        Plotly.Figure: The Figure object containing the RSI chart, which can be shown below a price chart for context.

    Example:
        >>> fig = plot_rsi(df, window=14)
        >>> fig.show()
    """
    import plotly.graph_objects as go
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # Try to find RSI column (case-insensitive)
    rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
    if not rsi_cols:
        # If not present, try to calculate RSI from close prices
        if 'close' in df.columns:
            close = df['close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=window, min_periods=window).mean()
            avg_loss = loss.rolling(window=window, min_periods=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df['RSI'] = rsi
            rsi_col = 'RSI'
        else:
            # No close price, return empty figure
            return go.Figure()
    else:
        rsi_col = rsi_cols[0]
    x = df['date'] if 'date' in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[rsi_col],
            mode="lines",
            name="RSI",
            line=dict(color="#636EFA", width=2),
            hovertemplate=f"<b>RSI</b><br>Date: %{{x|%Y-%m-%d}}<br>RSI: %{{y:.2f}}<extra></extra>"
        )
    )
    # Add overbought/oversold reference lines
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[70]*len(x),
            mode="lines",
            name="Overbought (70)",
            line=dict(color="red", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[30]*len(x),
            mode="lines",
            name="Oversold (30)",
            line=dict(color="green", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.update_layout(
        title={
            "text": "Relative Strength Index (RSI)",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig


def plot_macd(data, fast=12, slow=26, signal=9):
    """
    Plots the Moving Average Convergence Divergence (MACD) indicator.

    MACD is calculated as the difference between a fast (short-term) EMA and a slow (long-term) EMA of the price, and it is usually accompanied by a signal line (an EMA of the MACD line) and a histogram showing the difference between MACD and the signal. On the chart, MACD oscillates above and below a zero line; values above zero indicate the short-term average is above the long-term average (upward bias), while values below zero indicate the opposite (downward bias):contentReference[oaicite:7]{index=7}. When the MACD line crosses below the signal line, it indicates weakening momentum to the upside (a bearish signal), and when the MACD line crosses above the signal line, it indicates strengthening upward momentum (a bullish signal):contentReference[oaicite:8]{index=8}.

    The function uses `calculate_macd` to compute the MACD line, signal line, and histogram from the input price series (typically closing prices). It then plots these components (MACD and signal as lines, and the histogram as bars) in a separate panel, often with a horizontal zero reference line for context.

    Parameters:
        data (pd.DataFrame or pd.Series): Price data for computing MACD (if DataFrame, the "Close" column is used).
        fast (int): Period for the fast EMA (default 12 days).
        slow (int): Period for the slow EMA (default 26 days).
        signal (int): Period for the signal line EMA (default 9 days).

    Returns:
        Plotly.Figure: The Figure object containing the MACD chart (MACD line, signal line, and histogram).

    Example:
        >>> fig = plot_macd(df)
        >>> fig.show()
    """
    """
    Plots the Moving Average Convergence Divergence (MACD) indicator.

    MACD is calculated as the difference between a fast (short-term) EMA and a slow (long-term) EMA of the price, and it is usually accompanied by a signal line (an EMA of the MACD line) and a histogram showing the difference between MACD and the signal. On the chart, MACD oscillates above and below a zero line; values above zero indicate the short-term average is above the long-term average (upward bias), while values below zero indicate the opposite (downward bias). When the MACD line crosses below the signal line, it indicates weakening momentum to the upside (a bearish signal), and when the MACD line crosses above the signal line, it indicates strengthening upward momentum (a bullish signal).

    The function uses `calculate_macd` to compute the MACD line, signal line, and histogram from the input price series (typically closing prices). It then plots these components (MACD and signal as lines, and the histogram as bars) in a separate panel, often with a horizontal zero reference line for context.
    """
    import plotly.graph_objects as go
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # Only use precomputed columns
    macd_line_col = next((col for col in df.columns if 'macd' in col.lower() and 'hist' not in col.lower() and 'signal' not in col.lower()), None)
    signal_line_col = next((col for col in df.columns if 'macd_signal' in col.lower() or 'signal' in col.lower()), None)
    hist_col = next((col for col in df.columns if 'macd_hist' in col.lower() or 'hist' in col.lower()), None)
    if not (macd_line_col and signal_line_col and hist_col):
        return go.Figure()
    x = df['date'] if 'date' in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[macd_line_col],
            mode="lines",
            name="MACD Line",
            line=dict(color="#636EFA", width=2),
            hovertemplate=f"<b>MACD Line</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[signal_line_col],
            mode="lines",
            name="Signal Line",
            line=dict(color="#EF553B", width=2, dash="dash"),
            hovertemplate=f"<b>Signal Line</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=df[hist_col],
            name="MACD Histogram",
            marker_color="#00CC96",
            opacity=0.5,
            hovertemplate=f"<b>MACD Histogram</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[0]*len(x),
            mode="lines",
            name="Zero Line",
            line=dict(color="gray", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.update_layout(
        title={
            "text": "MACD (Moving Average Convergence Divergence)",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Date",
        yaxis_title="MACD",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig


def plot_stochastic(data, k_window=14, d_window=3):
    """
    Plots the Stochastic Oscillator (%K and %D lines).

    The Stochastic Oscillator is a momentum indicator that compares a security's closing price relative to its price range over a recent period. It produces two lines: %K (fast line) and %D (slow line, which is a moving average of %K). Both %K and %D oscillate between 0 and 100. Values above 80 typically indicate the price is near the top of its recent range (potentially overbought), while values below 20 indicate the price is near the bottom of its range (potentially oversold):contentReference[oaicite:9]{index=9}. Traders often look for crossovers of these lines for signals – for instance, if %K crosses below %D above 80, it could signal a bearish reversal, whereas %K crossing above %D below 20 could signal a bullish reversal:contentReference[oaicite:10]{index=10}.

    This function uses `calculate_stochastic` to compute the %K and %D series from the input data (requiring High, Low, and Close prices). It then plots the %K and %D lines, typically with horizontal reference lines at 20 and 80 to highlight the common threshold levels.

    Parameters:
        data (pd.DataFrame): DataFrame with price data columns "High", "Low", and "Close" for each period.
        k_window (int): Look-back period for %K calculation (default 14).
        d_window (int): Period for the %D moving average (default 3).

    Returns:
        Plotly.Figure: The Figure object containing the stochastic oscillator chart.

    Example:
        >>> fig = plot_stochastic(df, k_window=14, d_window=3)
        >>> fig.show()
    """
    """
    Plots the Stochastic Oscillator (%K and %D lines).
    """
    import plotly.graph_objects as go
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # Only use precomputed columns
    k_col = next((col for col in df.columns if '%k' in col.lower()), None)
    d_col = next((col for col in df.columns if '%d' in col.lower()), None)
    if not (k_col and d_col):
        return go.Figure()
    x = df['date'] if 'date' in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[k_col],
            mode="lines",
            name="%K",
            line=dict(color="#636EFA", width=2),
            hovertemplate=f"<b>%K</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[d_col],
            mode="lines",
            name="%D",
            line=dict(color="#EF553B", width=2, dash="dash"),
            hovertemplate=f"<b>%D</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[80]*len(x),
            mode="lines",
            name="Overbought (80)",
            line=dict(color="red", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[20]*len(x),
            mode="lines",
            name="Oversold (20)",
            line=dict(color="green", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.update_layout(
        title={
            "text": "Stochastic Oscillator (%K, %D)",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Date",
        yaxis_title="Stochastic",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig


def plot_obv(data):
    """
    Plots the On-Balance Volume (OBV) indicator as a line chart.

    OBV is a cumulative volume-based indicator that measures buying and selling pressure by adding the day's volume on periods when the price closes higher than the previous close and subtracting the day's volume when the price closes lower:contentReference[oaicite:11]{index=11}. (If the closing price is unchanged, OBV remains the same.) Over time, the OBV line’s trajectory can confirm price trends or reveal divergences. For example, a rising OBV line alongside rising prices indicates that volume is confirming the uptrend, whereas if price is making new highs but OBV is flat or falling, it may signal weakening strength in the trend.

    This function uses `calculate_obv` to compute the OBV series from the input data’s closing prices and volumes. It then plots the OBV line, usually in a separate subplot beneath the price chart, since the absolute OBV value is arbitrary (it depends on a starting point) and it's the trend of OBV that matters:contentReference[oaicite:12]{index=12}.

    Parameters:
        data (pd.DataFrame): DataFrame with columns "Close" and "Volume", providing the price and volume series needed for OBV.

    Returns:
        Plotly.Figure: The Figure object containing the OBV line plot.

    Example:
        >>> fig = plot_obv(df)
        >>> fig.show()
    """
    """
    Plots the On-Balance Volume (OBV) indicator as a line chart.
    """
    import plotly.graph_objects as go
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    obv_col = next((col for col in df.columns if 'obv' in col.lower()), None)
    if not obv_col:
        return go.Figure()
    x = df['date'] if 'date' in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[obv_col],
            mode="lines",
            name="OBV",
            line=dict(color="#636EFA", width=2),
            hovertemplate=f"<b>OBV</b><br>Date: %{{x|%Y-%m-%d}}<br>OBV: %{{y:.0f}}<extra></extra>"
        )
    )
    fig.update_layout(
        title={
            "text": "On-Balance Volume (OBV)",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Date",
        yaxis_title="OBV",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig


def plot_adx(data, window=14):
    """
    Plots the Average Directional Index (ADX) and its +DI / -DI lines.

    The Average Directional Index is a trend-strength indicator that measures how strong a trend is, regardless of whether it is up or down:contentReference[oaicite:13]{index=13}. The ADX is plotted alongside two companion lines: +DI (positive directional indicator) and -DI (negative directional indicator), which represent the strength of upward and downward movements, respectively. On the chart, a high ADX value (for example, above 40) indicates a very strong trend, while a low ADX value (for example, below 20) indicates a weak or non-trending market:contentReference[oaicite:14]{index=14}. The relative positioning of +DI and -DI reveals the trend direction (e.g., +DI above -DI signifies an uptrend, and vice versa, provided ADX indicates a trending market).

    This function uses `calculate_adx` to compute the ADX as well as the +DI and -DI series from the high, low, and close price data (commonly using a 14-period look-back). It then plots all three lines together, allowing traders to assess both the strength and the direction of the trend.

    Parameters:
        data (pd.DataFrame): DataFrame with price data columns "High", "Low", "Close" for each period.
        window (int): The period over which to calculate ADX and the directional indicators (default 14).

    Returns:
        Plotly.Figure: The Figure object containing the ADX plot (with ADX, +DI, and -DI lines).

    Example:
        >>> fig = plot_adx(df, window=14)
        >>> fig.show()
    """
    """
    Plots the Average Directional Index (ADX) and its +DI / -DI lines.
    """
    import plotly.graph_objects as go
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    adx_col = next((col for col in df.columns if 'adx' in col.lower()), None)
    plus_di_col = next((col for col in df.columns if '+di' in col.lower() or 'plus_di' in col.lower()), None)
    minus_di_col = next((col for col in df.columns if '-di' in col.lower() or 'minus_di' in col.lower()), None)
    if not (adx_col and plus_di_col and minus_di_col):
        return go.Figure()
    x = df['date'] if 'date' in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[adx_col],
            mode="lines",
            name="ADX",
            line=dict(color="#636EFA", width=2),
            hovertemplate=f"<b>ADX</b><br>Date: %{{x|%Y-%m-%d}}<br>ADX: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[plus_di_col],
            mode="lines",
            name="+DI",
            line=dict(color="#00CC96", width=2, dash="dash"),
            hovertemplate=f"<b>+DI</b><br>Date: %{{x|%Y-%m-%d}}<br>+DI: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[minus_di_col],
            mode="lines",
            name="-DI",
            line=dict(color="#EF553B", width=2, dash="dot"),
            hovertemplate=f"<b>-DI</b><br>Date: %{{x|%Y-%m-%d}}<br>-DI: %{{y:.2f}}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[20]*len(x),
            mode="lines",
            name="Weak Trend (20)",
            line=dict(color="gray", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[40]*len(x),
            mode="lines",
            name="Strong Trend (40)",
            line=dict(color="black", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=True
        )
    )
    fig.update_layout(
        title={
            "text": "Average Directional Index (ADX)",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Date",
        yaxis_title="ADX / DI",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig



def add_macd_subplot(price_table: pd.DataFrame,
                     figure: go.Figure) -> go.Figure:

    price_table = AddPriceIndicators(table=price_table).macd()

    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['MACD_line'],
            mode='lines',
            name='MACD Line'), row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['signal_line'],
            mode='lines',
            name='Signal Line'), row=2, col=1)
    figure.add_trace(
        go.Bar(
            x=price_table['date'],
            y=price_table['MACD_histogram'],
            name='MACD Histogram'), row=2, col=1)

    return figure


def add_rsi_subplot(price_table: pd.DataFrame,
                    figure: go.Figure,
                    include_macd: bool) -> go.Figure:

    price_table = AddPriceIndicators(table=price_table).rsi()

    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['RSI'],
            mode='lines',
            name='RSI'), row=3 if include_macd else 2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=[70] * len(price_table),  # Overbought level
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Overbought (70)'), row=3 if include_macd else 2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=[30] * len(price_table),  # Oversold level
            mode='lines',
            line=dict(dash='dash', color='green'),
            name='Oversold (30)'), row=3 if include_macd else 2, col=1)

    return figure

def add_vwap_subplot(price_table: pd.DataFrame,
                      figure: go.Figure) -> go.Figure:
    """
    Adds a VWAP (Volume Weighted Average Price) subplot to the figure.

    Parameters:
        price_table (pd.DataFrame): DataFrame containing price data with 'date', 'close', and 'volume' columns.
        figure (go.Figure): The Plotly figure to which the VWAP subplot will be added.

    Returns:
        go.Figure: The updated figure with the VWAP subplot added.
    """
    # Calculate VWAP using AddPriceIndicators
    price_table = AddPriceIndicators(table=price_table).vwap()

    # Add VWAP line to the subplot
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='purple')
        ),
        row=4, col=1
    )

    return figure

def plot_price_with_indicators(price_table,
                               include_macd: bool = False,
                               include_rsi: bool = False,
                               include_vwap: bool = False):
    """
    Plots price data with optional MACD, RSI, VWAP, and dividends using Plotly's make_subplots.

    Parameters:
    price_table (pd.DataFrame): DataFrame containing price data (['date', 'open', 'high', 'low', 'close']).
    include_macd (bool): Whether to add MACD visualization if columns exist.
    include_rsi (bool): Whether to add RSI visualization if columns exist.
    include_vwap (bool): Whether to add VWAP visualization if columns exist.

    Returns:
    Plotly figure object.
    """
    # Initialize subplots
    rows = 1 + int(include_macd) + int(include_rsi) + int(include_vwap)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.25] * (rows - 1),
        subplot_titles=["Price Data"] +
        (["MACD"] if include_macd else []) +
        (["RSI"] if include_rsi else []) +
        (["VWAP"] if include_vwap else [])
    )

    # Plot base candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_table['date'],
            open=price_table['open'],
            high=price_table['high'],
            low=price_table['low'],
            close=price_table['close'],
            name="Price",
        ),
        row=1, col=1
    )

    # Add MACD visualization if requested
    if include_macd:
        try:
            fig = add_macd_subplot(price_table=price_table, figure=fig)
        except KeyError as e:
            print(
                f"Error: MACD values not found in the price table. Missing column: {e}")

    # Add RSI visualization if requested
    if include_rsi:
        try:
            add_rsi_subplot(price_table=price_table,
                            figure=fig,
                            include_macd=include_macd)
        except KeyError as e:
            print(
                f"Error: RSI values not found in the price table. Missing column: {e}")

    # Add VWAP visualization if requested
    if include_vwap:
        try:
            fig = add_vwap_subplot(price_table=price_table, figure=fig)
        except KeyError as e:
            print(
                f"Error: VWAP values not found in the price table. Missing column: {e}")

    # Update layout to add date labels to each x-axis
    fig.update_layout(
        title={
            "text": "Price Data with Indicators",
            "x": 0.5,  
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20} 
        },
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_WIDTH,  # This plot is intentionally square (1800x1800)
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )

    # Add date labels to each x-axis
    for i in range(1, rows + 1):
        fig.update_xaxes(title_text="Date", row=i, col=1)

    return fig
