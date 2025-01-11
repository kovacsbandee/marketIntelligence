import plotly.graph_objects as go
from plotly.subplots import make_subplots


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_price_with_indicators(price_table, dividend_table=None, 
                               include_macd=False, include_rsi=False, 
                               dividend_date_col='ex_dividend_date',
                               ):
    """
    Plots price data with optional MACD, RSI, and dividends using Plotly's make_subplots.

    Parameters:
    price_table (pd.DataFrame): DataFrame containing price data (['date', 'open', 'high', 'low', 'close']).
    dividend_table (pd.DataFrame): DataFrame containing dividend data with a date column and amount column.
    include_macd (bool): Whether to add MACD visualization if columns exist.
    include_rsi (bool): Whether to add RSI visualization if columns exist.
    dividend_date_col (str): The column name for dividend dates in dividend_table.
    dividend_filter_range (tuple): Tuple of start and end dates (e.g., ('2023-01-01', '2023-12-31')) to filter dividends.

    Returns:
    Plotly figure object.
    """
    # Initialize subplots
    rows = 1 + int(include_macd) + int(include_rsi)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5] + [0.25] * (rows - 1),
        subplot_titles=["Price Data"] +
                       (["MACD"] if include_macd else []) +
                       (["RSI"] if include_rsi else [])
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

    # Add dividends as yellow dots
    if dividend_table is not None:
        try:
            # Filter dividend table by date range if provided
            filtered_dividends = dividend_table[
                (dividend_table[dividend_date_col] >= price_table["date"].min()) &
                (dividend_table[dividend_date_col] <= price_table["date"].max())
            ]
        
            # Match dividend dates to closest available dates in price_table
            dividend_points = filtered_dividends[dividend_date_col].apply(
                lambda d: price_table.iloc[
                    (price_table['date'] - d).abs().idxmin()
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=dividend_points['date'],
                    y=dividend_points['close'],
                    mode="markers",
                    marker=dict(size=4, color="yellow"),
                    name="Dividends",
                    hovertext=[
                        f"Date: {row[dividend_date_col]}, Amount: {row['amount']}"
                        for _, row in filtered_dividends.iterrows()
                    ],
                    hoverinfo="text"
                ),
                row=1, col=1
            )
        except KeyError as e:
            print(f"Error: Missing column in dividend table: {e}")
        except Exception as e:
            print(f"Error processing dividends: {e}")

    # Add MACD visualization if requested
    if include_macd:
        try:
            fig.add_trace(
                go.Scatter(
                    x=price_table['date'],
                    y=price_table['MACD_line'],
                    mode='lines',
                    name='MACD Line'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=price_table['date'],
                    y=price_table['signal_line'],
                    mode='lines',
                    name='Signal Line'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=price_table['date'],
                    y=price_table['MACD_histogram'],
                    name='MACD Histogram'
                ),
                row=2, col=1
            )
        except KeyError as e:
            print(f"Error: MACD values not found in the price table. Missing column: {e}")

    # Add RSI visualization if requested
    if include_rsi:
        try:
            fig.add_trace(
                go.Scatter(
                    x=price_table['date'],
                    y=price_table['RSI'],
                    mode='lines',
                    name='RSI'
                ),
                row=3 if include_macd else 2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=price_table['date'],
                    y=[70] * len(price_table),  # Overbought level
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='Overbought (70)'
                ),
                row=3 if include_macd else 2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=price_table['date'],
                    y=[30] * len(price_table),  # Oversold level
                    mode='lines',
                    line=dict(dash='dash', color='green'),
                    name='Oversold (30)'
                ),
                row=3 if include_macd else 2, col=1
            )
        except KeyError as e:
            print(f"Error: RSI values not found in the price table. Missing column: {e}")
    fig.update_layout(
    title={
            "text": "Price Data with Indicators and Dividends",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}  # Larger title font size
        },
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    showlegend=True,
    width=1200,
    height=800, 
    margin=dict(l=50, r=50, t=80, b=50),  
    font=dict(size=14))

    return fig