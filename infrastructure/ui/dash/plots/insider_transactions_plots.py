import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

import pandas as pd

from infrastructure.ui.dash.plot_utils import DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT

def prepare_insider_data(insider_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw insider Form 4 data for plotting.
    
    - Parse the 'transaction_date' column into datetime.  
    - Map the Form 4 code 'A'/'D' to a new 'trade_type' column with values 'Buy' or 'Sell'.  
    - Categorize each insider’s title into broad officer categories:
      e.g. 'CEO/President', 'CFO', 'Director', 'Officer' (for VPs/EVPs/etc.), or 'Other'.  
      This lets us group trades by executive role when plotting.  
    - Return the cleaned DataFrame, with new columns 'trade_type' and 'officer_category'.
    """
    df = insider_df.copy()
    # Parse dates
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Map A/D codes to Buy/Sell
    df['trade_type'] = df.get('acquisition_or_disposal', '').map({'A': 'Buy', 'D': 'Sell'})
    # Categorize executive roles by keywords
    def categorize_role(title):
        t = title.lower()
        # Vice Presidents, SVPs, EVPs, etc.
        if any(x in t for x in ['vice president', 'vp', 'evp', 'svp']):
            return 'Officer'
        # CEO or President
        elif any(x in t for x in ['chief executive', 'ceo', 'president']):
            return 'CEO/President'
        # CFO or similar
        elif any(x in t for x in ['chief financial', 'cfo']):
            return 'CFO'
        # Director
        elif 'director' in t:
            return 'Director'
        else:
            return 'Other'
    df['officer_category'] = df['executive_title'].astype(str).apply(categorize_role)
    return df


def plot_insider_price_chart(price_df: pd.DataFrame, insider_df: pd.DataFrame) -> go.Figure:
    """
    Plot daily stock price with insider trades annotated. 
    
    - `price_df` must have columns `date` (daily dates) and `close` (closing prices).  
    - `insider_df` is the cleaned insider DataFrame (e.g. from `prepare_insider_data`) with
      'transaction_date', 'trade_type' ('Buy'/'Sell'), 'officer_category', and 'shares'.  
    The function overlays the price line (date vs. close) and adds scatter markers at each
    insider trade date. Buys are green markers, sells are red markers. Marker symbols
    or hover labels distinguish officer categories (CEO, CFO, Director, etc.) so one can
    compare how each group is trading. Marker size can be set proportional to `shares` 
    to indicate trade size. 
    
    Returns a Plotly Figure with this time-series chart. The x-axis is daily dates and
    the y-axis is stock price. This follows industry practice of highlighting insider buys/sells on
    the stock chart:contentReference[oaicite:9]{index=9} for visual pattern analysis.
    """
    # --- Example outline (implementation details may vary) ---
    price_df = price_df.copy()
    if "date" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df = price_df.sort_values("date")

    fig = go.Figure()
    # Add the stock price line
    fig.add_trace(go.Scatter(
        x=price_df['date'], y=price_df['close'],
        mode='lines', name='Price',
        line=dict(color='black')
    ))
    # Loop over each officer category and trade type to add markers
    categories = insider_df['officer_category'].unique()
    for category in categories:
        df_cat = insider_df[insider_df['officer_category'] == category]
        for trade_type, color in [('Buy','green'), ('Sell','red')]:
            df_tt = df_cat[df_cat['trade_type'] == trade_type]
            if df_tt.empty:
                continue
            size_scale = df_tt['shares'].max() if df_tt['shares'].max() > 0 else 1
            fig.add_trace(go.Scatter(
                x=df_tt['transaction_date'],
                y=[price_df.loc[price_df['date'] == d, 'close'].values[0]
                   if d in set(price_df['date']) else None
                   for d in df_tt['transaction_date']],
                mode='markers',
                name=f"{category} {trade_type}",
                marker=dict(
                    symbol='triangle-up' if trade_type == 'Buy' else 'triangle-down',
                    color=color,
                    size=df_tt['shares'] / size_scale * 20 + 5
                ),
                customdata=df_tt['shares'],
                hovertemplate=(
                    f"<b>{category} {trade_type}</b><br>Date: %{{x|%Y-%m-%d}}<br>"
                    "Price: %{y:.2f}<br>Shares: %{customdata:,.0f}<extra></extra>"
                )
            ))
    price_min = price_df["close"].min()
    price_max = price_df["close"].max()
    padding = (price_max - price_min) * 0.08 if pd.notnull(price_min) and pd.notnull(price_max) else 0

    fig.update_layout(
        title="Insider trades on price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    if pd.notnull(price_min) and pd.notnull(price_max):
        fig.update_yaxes(range=[price_min - padding, price_max + padding])
    return fig


def plot_insider_transactions_over_time(insider_df: pd.DataFrame) -> go.Figure:
    """
    Plot insider trades as a price-vs-time scatter (no stock price line). 
    
    - `insider_df` should have 'transaction_date', 'share_price' (or 'transaction_price'), 
      'trade_type', and 'officer_category'.  
    The chart is a scatter with x = transaction_date and y = price per share of that trade.
    Each point is colored green for buys and red for sells, with hover text showing officer details.
    Point size can reflect trade volume (`shares`) so that large trades stand out. 
    This lets us visually spot clusters of buying or selling over time:contentReference[oaicite:10]{index=10}.  
    Returns a Plotly Figure of this timeline chart.
    """
    fig = go.Figure()
    for trade_type, color in [('Buy','green'), ('Sell','red')]:
        df_tt = insider_df[insider_df['trade_type'] == trade_type]
        if df_tt.empty:
            continue
        size_scale = df_tt['shares'].max() if df_tt['shares'].max() > 0 else 1
        fig.add_trace(go.Scatter(
            x=df_tt['transaction_date'],
            y=df_tt['share_price'],
            mode='markers',
            name=trade_type,
            marker=dict(
                color=color,
                size=df_tt['shares'] / size_scale * 20 + 5
            ),
            customdata=df_tt['shares'],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>"
                          "Price: %{y:.2f}<br>"
                          "Shares: %{customdata:,.0f}<br>"
                          "Role: %{text}<extra></extra>",
            text=df_tt['officer_category']
        ))
    fig.update_layout(
        title="Insider trade prices",
        xaxis_title="Date",
        yaxis_title="Trade price (USD)",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig
