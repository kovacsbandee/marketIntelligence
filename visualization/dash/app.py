"""
Dash web application for visualizing stock price data with technical indicators.

This app allows users to input a stock symbol, loads the corresponding data from the database
using DBSymbolStorage, and displays a candlestick chart with optional MACD and RSI indicators.
A status message is shown to inform the user about the data loading process.
"""

import dash
from dash import dcc, html, Input, Output, State
from data_manager.src_postgre_db.db_etl_jobs.db_load_from_db_runner import DBSymbolStorage
from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
from visualization.base_plots import plot_price_with_indicators

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Stock Price with Indicators"),
    dcc.Input(id="symbol-input", type="text", placeholder="Enter symbol (e.g. MSFT)", debounce=True),
    html.Button("Load", id="load-btn", n_clicks=0),
    html.Div(id="status-div", style={"margin": "10px 0", "color": "blue"}),
    html.Div(id="output-div")
])

@app.callback(
    Output("status-div", "children"),
    Output("output-div", "children"),
    Input("load-btn", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True
)
def load_symbol(n_clicks, symbol):
    """
    Callback to load and display stock data for the given symbol.

    Args:
        n_clicks (int): Number of times the Load button has been clicked.
        symbol (str): The stock symbol entered by the user.

    Returns:
        Tuple[str, dash.development.base_component.Component]:
            - Status message about the data loading process.
            - Plotly graph component with the price and indicators, or None if no data.
    """
    if not symbol:
        return "Please enter a symbol.", None
    adapter = PostgresAdapter()
    storage = DBSymbolStorage(adapter, symbol.upper())
    price_df = storage.get_table("daily_timeseries")
    if price_df is None or price_df.empty:
        return f"No data found for symbol '{symbol.upper()}'.", None
    dividends_df = storage.get_table("dividends")
    fig = plot_price_with_indicators(
        price_table=price_df,
        dividend_table=dividends_df,
        include_macd=True,
        include_rsi=True
    )
    return f"Data loaded for symbol '{symbol.upper()}'.", dcc.Graph(figure=fig)

if __name__ == "__main__":
    app.run(debug=True)