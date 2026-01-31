"""
Dash web application for visualizing stock price data with technical indicators.

This app allows users to input a stock symbol, loads the corresponding data from the database
using DBSymbolStorage, and displays a candlestick chart with optional MACD and RSI indicators.
A status message is shown to inform the user about the data loading process.
"""

import dash
from dash import dcc
import dash_mantine_components as dmc

from infrastructure.ui.dash.app_callbacks import register_callbacks
from infrastructure.ui.dash.ids import Ids, Tabs
from infrastructure.ui.dash.components import tab_panel

app = dash.Dash(
    __name__,
    external_stylesheets=["https://unpkg.com/@mantine/ds@latest/styles.css"]
)

# --- Mantine Default Theme ---
mantine_theme = {
    "colorScheme": "light",
    "primaryColor": "blue",
    "fontFamily": "Inter, sans-serif",
}

app.layout = dmc.MantineProvider(
    theme=mantine_theme,
    children=[
        dmc.Paper([
            dmc.Text("Market Intelligence Dashboard", size="xl", fw=700, mb=20),
            # Global Controls
            dmc.Group([
                dmc.Stack([
                    dmc.TextInput(
                        id=Ids.SYMBOL_INPUT,
                        placeholder="Enter symbol (e.g. MSFT)",
                        debounce=True,
                        style={"width": 300},
                        withAsterisk=True,
                        inputProps={"autoFocus": True},
                    ),
                    dmc.Button(
                        "Load",
                        id=Ids.LOAD_BUTTON,
                        n_clicks=0,
                        style={"width": 120, "marginTop": 8},
                        loading=False,
                    ),
                ], gap=8),
                dmc.Stack([
                    dmc.Text("Start date", size="sm", mb=2, fw=700),
                    dmc.DatePicker(id=Ids.START_DATE_PICKER, style={"width": 160, "marginTop": 0}, allowDeselect=True),
                ], gap=4, style={"marginRight": 60}),
                dmc.Stack([
                    dmc.Text("End date", size="sm", mb=2, fw=700),
                    dmc.DatePicker(id=Ids.END_DATE_PICKER, style={"width": 160, "marginTop": 0}, allowDeselect=True),
                ], gap=4),
                dmc.Stack([
                    dmc.Text("Quick ranges", size="sm", mb=2, fw=700),
                    dmc.SegmentedControl(
                        id=Ids.RANGE_PRESET,
                        data=[
                            {"label": "6M", "value": "6m"},
                            {"label": "1Y", "value": "1y"},
                            {"label": "YTD", "value": "ytd"},
                            {"label": "Max", "value": "max"},
                        ],
                        value=None,
                        size="sm",
                        radius="sm",
                    ),
                ], gap=4),
            ], gap=30, align="flex-start", wrap="wrap"),
            dmc.Notification(id=Ids.STATUS_DIV, message="", color="blue", style={"margin": "10px 0"}, withCloseButton=False, action=None),
            # Tabs on the left
            dmc.Group([
                dmc.Paper(
                    dmc.Tabs(
                        id=Ids.MAIN_TABS,
                        value=Tabs.COMPANY_BASE,
                        orientation="vertical",
                        children=[
                            dmc.TabsList([
                                *(dmc.TabsTab(label, value=value) for label, value in Tabs.ITEMS)
                            ]),
                            tab_panel(
                                value=Tabs.COMPANY_BASE,
                                content_id=Ids.COMPANY_BASE_CONTENT,
                                loading_id=Ids.COMPANY_BASE_LOADING,
                                placeholder="Company base info will appear here.",
                                paper_style={"width": "100%", "minWidth": 1200, "maxWidth": 1800},
                            ),
                            tab_panel(
                                value=Tabs.PRICE_INDICATOR,
                                content_id=Ids.PRICE_INDICATOR_CONTENT,
                                loading_id=Ids.PRICE_INDICATOR_LOADING,
                            ),
                            tab_panel(
                                value=Tabs.EARNINGS,
                                content_id=Ids.EARNINGS_CONTENT,
                                loading_id=Ids.EARNINGS_LOADING,
                                placeholder="Earnings info will appear here.",
                            ),
                            tab_panel(
                                value=Tabs.INCOME_STATEMENT,
                                content_id=Ids.INCOME_STATEMENT_CONTENT,
                                loading_id=Ids.INCOME_STATEMENT_LOADING,
                                placeholder="Income statement info will appear here.",
                            ),
                            tab_panel(
                                value=Tabs.BALANCE_SHEET,
                                content_id=Ids.BALANCE_SHEET_CONTENT,
                                loading_id=Ids.BALANCE_SHEET_LOADING,
                                paper_style={"width": "100%", "minWidth": 1200, "maxWidth": 1800},
                            ),
                            tab_panel(
                                value=Tabs.CASH_FLOW,
                                content_id=Ids.CASH_FLOW_CONTENT,
                                loading_id=Ids.CASH_FLOW_LOADING,
                                placeholder="Cash flow info will appear here.",
                            ),
                            tab_panel(
                                value=Tabs.INSIDER_TRANSACTIONS,
                                content_id=Ids.INSIDER_CONTENT,
                                loading_id=Ids.INSIDER_LOADING,
                                placeholder="Insider transactions info will appear here.",
                            ),
                        ],
                        style={"minWidth": 300, "height": 600}
                    ),
                    shadow="sm",
                    p=0,
                    style={"minWidth": 350, "maxWidth": 1800, "width": "100%"}
                ),
            ], style={"marginTop": 30, "justifyContent": "flex-start", "width": "100%"}),
        ]),
    dcc.Store(id=Ids.PRICE_STORE),
    dcc.Store(id=Ids.DIVIDENDS_STORE),
    dcc.Store(id=Ids.COMPANY_BASE_STORE),
    dcc.Store(id=Ids.Q_BALANCE_STORE),
    dcc.Store(id=Ids.A_BALANCE_STORE),
    dcc.Store(id=Ids.EARNINGS_STORE),
    dcc.Store(id=Ids.Q_INCOME_STORE),
    dcc.Store(id=Ids.CASHFLOW_STORE),
    dcc.Store(id=Ids.INSIDER_STORE)
    ]
)

# Register all callbacks from app_callbacks.py
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
