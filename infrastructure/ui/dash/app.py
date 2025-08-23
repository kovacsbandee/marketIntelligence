"""
Dash web application for visualizing stock price data with technical indicators.

This app allows users to input a stock symbol, loads the corresponding data from the database
using DBSymbolStorage, and displays a candlestick chart with optional MACD and RSI indicators.
A status message is shown to inform the user about the data loading process.
"""

import dash
from dash import dcc, html
import dash_mantine_components as dmc
from infrastructure.ui.dash.app_callbacks import register_callbacks

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
                        id="symbol-input",
                        placeholder="Enter symbol (e.g. MSFT)",
                        debounce=True,
                        style={"width": 300}
                    ),
                    dmc.Button(
                        "Load",
                        id="load-btn",
                        n_clicks=0,
                        style={"width": 100, "marginTop": 8}
                    ),
                ], gap=8),
                dmc.Stack([
                    dmc.Text("Start date", size="sm", mb=2, fw=700),
                    dmc.DatePicker(id="start-date-picker", style={"width": 140, "marginTop": 0}, allowDeselect=True),
                ], gap=4, style={"marginRight": 80}),
                dmc.Stack([
                    dmc.Text("End date", size="sm", mb=2, fw=700),
                    dmc.DatePicker(id="end-date-picker", style={"width": 140, "marginTop": 0}, allowDeselect=True),
                ], gap=4),
            ], gap=40, align="flex-start"),
            dmc.Notification(id="status-div", message="", color="blue", style={"margin": "10px 0"}, withCloseButton=False, action=None),
            # Tabs on the left
            dmc.Group([
                dmc.Paper(
                    dmc.Tabs(
                        id="main-tabs",
                        value="company-base",
                        orientation="vertical",
                        children=[
                            dmc.TabsList([
                                dmc.TabsTab("Company Base", value="company-base"),
                                dmc.TabsTab("Price Indicator", value="price-indicator"),
                                dmc.TabsTab("Earnings", value="earnings"),
                                dmc.TabsTab("Income Statement", value="income-statement"),
                                dmc.TabsTab("Balance Sheet", value="balance-sheet"),
                                dmc.TabsTab("Cash Flow", value="cash-flow"),
                                dmc.TabsTab("Insider Transactions", value="insider-transactions"),
                            ]),
                            dmc.TabsPanel(value="company-base", children=[
                                dmc.Box([
                                    dmc.Paper(
                                        id="company-base-content",
                                        p=10,
                                        style={"width": "100%", "minWidth": 1200, "maxWidth": 1800},
                                        children=[
                                            dmc.Text("Company base info will appear here.", c="dimmed")
                                        ]
                                    ),
                                    dmc.LoadingOverlay(
                                        id="company-base-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="price-indicator", children=[
                                dmc.Box([
                                    dmc.Paper(id="price-indicator-content", p=10),
                                    dmc.LoadingOverlay(
                                        id="price-indicator-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="earnings", children=[
                                dmc.Box([
                                    dmc.Paper(id="earnings-content", p=10, children=[
                                        dmc.Text("Earnings info will appear here.", c="dimmed")
                                    ]),
                                    dmc.LoadingOverlay(
                                        id="earnings-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="income-statement", children=[
                                dmc.Box([
                                    dmc.Paper(id="income-statement-content", p=10, children=[
                                        dmc.Text("Income statement info will appear here.", c="dimmed")
                                    ]),
                                    dmc.LoadingOverlay(
                                        id="income-statement-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="balance-sheet", children=[
                                dmc.Box([
                                    dmc.Paper(
                                        id="balance-sheet-content", 
                                        p=10, 
                                        style={"width": "100%", "minWidth": 1200, "maxWidth": 1800},
                                    ),
                                    dmc.LoadingOverlay(
                                        id="balance-sheet-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="cash-flow", children=[
                                dmc.Box([
                                    dmc.Paper(id="cash-flow-content", p=10, children=[
                                        dmc.Text("Cash flow info will appear here.", c="dimmed")
                                    ]),
                                    dmc.LoadingOverlay(
                                        id="cash-flow-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                            dmc.TabsPanel(value="insider-transactions", children=[
                                dmc.Box([
                                    dmc.Paper(id="insider-transactions-content", p=10, children=[
                                        dmc.Text("Insider transactions info will appear here.", c="dimmed")
                                    ]),
                                    dmc.LoadingOverlay(
                                        id="insider-transactions-loading",
                                        visible=False,
                                    ),
                                ])
                            ]),
                        ],
                        style={"minWidth": 300, "height": 600}
                    ),
                    shadow="sm",
                    p=0,
                    style={"minWidth": 350, "maxWidth": 1800, "width": "100%"}
                ),
            ], style={"marginTop": 30, "justifyContent": "flex-start", "width": "100%"}),
        ]),
        dcc.Store(id="price-store"),
        dcc.Store(id="dividends-store"),
        dcc.Store(id="company-base-store"),
        dcc.Store(id="q_balance-store"),
        dcc.Store(id="a_balance-store"),
    ]
)

# Register all callbacks from app_callbacks.py
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
