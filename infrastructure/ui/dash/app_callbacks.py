"""
Dash app callback registrations for the Market Intelligence dashboard UI.

This module defines all Dash callback functions for updating UI components
in response to user interactions, including data loading, chart updates,
and tab content rendering. It connects the UI to the data service and
plotting utilities, and manages the state and interactivity of the dashboard.

Notes on guards & filtering:
- `guard_store` provides a single, consistent early-return for missing/empty store data to avoid noisy stack traces and show user-friendly messages.
- `filter_by_date` centralizes inclusive date-range filtering and no-ops when dates/columns are absent, keeping callbacks concise and defensive.
"""

import dash
import pandas as pd
from typing import Any
from dash import Output, Input, State, no_update
import dash_mantine_components as dmc
from dash.development.base_component import Component

from infrastructure.ui.dash.data_service import load_symbol_data

from infrastructure.ui.dash.app_util import empty_load, Records, records_to_df, get_last_6_months_range, filter_by_date
from infrastructure.ui.dash.ids import Ids, Tabs
from infrastructure.ui.dash.panel_builders import (
    build_balance_sheet_panel,
    build_cash_flow_panel,
    build_company_base_panel,
    build_earnings_panel,
    build_income_statement_panel,
    build_insider_panel,
    build_price_panel,
)
from infrastructure.ui.dash.plots.daily_timeseries_plots import plot_candlestick_with_overlays

def register_callbacks(app: dash.Dash) -> None:

    @app.callback(
        Output(Ids.START_DATE_PICKER, "value"),
        Output(Ids.END_DATE_PICKER, "value"),
        Output(Ids.STATUS_DIV, "title"),
        Output(Ids.PRICE_STORE, "data"),
        Output(Ids.DIVIDENDS_STORE, "data"),
        Output(Ids.COMPANY_BASE_STORE, "data"),
        Output(Ids.Q_BALANCE_STORE, "data"),
        Output(Ids.A_BALANCE_STORE, "data"),
        Output(Ids.EARNINGS_STORE, "data"),
        Output(Ids.Q_INCOME_STORE, "data"),
        Output(Ids.CASHFLOW_STORE, "data"),
        Output(Ids.INSIDER_STORE, "data"),
        Input(Ids.LOAD_BUTTON, "n_clicks"),
        Input(Ids.SYMBOL_INPUT, "n_submit"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def load_symbol(
        n_clicks: int,
        n_submit: int,
        symbol: str
    ) -> tuple:
        """
        Callback to load all relevant data for a given symbol and update stores and UI, including earnings data.
        """
        if not symbol:
            return empty_load("Please enter a symbol.")
        data = load_symbol_data(symbol)
        if data["daily_timeseries"] is None:
            return empty_load(data["status_message"])

        return (
            data["start_date"],
            data["end_date"],
            data["status_message"],
            data["daily_timeseries"],
            data["dividends"],
            data["company_fundamentals"],
            data["balance_sheet_quarterly"],
            data["annual_balance_sheet"],
            data["earnings"],
            data["income_statement_quarterly"],
            data["cashflow_statement_quarterly"],
            data["insider_transactions"],
        )

    @app.callback(
        Output(Ids.COMPANY_BASE_CONTENT, "children"),
        Output(Ids.COMPANY_BASE_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.COMPANY_BASE_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_company_base(
        tab: str,
    company_fundamentals: Records,
        symbol_input: str | None
    ) -> tuple[Component, bool]:
        """
        Callback to update the company base info panel with a Plotly table.

        Args:
            tab (str): The currently selected tab.
            company_base_data (dict): The company base data from the store.

        Returns:
            tuple: Updated content and loading state for the company base panel.
        """
        if tab != Tabs.COMPANY_BASE:
            return no_update, False
        try:
            return build_company_base_panel(company_fundamentals, symbol_input)
        except Exception as e:
            return dmc.Text(f"Error displaying company fundamentals: {e}", c="red"), False

    @app.callback(
        Output(Ids.PRICE_INDICATOR_CONTENT, "children"),
        Output(Ids.PRICE_INDICATOR_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True
    )
    def price_charts_and_indicators(
        tab: str,
        start_date: str,
        end_date: str,
    daily_timeseries: Records
    ) -> tuple[Component]:
        """
        Callback to update the price indicator panel with a plot.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            price_data (list[dict]): Price data from the store.
            dividends_data (list[dict]): Dividends data from the store.

        Returns:
            tuple: Updated content and loading state for the price indicator panel.
        """
        if tab != Tabs.PRICE_INDICATOR:
            return no_update, False
        try:
            return build_price_panel(daily_timeseries, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False

    @app.callback(
        Output(Ids.PRICE_CHART, "figure"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.PRICE_STORE, "data"),
        Input(Ids.PRICE_OVERLAY_TOGGLE, "value"),
        prevent_initial_call=True
    )
    def update_price_chart_overlays(
        tab: str,
        start_date: str,
        end_date: str,
        daily_timeseries: Records,
        overlays: list[str] | None,
    ) -> Any:
        """Toggle overlays on the primary price chart without removing other panels."""
        if tab != Tabs.PRICE_INDICATOR:
            return no_update

        overlays = overlays or []
        if not daily_timeseries:
            return {"data": [], "layout": {"title": "No price data loaded."}}

        df = records_to_df(daily_timeseries, table="daily_timeseries")
        if df.empty:
            return {"data": [], "layout": {"title": "No price data loaded."}}

        if not start_date or not end_date:
            start_date, end_date = get_last_6_months_range(df)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            return {"data": [], "layout": {"title": "Start date must be on or before end date."}}

        selected_timeseries = filter_by_date(df, start_dt, end_dt, "date")
        if selected_timeseries.empty:
            return {"data": [], "layout": {"title": "No price data in selected date range."}}

        fig = plot_candlestick_with_overlays(
            selected_timeseries,
            show_ma="ma" in overlays,
            show_bb="bb" in overlays,
            show_vwap="vwap" in overlays,
        )
        fig.update_layout(xaxis_range=[start_dt, end_dt])
        return fig

    @app.callback(
        Output(Ids.EARNINGS_CONTENT, "children"),
        Output(Ids.EARNINGS_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.EARNINGS_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_earnings_panel(
        tab: str,
    earnings_data: Records,
        symbol_input: str | None
    ):
        if tab != Tabs.EARNINGS:
            return no_update, False
        try:
            return build_earnings_panel(earnings_data, symbol_input)
        except Exception as e:
            return dmc.Text(f"Error displaying earnings plots: {e}", c="red"), False

    @app.callback(
        Output(Ids.INCOME_STATEMENT_CONTENT, "children"),
        Output(Ids.INCOME_STATEMENT_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.PRICE_STORE, "data"),
        Input(Ids.Q_INCOME_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_income_statement_tab(
        tab: str,
    price_data: Records,
    income_statement_quarterly: Records,
        symbol_input: str | None
    ):
        """
        Callback to update the income statement tab with all relevant plots.

        Args:
            tab (str): The currently selected tab.
            price_data (list[dict] | None): Daily timeseries price data.
            income_statement_quarterly (list[dict] | None): Quarterly income statement data.

        Returns:
            tuple: Updated content and loading state for the income statement panel.
        """
        if tab != Tabs.INCOME_STATEMENT:
            return no_update, False
        try:
            return build_income_statement_panel(price_data, income_statement_quarterly, symbol_input)
        except Exception as e:
            print(f"[DEBUG] Exception in update_income_statement_tab: {e}")
            return dmc.Text(f"Error displaying income statement plots: {e}", c="red"), False

    @app.callback(
        Output(Ids.BALANCE_SHEET_CONTENT, "children"),
        Output(Ids.BALANCE_SHEET_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.Q_BALANCE_STORE, "data"),
        prevent_initial_call=True
    )
    def update_balance_sheet(
        tab: str,
        start_date: str,
        end_date: str,
    balance_sheet_quarterly: Records
    ) -> tuple[Component, bool]:
        """
        Callback to update the balance sheet panel with a time series plot.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            balance_sheet_quarterly (list[dict]): Quarterly balance sheet data from the store.

        Returns:
            tuple: Updated content and loading state for the balance sheet panel.
        """
        if tab != Tabs.BALANCE_SHEET:
            return no_update, False
        try:
            return build_balance_sheet_panel(balance_sheet_quarterly, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False
        
    @app.callback(
        Output(Ids.CASH_FLOW_CONTENT, "children"),
        Output(Ids.CASH_FLOW_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.CASHFLOW_STORE, "data"),
        prevent_initial_call=True
    )
    def update_cash_flow(
        tab: str,
        start_date: str,
        end_date: str,
    cashflow_data: Records
    ) -> tuple[Component, bool]:
        """
        Callback to update the cash flow panel with all relevant plots.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            cashflow_data (list[dict]): Cash flow data from the store.

        Returns:
            tuple: Updated content and loading state for the cash flow panel.
        """
        if tab != Tabs.CASH_FLOW:
            return no_update, False
        try:
            return build_cash_flow_panel(cashflow_data, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating cash flow plots: {e}", c="red"), False        

    @app.callback(
        Output(Ids.INSIDER_CONTENT, "children"),
        Output(Ids.INSIDER_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.INSIDER_STORE, "data"),
        Input(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True
    )
    def update_insider_transactions(
        tab: str,
        start_date: str,
        end_date: str,
    insider_transactions: Records,
    price_data: Records
    ) -> tuple[Component, bool]:
        """
        Callback to update the insider transactions tab with relevant plots.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            insider_transactions (list[dict] | None): Insider transactions data from the store.
            price_data (list[dict] | None): Daily price data from the store.

        Returns:
            tuple: Updated content and loading state for the insider transactions panel.
        """

        if tab != Tabs.INSIDER_TRANSACTIONS:
            return no_update, False
        try:
            return build_insider_panel(insider_transactions, price_data, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error displaying insider transactions: {e}", c="red"), False

    @app.callback(
        Output(Ids.START_DATE_PICKER, "value", allow_duplicate=True),
        Output(Ids.END_DATE_PICKER, "value", allow_duplicate=True),
        Input(Ids.RANGE_PRESET, "value"),
        State(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True,
    )
    def apply_quick_range(preset: str | None, price_data: Records):
        if not preset or not price_data:
            return no_update, no_update

        df = records_to_df(price_data, table="daily_timeseries")
        if df.empty or "date" not in df.columns:
            return no_update, no_update
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max()
        earliest = df["date"].min()

        if preset == "6m":
            start = (latest - pd.Timedelta(days=182)).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "1y":
            start = (latest - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "ytd":
            start = pd.Timestamp(year=latest.year, month=1, day=1).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "max":
            start = earliest.strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        else:
            return no_update, no_update

        return start, end

    # Map button loading_state to button.loading for a spinner during requests
    app.clientside_callback(
        "function(ls){return ls && ls.is_loading;}",
        Output(Ids.LOAD_BUTTON, "loading"),
        Input(Ids.LOAD_BUTTON, "loading_state"),
    )