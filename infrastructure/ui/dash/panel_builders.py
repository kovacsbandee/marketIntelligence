"""Pure-ish panel builders used by Dash callbacks.

Each builder takes store records (and related params) and returns a tuple
``(children, loading_visible)`` matching the callback outputs. This keeps
callback bodies thin and makes the logic easy to unit test without Dash
callback wiring.
"""
from __future__ import annotations

import pandas as pd
import dash_mantine_components as dmc
from dash import dcc
import plotly.graph_objects as go

from infrastructure.ui.dash.app_util import get_last_6_months_range, records_to_df, guard_store, filter_by_date
from infrastructure.ui.dash.plots.company_fundamentals_plots import plot_company_fundamentals_table
from infrastructure.ui.dash.plots.daily_timeseries_plots import (
    plot_candlestick_chart,
    add_moving_averages_to_candlestick,
    add_bollinger_bands_to_candlestick,
    plot_candlestick_with_vwap,
    plot_rsi,
    plot_macd,
    plot_stochastic,
    plot_obv,
    plot_adx,
)
from infrastructure.ui.dash.plots.earnings_plots import (
    plot_eps_actual_vs_estimate,
    plot_eps_surprise_percentage,
    plot_eps_actual_vs_estimate_scatter,
)
from infrastructure.ui.dash.plots.income_plots import (
    plot_key_metrics_dashboard,
    plot_quarterly_profit_margins,
    plot_expense_breakdown_vs_revenue,
    plot_income_statement_waterfall,
    plot_operating_profit_ebit_ebitda_trends,
    plot_expense_growth_scatter,
    plot_tax_and_interest_effects,
    plot_metric_vs_future_stock_return,
    plot_quarterly_revenue_net_income_vs_stock_price,
)
from infrastructure.ui.dash.plots.balance_sheet_plots import (
    plot_balance_sheet_time_series,
    plot_balance_sheet_stacked_area,
    plot_balance_sheet_bar,
    plot_balance_sheet_pie,
    render_balance_sheet_metric_cards,
)
from infrastructure.ui.dash.plots.cashflow_plots import (
    plot_cash_flow_categories,
    plot_operating_vs_net_income,
    plot_free_cash_flow,
)
from infrastructure.ui.dash.plots.insider_transactions_plots import (
    prepare_insider_data,
    plot_insider_price_chart,
    plot_insider_transactions_over_time,
)

# Shared alias for Dash store payloads
Records = list[dict] | None


def build_company_base_panel(company_fundamentals: Records, symbol_input: str | None):
    """Render the company fundamentals table panel.

    Returns a (children, loading_visible) tuple. If data is missing, returns a guard response.
    """
    if (guard := guard_store(company_fundamentals, "No company info loaded.")):
        return guard

    company_fundamentals_df = records_to_df(company_fundamentals, table="company_fundamentals")
    company_fundamentals_row = company_fundamentals_df.iloc[[0]]
    symbol = (symbol_input or "").upper()
    fig = plot_company_fundamentals_table(company_fundamentals_row, symbol)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}), False


def build_price_panel(daily_timeseries: Records, start_date: str, end_date: str):
    """Render price/indicator plots for the selected date range."""
    if (guard := guard_store(daily_timeseries, "No price data loaded.")):
        return guard

    daily_timeseries_df = records_to_df(daily_timeseries, table="daily_timeseries")
    if not start_date or not end_date:
        start_date, end_date = get_last_6_months_range(daily_timeseries_df)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if start_dt > end_dt:
        return dmc.Text("Start date must be on or before end date.", c="red"), False

    selected_timeseries = filter_by_date(daily_timeseries_df, start_dt, end_dt, "date")

    fig = plot_candlestick_chart(selected_timeseries)
    fig.update_layout(xaxis_range=[start_dt, end_dt])
    fig_ma = add_moving_averages_to_candlestick(selected_timeseries)
    fig_ma.update_layout(xaxis_range=[start_dt, end_dt])
    fig_bb = add_bollinger_bands_to_candlestick(selected_timeseries)
    fig_bb.update_layout(xaxis_range=[start_dt, end_dt])
    fig_vwap = plot_candlestick_with_vwap(selected_timeseries)
    fig_vwap.update_layout(xaxis_range=[start_dt, end_dt])
    fig_rsi = plot_rsi(selected_timeseries)
    fig_rsi.update_layout(xaxis_range=[start_dt, end_dt])
    fig_macd = plot_macd(selected_timeseries)
    fig_stoch = plot_stochastic(selected_timeseries)
    fig_obv = plot_obv(selected_timeseries)
    fig_adx = plot_adx(selected_timeseries)

    return dmc.Accordion(
        multiple=True,
        value=["price", "trend"],
        children=[
            dmc.AccordionItem([
                dmc.AccordionControl("Price"),
                dmc.AccordionPanel(
                    dmc.Stack([
                        dmc.Text("Candlestick Chart", fw=600, size="sm"),
                        dcc.Graph(figure=fig),
                    ], gap=8)
                ),
            ], value="price"),
            dmc.AccordionItem([
                dmc.AccordionControl("Momentum"),
                dmc.AccordionPanel(
                    dmc.Stack([
                        dmc.Text("RSI", fw=600, size="sm"),
                        dcc.Graph(figure=fig_rsi),
                        dmc.Text("MACD", fw=600, size="sm", mt=12),
                        dcc.Graph(figure=fig_macd),
                        dmc.Text("Stochastic Oscillator", fw=600, size="sm", mt=12),
                        dcc.Graph(figure=fig_stoch),
                    ], gap=10)
                ),
            ], value="momentum"),
            dmc.AccordionItem([
                dmc.AccordionControl("Moving Averages & Bands"),
                dmc.AccordionPanel(
                    dmc.Stack([
                        dmc.Text("Candlestick with Moving Averages", fw=600, size="sm"),
                        dcc.Graph(figure=fig_ma),
                        dmc.Text("Candlestick with Bollinger Bands", fw=600, size="sm", mt=12),
                        dcc.Graph(figure=fig_bb),
                        dmc.Text("Candlestick with VWAP", fw=600, size="sm", mt=12),
                        dcc.Graph(figure=fig_vwap),
                    ], gap=10)
                ),
            ], value="trend"),
            dmc.AccordionItem([
                dmc.AccordionControl("Volume & Trend Strength"),
                dmc.AccordionPanel(
                    dmc.Stack([
                        dmc.Text("On-Balance Volume (OBV)", fw=600, size="sm"),
                        dcc.Graph(figure=fig_obv),
                        dmc.Text("Average Directional Index (ADX)", fw=600, size="sm", mt=12),
                        dcc.Graph(figure=fig_adx),
                    ], gap=10)
                ),
            ], value="volume"),
        ],
    ), False


def build_earnings_panel(earnings_data: Records, symbol_input: str | None):
    """Render earnings-related plots (EPS vs estimate, surprises)."""
    if (guard := guard_store(earnings_data, "No earnings data loaded.")):
        return guard

    earnings_df = records_to_df(earnings_data, table="earnings")
    symbol = (symbol_input or "").upper()
    fig_eps_vs_est = plot_eps_actual_vs_estimate(symbol, earnings_df)
    fig_surprise_pct = plot_eps_surprise_percentage(symbol, earnings_df)
    fig_eps_scatter = plot_eps_actual_vs_estimate_scatter(symbol, earnings_df)
    return dmc.Stack([
        dmc.Divider(label="EPS Actual vs Estimate", my=10),
        dcc.Graph(figure=fig_eps_vs_est),
        dmc.Divider(label="EPS Surprise Percentage", my=10),
        dcc.Graph(figure=fig_surprise_pct),
        dmc.Divider(label="EPS Actual vs Estimate Scatter", my=10),
        dcc.Graph(figure=fig_eps_scatter),
    ], gap=16), False


def build_income_statement_panel(price_data: Records, income_statement_quarterly: Records, symbol_input: str | None):
    """Render income-statement visuals alongside price data."""
    if (guard := guard_store(income_statement_quarterly, "No income statement data loaded.")):
        return guard
    if (guard := guard_store(price_data, "No price data loaded.")):
        return guard

    income_df = records_to_df(income_statement_quarterly, table="income_statement_quarterly")
    price_df = records_to_df(price_data, table="daily_timeseries")
    symbol = (symbol_input or "").upper()
    fig_dashboard = plot_key_metrics_dashboard(symbol, income_df, price_df)
    fig_margins = plot_quarterly_profit_margins(symbol, income_df)
    fig_expenses = plot_expense_breakdown_vs_revenue(symbol, income_df)
    fig_expense_growth = plot_expense_growth_scatter(symbol, income_df)
    fig_tax_interest = plot_tax_and_interest_effects(symbol, income_df)
    fig_op_trends = plot_operating_profit_ebit_ebitda_trends(symbol, income_df)
    fig_waterfall = plot_income_statement_waterfall(symbol, income_df)
    fig = plot_quarterly_revenue_net_income_vs_stock_price(symbol, income_df, price_df)
    default_metric = "net_income" if "net_income" in income_df.columns else income_df.columns[0]
    fig_metric_vs_return = plot_metric_vs_future_stock_return(symbol, income_df, price_df, default_metric)
    return dmc.Stack([
        dmc.Divider(label="Key Metrics Dashboard", my=10),
        dcc.Graph(figure=fig_dashboard),
        dmc.Divider(label="Quarterly Profit Margins", my=10),
        dcc.Graph(figure=fig_margins),
        dmc.Divider(label="Expense Breakdown vs Revenue", my=10),
        dcc.Graph(figure=fig_expenses),
        dmc.Divider(label="Expense Growth vs Revenue (Bubble Chart)", my=10),
        dcc.Graph(figure=fig_expense_growth),
        dmc.Divider(label="Tax & Interest Effects on Pre-Tax and Net Income", my=10),
        dcc.Graph(figure=fig_tax_interest),
        dmc.Divider(label="Operating Profit, EBIT & EBITDA Trends", my=10),
        dcc.Graph(figure=fig_op_trends),
        dmc.Divider(label="Income Statement Waterfall (Most Recent Quarter)", my=10),
        dcc.Graph(figure=fig_waterfall),
        dmc.Divider(label="Quarterly Revenue, Net Income & Stock Price", my=10),
        dcc.Graph(figure=fig),
        dmc.Divider(label="Metric vs Future Stock Return (Quarterly)", my=10),
        dcc.Graph(figure=fig_metric_vs_return),
    ], gap=16), False


def build_balance_sheet_panel(balance_sheet_quarterly: Records, start_date: str, end_date: str):
    """Render balance sheet cards and charts for the selected date range."""
    if (guard := guard_store(balance_sheet_quarterly, "No balance sheet data loaded.")):
        return guard

    balance_sheet_quarterly_df = records_to_df(balance_sheet_quarterly, table="balance_sheet_quarterly")
    if balance_sheet_quarterly_df.empty or "fiscal_date_ending" not in balance_sheet_quarterly_df.columns:
        return dmc.Text("No balance sheet data loaded.", c="red"), False

    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            return dmc.Text("Start date must be on or before end date.", c="red"), False

    selected_balance_sheet = filter_by_date(balance_sheet_quarterly_df, start_date, end_date, "fiscal_date_ending")

    if selected_balance_sheet.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No balance sheet data available for the selected date range.",
            xaxis_title="Fiscal Date Ending",
            yaxis_title="Value"
        )
        return dcc.Graph(figure=fig), False

    latest_date = str(selected_balance_sheet["fiscal_date_ending"].max().date())

    time_series_fig = plot_balance_sheet_time_series(selected_balance_sheet)
    stacked_area_fig = plot_balance_sheet_stacked_area(selected_balance_sheet)
    bar_fig = plot_balance_sheet_bar(selected_balance_sheet)
    pie_fig = plot_balance_sheet_pie(selected_balance_sheet, latest_date)
    cards = render_balance_sheet_metric_cards(selected_balance_sheet, latest_date)

    return dmc.Stack([
        dmc.Group(cards, justify="flex-start", gap="xs"),
        dmc.Divider(label="Time Series", my=10),
        dcc.Graph(figure=time_series_fig),
        dmc.Divider(label="Stacked Area", my=10),
        dcc.Graph(
            figure=stacked_area_fig,
            style={"width": "100vw", "minWidth": "1200px", "height": "700px"},
            config={"responsive": True}
        ),
        dmc.Divider(label="Bar Chart", my=10),
        dcc.Graph(figure=bar_fig),
        dmc.Divider(label="Pie Chart", my=10),
        dcc.Graph(figure=pie_fig),
    ], gap=16), False


def build_cash_flow_panel(cashflow_data: Records, start_date: str, end_date: str):
    """Render cash-flow plots for the selected date range."""
    if (guard := guard_store(cashflow_data, "No cash flow data loaded.")):
        return guard
    cashflow_df = records_to_df(cashflow_data, table="cashflow_statement_quarterly")
    if cashflow_df.empty:
        return dmc.Text("No cash flow data loaded.", c="red"), False
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            return dmc.Text("Start date must be on or before end date.", c="red"), False
    date_col = next((col for col in cashflow_df.columns if 'fiscal_date' in col.lower() or 'date' in col.lower()), None)
    selected_cashflow = filter_by_date(cashflow_df, start_date, end_date, date_col)
    if selected_cashflow is None or selected_cashflow.empty:
        selected_cashflow = cashflow_df

    fig_categories = plot_cash_flow_categories(selected_cashflow)
    fig_ocf_vs_ni = plot_operating_vs_net_income(selected_cashflow)
    fig_fcf = plot_free_cash_flow(selected_cashflow)
    return dmc.Stack([
        dmc.Divider(label="Cash Flow Categories (Operating, Investing, Financing)", my=10),
        dcc.Graph(figure=fig_categories),
        dmc.Divider(label="Operating Cash Flow vs Net Income", my=10),
        dcc.Graph(figure=fig_ocf_vs_ni),
        dmc.Divider(label="Free Cash Flow per Period", my=10),
        dcc.Graph(figure=fig_fcf),
    ], gap=16), False


def build_insider_panel(insider_transactions: Records, price_data: Records, start_date: str, end_date: str):
    """Render insider transaction visuals aligned with price data."""
    if (guard := guard_store(insider_transactions, "No insider transactions data available.", color="dimmed")):
        return guard
    if (guard := guard_store(price_data, "No price data available for insider transactions plot.", color="dimmed")):
        return guard

    insider_df = records_to_df(insider_transactions, table="insider_transactions")
    price_df = records_to_df(price_data, table="daily_timeseries")
    if insider_df.empty:
        return dmc.Text("No insider transactions available for this symbol.", c="dimmed"), False
    if price_df.empty:
        return dmc.Text("No price data available for this symbol.", c="dimmed"), False

    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            return dmc.Text("Start date must be on or before end date.", c="dimmed"), False

    insider_df = filter_by_date(insider_df, start_date, end_date, "transaction_date")
    price_df = filter_by_date(price_df, start_date, end_date, "date")

    if insider_df.empty:
        return dmc.Text("No insider transactions in selected date range.", c="dimmed"), False
    if price_df.empty:
        return dmc.Text("No price data in selected date range.", c="dimmed"), False

    insider_df = prepare_insider_data(insider_df)
    fig1 = plot_insider_price_chart(price_df, insider_df)
    fig2 = plot_insider_transactions_over_time(insider_df)
    content = dmc.Stack([
        dmc.Text("Insider Trades on Price Chart", size="md", fw=700),
        dcc.Graph(figure=fig1, style={"height": 400}),
        dmc.Text("Insider Transaction Prices Over Time", size="md", fw=700, mt=20),
        dcc.Graph(figure=fig2, style={"height": 400}),
    ], gap=24)
    return content, False
