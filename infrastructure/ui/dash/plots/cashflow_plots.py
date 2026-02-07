
import plotly.graph_objects as go
from infrastructure.ui.dash.plot_utils import _auto_load_table_descriptions, DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT

CASHFLOW_LABELS = {
    "operating_cashflow": "Operating cash flow (cash from core business)",
    "cashflow_from_investment": "Investing cash flow (cash used for long‑term assets)",
    "cashflow_from_financing": "Financing cash flow (cash from debt/equity)",
    "net_income": "Net income (profit after taxes)",
    "capital_expenditures": "Capital expenditures (cash spent on assets)",
}

def plot_cash_flow_categories(df):
    """
    Create a bar chart comparing cash flows from operating, investing, and financing activities over time.
    
    This function takes a DataFrame containing cash flow statement data for a single company (e.g., quarterly or annual data as obtained from Alpha Vantage) and produces a Plotly bar chart. The chart will have grouped bars for each period, with each group showing:
      - Operating Cash Flow (OCF)
      - Investing Cash Flow (CFI)
      - Financing Cash Flow (CFF)
    
    **Purpose:** Visualizing these categories side-by-side allows analysts to see how each type of cash flow contributes to the company's net cash position in each period. For example, a healthy firm might show consistently positive operating cash flows (bringing in cash from operations) while investing cash flows are negative (spending cash on growth opportunities):contentReference[oaicite:20]{index=20}. Financing flows can vary between positive and negative depending on capital raising or repayments:contentReference[oaicite:21]{index=21}.
    
    **Data Requirements:** 
    - The DataFrame `df` should have at least the following columns:
        * 'operating_cashflow' – Cash from operating activities (numeric, can be positive or negative).
        * 'cashflow_from_investment' – Cash from investing activities (numeric, typically negative for investments):contentReference[oaicite:22]{index=22}.
        * 'cashflow_from_financing' – Cash from financing activities (numeric).
        * 'fiscal_date_ending' or similar date column to use for the x-axis (period labels).
    - Each row in `df` represents a period (e.g., a quarter or year). The function will use the date column for the x-axis tick labels. It's assumed that `df` is already sorted by date (oldest to newest) for a logical timeline.
    
    **Behavior:** The function will plot three bars per period (grouped by period on the x-axis). Operating cash flow is typically shown in one color (often green if positive, red if negative), investing in another, and financing in a third color for distinction. By default, the Plotly figure uses a grouped bar mode so that the OCF, CFI, and CFF bars stand side by side for each time period.
    - **Missing Data:** If any of the required cash flow columns are missing or entirely `NaN` in the DataFrame, the function will print a note to the console and skip plotting that category. For instance, if `cashflow_from_financing` is not available, it will notify the user (via `print`) and only plot operating and investing bars. This ensures we don't display an empty series in the chart.
    - The y-axis represents cash flow in the reported currency (values can be positive or negative). Positive values indicate cash inflows, while negative values indicate cash outflows. It's common for investing cash flow to be negative (cash out for investments), which in the chart will show as bars extending below the x-axis:contentReference[oaicite:23]{index=23}. 
    
    **Returns:** 
    - `plotly.graph_objs.Figure` – A Plotly Figure object containing the grouped bar chart. The figure will include appropriate titles and axis labels. The x-axis is labeled with the period (e.g., fiscal quarter or year), and the y-axis is labeled "Cash Flow (in USD)" (or the relevant currency). A legend is included to distinguish Operating, Investing, and Financing bars.
    
    **Example:** For a DataFrame `df` of quarterly cash flows for Company X:
    ```python
    fig = plot_cash_flow_categories(df)
    fig.show()
    ```
    This might produce a chart where each quarter has three bars: a green bar for OCF, a red bar for CFI (negative values going downward), and a blue bar for CFF. Analysts can quickly see trends, such as improving operating cash flow or increasing investment outflows over time.
    """
    if df is None or df.empty:
        return go.Figure()
    descriptions = _auto_load_table_descriptions(df)
    x_col = next((col for col in df.columns if 'fiscal_date' in col.lower() or 'date' in col.lower()), None)
    if not x_col:
        return go.Figure()
    x = df[x_col]
    fig = go.Figure()
    # Operating Cash Flow
    if 'operating_cashflow' in df.columns and df['operating_cashflow'].notnull().any():
        label = CASHFLOW_LABELS.get("operating_cashflow", descriptions.get('operating_cashflow', 'Operating Cash Flow'))
        fig.add_trace(go.Bar(
            x=x,
            y=df['operating_cashflow'],
            name=label,
            marker_color='green',
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    else:
        print("Note: 'operating_cashflow' column missing or empty.")
    # Investing Cash Flow
    if 'cashflow_from_investment' in df.columns and df['cashflow_from_investment'].notnull().any():
        label = CASHFLOW_LABELS.get("cashflow_from_investment", descriptions.get('cashflow_from_investment', 'Investing Cash Flow'))
        fig.add_trace(go.Bar(
            x=x,
            y=df['cashflow_from_investment'],
            name=label,
            marker_color='red',
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    else:
        print("Note: 'cashflow_from_investment' column missing or empty.")
    # Financing Cash Flow
    if 'cashflow_from_financing' in df.columns and df['cashflow_from_financing'].notnull().any():
        label = CASHFLOW_LABELS.get("cashflow_from_financing", descriptions.get('cashflow_from_financing', 'Financing Cash Flow'))
        fig.add_trace(go.Bar(
            x=x,
            y=df['cashflow_from_financing'],
            name=label,
            marker_color='blue',
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    else:
        print("Note: 'cashflow_from_financing' column missing or empty.")
    fig.update_layout(
        barmode='group',
        title={
            "text": "Cash flow categories by period",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Fiscal period",
        yaxis_title="Cash flow (USD)",
        template="plotly_white",
    width=DEFAULT_PLOTLY_WIDTH,
    height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig

def plot_operating_vs_net_income(df):
    """
    Plot an interactive line chart comparing Operating Cash Flow (OCF) to Net Income over time.
    
    This function generates a Plotly line graph with two lines: one for operating cash flow and one for net income, for each period in the provided DataFrame. It helps visualize the relationship and gaps between a company's cash generation from operations and its accounting profit.
    
    **Purpose:** Comparing OCF to net income is an industry-standard analysis to assess earnings quality:contentReference[oaicite:24]{index=24}. Under healthy conditions, these two metrics trend together, but significant divergence can indicate issues:
      - If net income is high but OCF is consistently low, it may signal that earnings are not translating into cash (potential red flags like aggressive revenue recognition or poor working capital management):contentReference[oaicite:25]{index=25}:contentReference[oaicite:26]{index=26}.
      - If OCF exceeds net income, it could mean strong cash generation (possibly due to non-cash expenses like depreciation or very effective collection of receivables):contentReference[oaicite:27]{index=27}. Consistent **positive OCF is a strong indicator of financial health**, often considered more reliable than net profit:contentReference[oaicite:28]{index=28}:contentReference[oaicite:29]{index=29}.
    
    **Data Requirements:**
    - The DataFrame `df` should contain:
        * 'operating_cashflow' – Cash flow from operations for each period.
        * 'net_income' – Net income (profit after taxes) for each period.
        * A date or period column (e.g., 'fiscal_date_ending') for the x-axis.
    - Both 'operating_cashflow' and 'net_income' should be numeric. It's expected that the periods align (e.g., each row provides both metrics for the same quarter or year).
    
    **Behavior:** The function plots two lines on the same figure:
      - **Operating Cash Flow** line (e.g., solid line in one color).
      - **Net Income** line (e.g., dashed line in a contrasting color).
    Data points are plotted in chronological order along the x-axis. The y-axis represents values in the company's reported currency. If the magnitudes of OCF and net income differ greatly, both lines will still share the same y-axis for direct comparison (consider using a secondary y-axis if needed, but by default we use one axis for clarity).
    - Each line is labeled in the legend, and hovering over the lines will show the period, OCF, and net income values.
    - **Missing Data:** If one of the required columns is missing or has no valid data, the function prints a warning message. For example, if 'net_income' is missing, it will notify the user and only plot the available 'operating_cashflow' as a single line (though comparing a single line to nothing is not very useful, so ideally the data should include both).
    
    **Returns:**
    - `plotly.graph_objs.Figure` – A Plotly Figure with the dual-line chart. The layout will include a title (e.g., "Operating Cash Flow vs Net Income"), x-axis labels for the period, and a y-axis label like "Amount (in USD)".
    
    **Interpretation:** This visualization helps stakeholders check if a company's reported earnings are backed by cash flow. Ideally, the OCF line should track at or above the net income line over time. A persistent gap where net income >> OCF could indicate **low quality of earnings** or cash flow problems:contentReference[oaicite:30]{index=30}. On the chart, large divergence between the lines would be immediately visible. By contrast, if OCF is consistently higher than net income, it suggests the company’s earnings are conservative and bolstered by strong cash collections (perhaps due to significant non-cash expenses like depreciation, which add back into OCF). Investors and analysts often view steady or rising operating cash flows as a positive sign, sometimes even more important than net profit:contentReference[oaicite:31]{index=31}.
    
    **Example:** If `df` contains yearly data for Company Y:
    ```python
    fig = plot_operating_vs_net_income(df)
    fig.show()
    ```
    The resulting line chart might show net income rising steadily from $500M to $800M over several years, while operating cash flow rises from $450M to $900M. In such a case, by the final year OCF exceeds net income – a potentially good sign that earnings are backed by cash. The docstring's references and the chart together emphasize how users can spot trends like a widening gap (which would warrant investigation into why cash isn’t keeping up with profit, or vice versa).
    """
    if df is None or df.empty:
        return go.Figure()
    descriptions = _auto_load_table_descriptions(df)
    x_col = next((col for col in df.columns if 'fiscal_date' in col.lower() or 'date' in col.lower()), None)
    if not x_col:
        return go.Figure()
    x = df[x_col]
    fig = go.Figure()
    # Operating Cash Flow
    if 'operating_cashflow' in df.columns and df['operating_cashflow'].notnull().any():
        label = CASHFLOW_LABELS.get("operating_cashflow", descriptions.get('operating_cashflow', 'Operating Cash Flow'))
        fig.add_trace(go.Scatter(
            x=x,
            y=df['operating_cashflow'],
            mode='lines+markers',
            name=label,
            line=dict(color='green', width=2),
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    else:
        print("Note: 'operating_cashflow' column missing or empty.")
    # Net Income
    if 'net_income' in df.columns and df['net_income'].notnull().any():
        label = CASHFLOW_LABELS.get("net_income", descriptions.get('net_income', 'Net Income'))
        fig.add_trace(go.Scatter(
            x=x,
            y=df['net_income'],
            mode='lines+markers',
            name=label,
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:,.0f}}<extra></extra>"
        ))
    else:
        print("Note: 'net_income' column missing or empty.")
    fig.update_layout(
        title={
            "text": "Operating cash flow vs net income",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Fiscal period",
        yaxis_title="Amount (USD)",
        template="plotly_white",
    width=DEFAULT_PLOTLY_WIDTH,
    height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig

def plot_free_cash_flow(df):
    """
    Plot a trend of Free Cash Flow (FCF) over time, optionally alongside operating cash flow or net income for context.
    
    This function computes Free Cash Flow for each period in the input DataFrame and produces a Plotly figure to visualize it. **Free Cash Flow** is defined as operating cash flow minus capital expenditures:contentReference[oaicite:32]{index=32}. It represents the cash generated by the business after investing in property, equipment, and other long-term assets – essentially, the cash that could be returned to shareholders or used for strategic initiatives.
    
    **Purpose:** FCF is a key metric for investors because it captures the company's ability to generate cash **beyond its operational needs** and mandatory investments. A consistently positive and growing FCF indicates a company has ample cash to pursue opportunities, pay dividends, or reduce debt. If FCF is much lower than operating cash flow, it means a large portion of the OCF is being reinvested into the business (which could be good for growth, but leaves less cash leftover).
    
    **Data Requirements:**
    - The DataFrame `df` must include:
        * 'operating_cashflow' – Cash flow from operating activities.
        * 'capital_expenditures' – Capital expenditures (typically a positive number in financial statements representing cash outflow for investments).
        * A time-period column (e.g., 'fiscal_date_ending') for labeling the x-axis.
    - 'capital_expenditures' should be provided as positive values (the amount spent on capex). The function will internally treat these as cash outflows by subtracting them from OCF to compute free cash flow for each period.
    - Optionally, if 'net_income' is present, the function can include it for comparison (for example, to show how free cash flow compares to accounting earnings, since FCF and net income can differ significantly).
    
    **Behavior:** For each period, the function calculates:
        FCF = operating_cashflow – capital_expenditures.
    It then plots FCF as a bar or line chart over time. By default, we'll use a bar chart for FCF values per period (since FCF can be positive or negative, bars make it easy to see when FCF dips below zero). We also differentiate the bars by color (e.g., green for positive FCF, red for negative) to quickly indicate periods of healthy vs. poor free cash flow.
    - If `include_net_income` or `include_ocf` parameters were supported (not in this simple version, but conceptually), we could overlay a line for net income or OCF to provide context. For simplicity, this function focuses on FCF alone, but the user can plot multiple metrics if needed by using other functions or Plotly features.
    - **Missing Data:** If the DataFrame lacks 'capital_expenditures' data (or if it’s all null), the function will print a message like "Note: Capital expenditure data not available – cannot compute free cash flow." and return an empty figure or a figure with just OCF (if that was a fallback). Similarly, if 'operating_cashflow' is missing, the function cannot compute FCF and will warn the user. We ensure that we do not attempt to plot an FCF series if the inputs are incomplete.
    
    **Returns:**
    - `plotly.graph_objs.Figure` – A Plotly Figure object representing the free cash flow trend. The figure will have the time period on the x-axis and FCF values on the y-axis. It includes a title (e.g., "Free Cash Flow per Quarter") and axis labels. If additional context (like net income) is included, they will be added as additional traces (with a secondary y-axis if their scale differs greatly, though usually comparing FCF and net income on one axis is fine since both are dollar amounts).
    
    **Interpretation:** This chart helps determine whether the company is generating surplus cash. For instance, if a company has high operating cash flow but equally high capital expenditures, its free cash flow might be near zero – meaning it is reinvesting all its cash into growth. Investors often watch the FCF trend closely: **rising free cash flow** over time is usually positive, as it means more cash is available for discretionary uses or shareholder returns:contentReference[oaicite:33]{index=33}. On the other hand, negative free cash flow in a period (which would show as a negative bar) isn’t inherently bad if it’s due to large one-time investments; however, a long-term pattern of negative FCF could be concerning unless accompanied by strong growth prospects.
    
    **Example:** Using quarterly data in `df` for Company Z:
    ```python
    fig = plot_free_cash_flow(df)
    fig.show()
    ```
    The output might be a bar chart where most bars are positive and growing, say from $1B to $3B over several years, except for a couple of quarters where FCF is slightly negative due to a big capex project (shown by a red bar). Such visualization quickly conveys how much real cash the company is generating after investments, complementing the understanding gained from the operating cash flow alone.
    """
    if df is None or df.empty:
        return go.Figure()
    x_col = next((col for col in df.columns if 'fiscal_date' in col.lower() or 'date' in col.lower()), None)
    if not x_col:
        return go.Figure()
    x = df[x_col]
    if 'operating_cashflow' not in df.columns or 'capital_expenditures' not in df.columns:
        print("Note: Required columns for FCF ('operating_cashflow', 'capital_expenditures') missing.")
        return go.Figure()
    fcf = df['operating_cashflow'] - df['capital_expenditures']
    colors = ['green' if v >= 0 else 'red' for v in fcf]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=fcf,
        name="Free cash flow (operating cash flow minus capex)",
        marker_color=colors,
        hovertemplate="<b>Free cash flow</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title={
            "text": "Free cash flow by period",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="Fiscal period",
        yaxis_title="Free cash flow (USD)",
        template="plotly_white",
    width=DEFAULT_PLOTLY_WIDTH,
    height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14),
        showlegend=True
    )
    return fig

