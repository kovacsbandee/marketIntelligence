
import plotly.graph_objects as go
from price_data_loader.load_price_data_alpha_vantage import LoadExampleData

loader = LoadExampleData()
loader.load()

loader.balance_sheet_annual

df = loader.daily_time_series
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])

fig.show()
