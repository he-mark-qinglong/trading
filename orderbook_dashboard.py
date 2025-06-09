# app.py
import pandas as pd
from db_client import OrderbookWALClient
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas_ta as ta

from plotly.subplots import make_subplots

# —— 配置 —— 
DB_PATH = "ETH-USDT-SWAP.db"
TABLE   = "orderbook_snap"
TOP_N   = 5
LIMIT   = 500
REFRESH_INTERVAL_MS = 5 * 1000  # 5 秒

# —— 数据读取 & 图表生成 —— 
def read_orderbook_df(db_path: str,
                      table: str,
                      limit: int,
                      top_n: int) -> pd.DataFrame:
    """
    直接取出 ts,bid1..bidN,ask1..askN,sum_bid,sum_ask,obpi。
    """
    cli = OrderbookWALClient(db_path=db_path,
                              table=table,
                              primary_key="ts",
                              top_n=top_n)
    df = cli.read_df(limit=limit, order_by="ts DESC")
    if df.empty:
        return df
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="s")
    return df

def make_orderbook_figure(df: pd.DataFrame,
                          symbol: str,
                          top_n: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]],
                        subplot_titles=(f"{symbol} Top{top_n} Orderbook 快照",))
    # OBPI 折线
    obpi = ta.rma(df["obpi"], length=30)
    fig.add_trace(
        go.Scatter(x=df["datetime"],
                   y=obpi,
                   mode="lines",
                   name="OBPI",
                   line=dict(color="orange", width=2)),
        secondary_y=False
    )
    # SumBid / SumAsk 柱状
    # fig.add_trace(
    #     go.Bar(x=df["datetime"],
    #            y=df["sum_bid"],
    #            name="SumBid",
    #            marker_color="green", opacity=0.4),
    #     secondary_y=True
    # )
    # fig.add_trace(
    #     go.Bar(x=df["datetime"],
    #            y=df["sum_ask"],
    #            name="SumAsk",
    #            marker_color="red", opacity=0.4),
    #     secondary_y=True
    # )

    fig.update_layout(
        xaxis=dict(title="Time"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=40),
        height=450
    )
    fig.update_yaxes(title_text="OBPI",   secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    return fig

# —— Dash App —— 
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="ob-graph"),
    dcc.Interval(id="interval",
                 interval=REFRESH_INTERVAL_MS,
                 n_intervals=0)
])

@app.callback(
    Output("ob-graph", "figure"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    df = read_orderbook_df(DB_PATH, TABLE, LIMIT, TOP_N)
    if df.empty:
        return go.Figure()  # 空图
    return make_orderbook_figure(df, symbol="ETH-USDT-SWAP", top_n=TOP_N)

if __name__ == "__main__":
    app.run(debug=True, port=8051)