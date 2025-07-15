
import os
import pandas as pd
import numpy as np
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import LHFrameStd
from db_client import SQLiteWALClient
from db_read import read_and_sort_df

from dynamic_kama import compute_dynamic_kama

BASIC_INTERVAL = 5
use30x = True
symbol = "ETH-USDT-SWAP"
DB_PATH = f'{symbol}.db'
client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")

windowConfig = LHFrameStd.WindowConfig()
LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 310)
LIMIT_K_N = 1000 + LIMIT_K_N_APPEND + 3000
LIMIT_K_N += 50000

app = Dash(__name__)
app.layout = html.Div([
    html.H2(f"OKX {30*BASIC_INTERVAL if use30x else BASIC_INTERVAL}s K-line OHLCV (Auto-refresh)"),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(
        id='interval',
        interval=(30 * BASIC_INTERVAL if use30x else BASIC_INTERVAL) * 1000,
        n_intervals=0
    ),
])

@app.callback(
    Output("kline-graph", "figure"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    # 1. 读数据
    df = read_and_sort_df(client, LIMIT_K_N)
    start = df.first_valid_index()
    if start is None:
        return go.Figure()
    df = df.loc[start:].copy()
    df["datetime"] = pd.to_datetime(df.index, unit="s")

    # 2. 计算 KAMA1 & KAMA2
    kama_params = dict(
        src_col="close",
        len_er=100,
        fast=24,
        slow2fast_times=2.0,
        slow=168,
        intervalP=0.01,
        minLen=10,
        maxLen=60,
        volLen=30
    )
    df_k = compute_dynamic_kama(df, **kama_params)

    # Masks for area fill
    mask_up = df_k["kama1"] >= df_k["kama2"]
    mask_dn = ~mask_up

    # 3. 绘制子图
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # 3.1 K 线
    fig.add_trace(go.Candlestick(
        x=df_k["datetime"], open=df_k["open"], high=df_k["high"],
        low=df_k["low"], close=df_k["close"], name="K-line"
    ), row=1, col=1)

    # 3.2 KAMA1 & KAMA2 lines
    fig.add_trace(go.Scatter(
        x=df_k["datetime"], y=df_k["kama1"],
        mode="lines", name="KAMA1", line=dict(color="green", width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_k["datetime"], y=df_k["kama2"],
        mode="lines", name="KAMA2", line=dict(color="blue", width=2)
    ), row=1, col=1)

    # 3.3 Green fill where kama1 >= kama2
    # lower bound for fill
    fig.add_trace(go.Scatter(
        x=df_k["datetime"][mask_up],
        y=df_k["kama2"][mask_up],
        mode="lines", line=dict(width=0),
        showlegend=False
    ), row=1, col=1)
    # upper bound with fill
    fig.add_trace(go.Scatter(
        x=df_k["datetime"][mask_up],
        y=df_k["kama1"][mask_up],
        mode="lines", fill='tonexty',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(width=0), name="KAMA1>KAMA2"
    ), row=1, col=1)

    # 3.4 Red fill where kama1 < kama2
    fig.add_trace(go.Scatter(
        x=df_k["datetime"][mask_dn],
        y=df_k["kama1"][mask_dn],
        mode="lines", line=dict(width=0),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_k["datetime"][mask_dn],
        y=df_k["kama2"][mask_dn],
        mode="lines", fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(width=0), name="KAMA1<KAMA2"
    ), row=1, col=1)

    # 4. 布局
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin={"t": 40, "b": 40, "l": 20, "r": 20},
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=9050)
