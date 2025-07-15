
import os
import pandas as pd
import numpy as np
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import LHFrameStd
# from yyyyy2_okx_5m import trade_coin

from db_client import SQLiteWALClient
# from db_read import read_and_sort_df
from history_kline import read_and_sort_df

# 引入我们之前写好的 KAMA 计算函数
from dynamic_kama import compute_dynamic_kama

BASIC_INTERVAL = 5
use30x = True
symbol = "ETH-USDT-SWAP"

DEBUG = False

DB_PATH = f'{symbol}.db'
client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")
trade_client = None

windowConfig = LHFrameStd.WindowConfig()
multiVwap = LHFrameStd.MultiTFvp_poc(
    window_LFrame=windowConfig.window_tau_l, 
    window_HFrame=windowConfig.window_tau_h,
    window_SFrame=windowConfig.window_tau_s
)

LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 310)
LIMIT_K_N = 1000 + LIMIT_K_N_APPEND + 3000
# LIMIT_K_N += 50000

app = Dash(__name__)
app.layout = html.Div([
    html.H2(f"OKX {30*BASIC_INTERVAL if use30x else BASIC_INTERVAL}s K-line OHLCV (Auto-refresh)"),
    dcc.ConfirmDialogProvider(
        children=html.Button("一键平仓", id="btn-close", n_clicks=0),
        id="confirm-close",
        message="⚠️ 确认要全部平仓？此操作不可撤销！"
    ),
    html.Div(id="close-status", style={"marginTop": "5px", "color": "green"}),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(
        id='interval',
        interval=(30 * BASIC_INTERVAL if use30x else BASIC_INTERVAL)*1000,
        n_intervals=0
    ),
    html.Div(id="status-msg", style={"color": "red", "marginTop": 10})
])

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    try:
        # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
        before = time.time()
        df = read_and_sort_df(client, LIMIT_K_N)
        print("read df and convert takes time:", time.time() - before)

        before = time.time()
        multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)
        print("calc vwap takes time:", time.time() - before)

        # 截断到第一个有效 SFrame_vwap_poc
        start = multiVwap.SFrame_vwap_poc.first_valid_index()
        if start is None:
            return go.Figure(), "暂无有效数据"
        df = df.loc[start:].copy()
        # 对 multiVwap 系列做同样截断
        for var in vars(multiVwap):
            ser = getattr(multiVwap, var)
            if isinstance(ser, pd.Series):
                setattr(multiVwap, var, ser.loc[start:])

        # 确保 datetime 列存在
        if "datetime" not in df:
            df["datetime"] = pd.to_datetime(df.index, unit="s")

        # --- 1.5 计算 Dynamic KAMA ---
        kama_params = dict(
            src_col="close",
            len_er=30,
            fast=6,
            slow2fast_times=2.0,
            slow=120,
            intervalP=0.01,
            minLen=10,
            maxLen=60,
            volLen=30
        )
        df_kama = compute_dynamic_kama(df, **kama_params)

        # --- 2. 子图：3 行 ---
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True, shared_yaxes=False,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("K-line + vp_poc/VWAP + KAMA", "Volume", "Anchored Momentum")
        )

        # (A) 行 1: K 线 + vp/VWAP 系列
        fig.add_trace(go.Candlestick(
            x=df["datetime"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="K-line"
        ), row=1, col=1)

        for name, color in {
            **{k:'purple'      for k in ["SFrame_vwap_poc"]},
            **{k:'magenta'     for k in ["HFrame_vwap_poc"]},
            **{k:'orangered'   for k in ["HFrame_vwap_up_poc","HFrame_vwap_down_poc"]},
            **{k:'deeppink'    for k in ["SFrame_vwap_up_getin","SFrame_vwap_down_getin"]},
            **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
            **{k:'blue'        for k in ["HFrame_vwap_up_sl2","HFrame_vwap_down_sl2"]},
            **{k:'darkslategray' for k in ["SFrame_vwap_up_sl2","SFrame_vwap_down_sl2"]},
        }.items():
            series = getattr(multiVwap, name, None)
            if isinstance(series, pd.Series):
                fig.add_trace(go.Scatter(
                    x=series.index.map(lambda ts: pd.to_datetime(ts,unit="s")),
                    y=series.values,
                    mode="lines", name=name,
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)

        # (A.5) 行1: 绘制 KAMA1/KAMA2 及区域填充
        # 线条
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"], y=df_kama["kama1"],
            mode="lines", name="KAMA1", line=dict(color="green", width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"], y=df_kama["kama2"],
            mode="lines", name="KAMA2", line=dict(color="blue", width=2)
        ), row=1, col=1)
        # 填充区域
        mask_up = df_kama["kama1"] >= df_kama["kama2"]
        mask_dn = ~mask_up
        # 绿色填充
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"][mask_up],
            y=df_kama["kama2"][mask_up],
            mode="lines", line=dict(width=0), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"][mask_up],
            y=df_kama["kama1"][mask_up],
            mode="lines", fill='tonexty',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(width=0), name="KAMA1≥KAMA2"
        ), row=1, col=1)
        # 红色填充
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"][mask_dn],
            y=df_kama["kama1"][mask_dn],
            mode="lines", line=dict(width=0), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_kama["datetime"][mask_dn],
            y=df_kama["kama2"][mask_dn],
            mode="lines", fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0), name="KAMA1<KAMA2"
        ), row=1, col=1)

        # (B) 行 2: 成交量柱 + 通道
        vol_df = multiVwap.vol_df.loc[df.index]

        def get_alpha(vol, low, high):
            if vol < low:
                return 1
            elif vol > high:
                return 1
            else:
                return 1

        vols  = vol_df['vol'].values
        lows  = vol_df['lower'].values
        highs = vol_df['upper'].values
        alphas = [get_alpha(v, l, h) for v, l, h in zip(vols, lows, highs)]

        opens  = df['open'].values
        closes = df['close'].values
        marker_colors = []
        for c, o, a in zip(closes, opens, alphas):
            if c > o:
                marker_colors.append(f"rgba(0,200,0,{a})")
            elif c < o:
                marker_colors.append(f"rgba(200,0,0,{a})")
            else:
                marker_colors.append(f"rgba(100,100,200,{a})")

        fig.add_trace(go.Bar(
            x=df["datetime"],
            y=vol_df["vol_scaled"],
            marker_color=marker_colors,
            name="Scaled Volume"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["sma_scaled"],
            mode="lines", line=dict(color="gray", width=1),
            name="Volume MA"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["lower_scaled"],
            mode="lines", line=dict(width=0), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["upper_scaled"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(173,216,230,0.2)", name="Volume Band"
        ), row=2, col=1)

        # (C) 行 3: 动能指标
        mom_df = multiVwap.momentum_df.reindex(df.index)
        fig.add_trace(go.Bar(
            x=df["datetime"], y=mom_df["hl"],
            marker_color=mom_df["hlc"], name="Hist", showlegend=False
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amom"],
            mode="lines", line=dict(color="red", width=1), name="AMOM"
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amoms"],
            mode="lines", line=dict(color="green", width=1), name="Signal"
        ), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=3, col=1)

        # ========== 布局 ==========
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=900,
            margin={"t":40, "b":60, "l":20, "r":20},
            legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top")
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Vol",   row=2, col=1)
        fig.update_yaxes(title_text="Mom",   row=3, col=1)

        return fig, "数据读取正常，自动刷新"
    except Exception as e:
        return go.Figure(), f"渲染错误: {e}"

# def execute_close_position(): ...

# @app.callback(...) close handlers...

if __name__ == '__main__':
    app.run(debug=True, port=8050 if use30x else 8051)
