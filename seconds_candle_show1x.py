import os
import time

import pandas as pd
import numpy as np

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import indicators.LHFrameStd as LHFrameStd
from yyyyy2_okx_5m import trade_coin
from db_client import SQLiteWALClient

# ========== 参数配置 ==========
BASIC_INTERVAL = 5
use1x = True
symbol = "ETH-USDT-SWAP"
DB_PATH = f"{symbol}.db"
DEBUG = False

# Anchored Momentum 参数
MOM_L   = 10   # 快线周期
MOM_SL  = 8    # 慢线（信号）周期
MOM_SM  = False  # 是否对 POC-VWAP 做 EMA 平滑
MOM_SMP = 7    # 平滑周期
MOM_SH  = True # 显示柱状图
MOM_EB  = True # 启用 bar colors

# ========== 数据客户端 & multiVwap 实例 ==========
client = SQLiteWALClient(db_path=DB_PATH,
                         table="combined_1x" if use1x else "ohlcv")
trade_client = None

windowConfig = LHFrameStd.WindowConfig()
multiVwap = LHFrameStd.MultiTFVWAP(
    window_LFrame=windowConfig.window_tau_l,
    window_HFrame=windowConfig.window_tau_h,
    window_SFrame=windowConfig.window_tau_s
)

LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 39)
LIMIT_K_N = 500 + LIMIT_K_N_APPEND  + 5000


# ========== 辅助函数 ==========
def read_and_sort_df(is_append=True):
    limit = LIMIT_K_N_APPEND if is_append else LIMIT_K_N
    df = client.read_df(limit=limit, order_by="ts DESC")
    print(df.head)
    required = {"ts","open","high","low","close","vol"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列：{required - set(df.columns)}")
    df["ts"] = df["ts"].astype(int)
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("ts", drop=True).sort_index()
    return df
# 颜色映射 & 要画的属性列表（包含 SFrame 和 HFrame 的所有线）
colors = {
    'LFrame_vwap_poc':     'firebrick',
    'SFrame_vwap_poc':            'purple',

    'SFrame_vwap_up_poc':          'blue',
    # 'SFrame_vwap_up_getin':    'yellow',
    # 'SFrame_vwap_up_sl':       'white',
    'SFrame_vwap_down_poc':        'blue',
    # 'SFrame_vwap_down_getin':  'deepskyblue',
    # 'SFrame_vwap_down_sl':     'seagreen',

    'HFrame_vwap_up_poc':          'magenta',
    'HFrame_vwap_up_getin':        'deeppink',
    'HFrame_vwap_up_sl':       'orangered',
    'HFrame_vwap_up_sl2':       'orangered',
    'HFrame_vwap_down_poc':        'magenta',  #'teal',
    'HFrame_vwap_down_getin':  'turquoise',
    'HFrame_vwap_down_sl':     'darkslategray',
    'HFrame_vwap_down_sl2':     'darkslategray',
}
vars_to_plot = list(colors.keys())

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    try:
        # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
        df = read_and_sort_df(is_append=False)
        multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)

        # 截断到第一个有效 SFrame_vwap_poc
        start = multiVwap.SFrame_vwap_poc.first_valid_index()
        if start is None:
            return go.Figure(), "暂无有效数据"
        df = df.loc[start:].copy()
        for var in vars(multiVwap):
            ser = getattr(multiVwap, var)
            if isinstance(ser, pd.Series):
                setattr(multiVwap, var, ser.loc[start:])

        # --- 2. 子图：3 行 ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, shared_yaxes=False,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("K-line + vp_poc/VWAP", "Volume", "Anchored Momentum")
        )

        # (A) 行 1: K 线 + vp/VWAP 系列
        fig.add_trace(go.Candlestick(
            x=df["datetime"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="K-line"
        ), row=1, col=1)
        # 所有 vp/VWAP/STD 系列
        for name, color in {
            **{k:'firebrick' for k in ["LFrame_vwap_poc"]},
            **{k:'purple'    for k in ["SFrame_vwap_poc"]},
            **{k:'magenta'    for k in ["HFrame_vwap_poc"]},
            **{k:'orangered'   for k in ["HFrame_vwap_up_poc","HFrame_vwap_down_poc"]},
            **{k:'deeppink'   for k in ["HFrame_vwap_up_getin","HFrame_vwap_down_getin"]},
            
            **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
            **{k:'khaki'   for k in ["SFrame_vwap_up_sl","SFrame_vwap_down_sl"]},
            **{k:'darkslategray'   for k in ["SFrame_vwap_up_sl2","SFrame_vwap_down_sl2"]},
            
        }.items():
            series = getattr(multiVwap, name, None)
            if isinstance(series, pd.Series):
                fig.add_trace(go.Scatter(
                    x=series.index.map(lambda ts: pd.to_datetime(ts,unit="s")),
                    y=series.values,
                    mode="lines", name=name,
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)

        # (B) 行 2: 成交量柱 + 通道
        # ------------------------ 替换这一段 ------------------------
        # 9. 添加成交量（使用 vol_df 渲染）
        vol_df = multiVwap.vol_df.loc[df.index]  # 对齐索引

        # 9.1 透明度函数：Pine 里是 0–255，这里归一到 0–1
        def get_alpha(vol, low, high):
            if vol < low:
                return 1# 180/255    # alpha=80
            elif vol > high:
                return 1#30/255    # alpha=30
            else:
                return 1#100/255    # alpha=50

        # 9.2 逐点计算 alpha
        vols  = vol_df['vol'].values
        lows  = vol_df['lower'].values
        highs = vol_df['upper'].values
        alphas = [get_alpha(v, l, h) for v, l, h in zip(vols, lows, highs)]

        # 9.3 根据当根 K 线涨跌生成 rgba 颜色串
        opens  = df['open'].values
        closes = df['close'].values
        marker_colors = []
        for c, o, a in zip(closes, opens, alphas):
            if c > o:
                marker_colors.append(f"rgba(0,200,0,{a})")    # 涨 -> 绿
            elif c < o:
                marker_colors.append(f"rgba(200,0,0,{a})")    # 跌 -> 红
            else:
                marker_colors.append(f"rgba(100,100,200,{a})")# 平 -> 蓝灰

        # 9.4 主柱：scaled volume + 色彩透明度
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=vol_df["vol_scaled"],
                marker_color=marker_colors,
                name="Scaled Volume"
            ),
            row=2, col=1
        )

        # 9.5 SMA 线
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=vol_df["sma_scaled"],
                mode="lines",
                line=dict(color="gray", width=1),
                name="Volume MA"
            ),
            row=2, col=1
        )

        # 9.6 通道带填充：先画下轨（invisible）
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=vol_df["lower_scaled"],
                mode="lines",
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=1
        )
        # 再画上轨并填充到上一条 trace
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=vol_df["upper_scaled"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(173,216,230,0.2)",
                name="Volume Band"
            ),
            row=2, col=1
        )
        # ----------------------------------------------------------


        # (C) 行 3: 动能指标
        mom_df = multiVwap.momentum_df.reindex(df.index)
        fig.add_trace(go.Bar(
            x=df["datetime"], y=mom_df["hl"],
            marker_color=mom_df["hlc"], name="Hist", showlegend=False
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amom"],
            mode="lines", line=dict(color="red",width=1), name="AMOM"
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amoms"],
            mode="lines", line=dict(color="green",width=1), name="Signal"
        ), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="gray",dash="dash"), row=3, col=1)

        # ========== 布局 ==========
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=900,
            margin={"t":40,"b":60,"l":20,"r":20},
            legend=dict(orientation="h",x=0.5,xanchor="center",y=0,yanchor="top")
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Vol",   row=2, col=1)
        fig.update_yaxes(title_text="Mom",   row=3, col=1)

        return fig, "数据读取正常，自动刷新"
    except Exception as e:
        return go.Figure(), f"渲染错误: {e}"
    
def execute_close_position():
    global trade_client
    try:
        if trade_client is None:
            trade_client = trade_coin(symbol, 'yyyyy2_okx', 1500)
        res = trade_client.close_all_positions()
        return {"success": True, **res}
    except Exception as e:
        return {"success": False, "errmsg": str(e)}

@app.callback(
    Output("close-status", "children"),
    Output("btn-close", "disabled"),
    Input("confirm-close", "submit_n_clicks"),
    State("btn-close", "disabled"),
)

def on_close(n, disabled):
    if not n or disabled:
        return "", False
    try:
        global trade_client
        if trade_client is None:
            trade_client = trade_coin(symbol,'yyyyy2_okx',1500)
        res = trade_client.close_all_positions()
        return f"✔ 平仓完成: {res}", True
    except Exception as e:
        return f"✘ 平仓失败: {e}", True

if __name__ == "__main__":
    # 预热计算一次
    df0 = read_and_sort_df(is_append=False)
    multiVwap.calculate_SFrame_vwap_poc_and_std(df0, DEBUG)
    app.run(debug=True, port=8051 if use1x else 8050)