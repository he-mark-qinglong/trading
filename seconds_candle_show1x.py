import os
import time

import pandas as pd
import numpy as np

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import LHFrameStd
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
                         table="combined_9x" if use1x else "ohlcv")
trade_client = None

windowConfig = LHFrameStd.WindowConfig()
multiVwap = LHFrameStd.MultiTFvp_poc(
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

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def anchored_momentum(
    df: pd.DataFrame,
    poc_series: pd.Series,
    l: int, sl: int,
    sm: bool, smp: int,
    sh: bool, eb: bool
) -> pd.DataFrame:
    """
    计算 amom, amoms, hl, hlc
    """
    out = pd.DataFrame(index=df.index)
    src = poc_series.reindex(df.index)
    if sm:
        src = ema(src, smp)
    p = 2*l + 1
    out["amom"]  = 100 * ( src / sma(poc_series, p).reindex(df.index) - 1 )
    out["amoms"] = sma(out["amom"], sl)

    # 柱状图 hl
    out["hl"] = 0.0
    if sh:
        pos = (out["amom"]>out["amoms"]) & (out["amom"]>0) & (out["amoms"]>0)
        neg = (out["amom"]<out["amoms"]) & (out["amom"]<0) & (out["amoms"]<0)
        out.loc[pos, "hl"] = np.minimum(out.loc[pos,"amom"],  out.loc[pos,"amoms"])
        out.loc[neg, "hl"] = np.maximum(out.loc[neg,"amom"],  out.loc[neg,"amoms"])
    # bar-color
    out["hlc"] = None
    fast_above = out["amom"] > out["amoms"]
    for idx, (fa, a, s) in enumerate(zip(fast_above, out["amom"], out["amoms"])):
        if fa:
            out.iat[idx, out.columns.get_loc("hlc")] = "green" if a>=0 else "orange"
        else:
            out.iat[idx, out.columns.get_loc("hlc")] = "orange" if a>=0 else "red"
    if not eb:
        out["hlc"] = None

    return out

# ========== Dash App ==========
app = Dash(__name__)
app.layout = html.Div([
    html.H2(f"OKX {BASIC_INTERVAL if not use1x else 1*BASIC_INTERVAL}s K-line OHLCV (Auto-refresh)"),
    dcc.ConfirmDialogProvider(
        children=html.Button("一键平仓", id="btn-close"),
        id="confirm-close",
        message="⚠️ 确认要全部平仓？此操作不可撤销！"
    ),
    html.Div(id="close-status", style={"marginTop":"5px","color":"green"}),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(
        id="interval",
        interval=(8*BASIC_INTERVAL if use1x else BASIC_INTERVAL)*1000,
        n_intervals=0
    ),
    html.Div(id="status-msg", style={"color":"red","marginTop":1})
])

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    try:
        # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
        df = read_and_sort_df(is_append=False)
        multiVwap.calculate_SFrame_vp_poc_and_std(df, DEBUG)

        # 截断到第一个有效 SFrame_vp_poc
        start = multiVwap.SFrame_vp_poc.first_valid_index()
        if start is None:
            return go.Figure(), "暂无有效数据"
        df = df.loc[start:].copy()
        for var in vars(multiVwap):
            ser = getattr(multiVwap, var)
            if isinstance(ser, pd.Series):
                setattr(multiVwap, var, ser.loc[start:])

        # --- 2. 子图：3 行 ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
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
        basic_colors = [
            'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet',
            'pink', 'brown', 'black', 'white', 'gray', 'cyan', 'magenta',
            'lime', 'navy', 'maroon', 'olive', 'teal', 'purple', 'gold',
            'silver', 'coral', 'salmon', 'turquoise', 'chocolate', 'khaki'
        ]
        # 所有 vp/VWAP/STD 系列
        for name, color in {
            **{k:'firebrick' for k in ["LFrame_vp_poc_series"]},
            **{k:'purple'    for k in ["SFrame_vp_poc"]},
            **{k:'magenta'   for k in ["HFrame_vwap_up_poc","HFrame_vwap_down_poc"]},
            **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
            **{k:'khaki'   for k in ["SFrame_vwap_up_sl","SFrame_vwap_down_sl"]}
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
        vol_df = multiVwap.vol_df.loc[df.index]
        def get_alpha(v, lo, hi):
            return 30/255 if v<lo else 254/255 if v>hi else 124/255
        vols = vol_df["vol"].values
        lows = vol_df["lower"].values
        highs= vol_df["upper"].values
        alphas = [get_alpha(v,l,h) for v,l,h in zip(vols,lows,highs)]
        ops = df["open"].values; cls = df["close"].values
        colors_bar = [
            f"rgba({0 if c>o else 200},{200 if c>o else 0},0,{a})"
            if c!=o else f"rgba(100,100,200,{a})"
            for c,o,a in zip(cls,ops,alphas)
        ]
        fig.add_trace(go.Bar(
            x=df["datetime"], y=vol_df["vol_scaled"],
            marker_color=colors_bar, name="Vol"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["sma_scaled"],
            mode="lines", line=dict(color="gray",width=1), name="Vol MA"
        ), row=2, col=1)
        # 通道填充
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["lower_scaled"],
            mode="lines", line=dict(width=0), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=vol_df["upper_scaled"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(173,216,230,0.4)",
            name="Vol Band"
        ), row=2, col=1)

        # (C) 行 3: 动能指标
        mom_df = multiVwap.momentum_df.reindex(df.index)
        
        print(mom_df.head)
        fig.add_trace(go.Bar(
            x=df["datetime"],
            y=mom_df["hl"],
            marker_color=mom_df["hlc"],
            name="Hist",
            showlegend=False
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amom"],
            mode="lines", line=dict(color="red", width=1),
            name="AMOM"
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df["datetime"], y=mom_df["amoms"],
            mode="lines", line=dict(color="green", width=1),
            name="Signal"
        ), row=3, col=1)

        fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=3, col=1)

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

@app.callback(
    Output("close-status","children"),
    Output("btn-close","disabled"),
    Input("confirm-close","submit_n_clicks"),
    State("btn-close","disabled")
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
    multiVwap.calculate_SFrame_vp_poc_and_std(df0, DEBUG)
    app.run(debug=True, port=8051 if use1x else 8050)