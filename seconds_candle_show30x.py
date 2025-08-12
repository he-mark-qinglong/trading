
import os
import pandas as pd
import numpy as np
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from indicators import LHFrameStd
# from yyyyy2_okx_5m import trade_coin

from db_client import SQLiteWALClient
# from db_read import read_and_sort_df
from history_kline import read_and_sort_df
from db_read import resample_to

from indicators import  anchored_momentum_via_kama, compute_dynamic_kama

BASIC_INTERVAL = 5
use30x = True
symbol = "ETH-USDT-SWAP"

DEBUG = False
# DEBUG = True

DB_PATH = f'{symbol}.db'
client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")
trade_client = None

windowConfig = LHFrameStd.WindowConfig()

LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 310)
LIMIT_K_N = 1700 + LIMIT_K_N_APPEND 
TREND_LENGTH = 2160000
# TREND_LENGTH = 2000
LIMIT_K_N += TREND_LENGTH

app = Dash(__name__)

def make_kline_section(idx):
    """生成一份 K 线区域，所有 id 带上 idx 后缀以免冲突"""
    suffix = f"-{idx}"
    return html.Div([
        html.H2(f"OKX {idx * 60*BASIC_INTERVAL if use30x else BASIC_INTERVAL}s K-line OHLCV (Auto-refresh)"),
        dcc.ConfirmDialogProvider(
            children=html.Button("一键平仓", id="btn-close" + suffix, n_clicks=0),
            id="confirm-close" + suffix,
            message="⚠️ 确认要全部平仓？此操作不可撤销！"
        ),
        html.Div(id="close-status" + suffix, style={"marginTop": "5px", "color": "green"}),
        dcc.Graph(id="kline-graph" + suffix),
        dcc.Interval(
            id='interval' + suffix,
            interval=(idx * 60 * BASIC_INTERVAL if use30x else BASIC_INTERVAL)*1000,
            n_intervals=0
        ),
        html.Div(id="status-msg" + suffix, style={"color": "red", "marginTop": 10})
    ], style={"marginBottom": "50px"})  # 下方留白，纵向分隔

# 主 layout：纵向两份
app.layout = html.Div([
    make_kline_section(1),
    make_kline_section(2),
    # make_kline_section(3),
])


# --- 新增的封装函数，放在文件顶部或合适位置 ---
def add_trade_signals_to_fig(fig, trade_df, row=1, col=1):
    if trade_df is None or trade_df.empty:
        return

    # 多头开仓
    mask_open_long = trade_df['action'].str.contains('open_long', na=False)
    fig.add_trace(go.Scatter(
        x=trade_df.index[mask_open_long],
        y=trade_df.loc[mask_open_long, 'price'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='green'),
        name='Open Long'
    ), row=row, col=col)

    # 多头平仓
    mask_close_long = trade_df['action'].str.contains('close_long|stop_loss_long|decay_long', na=False)
    fig.add_trace(go.Scatter(
        x=trade_df.index[mask_close_long],
        y=trade_df.loc[mask_close_long, 'price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=15, color='red'),
        name='Close Long'
    ), row=row, col=col)

    # 空头开仓
    mask_open_short = trade_df['action'].str.contains('open_short', na=False)
    fig.add_trace(go.Scatter(
        x=trade_df.index[mask_open_short],
        y=trade_df.loc[mask_open_short, 'price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=15, color='blue'),
        name='Open Short'
    ), row=row, col=col)

    # 空头平仓
    mask_close_short = trade_df['action'].str.contains('close_short|stop_loss_short|decay_short', na=False)
    fig.add_trace(go.Scatter(
        x=trade_df.index[mask_close_short],
        y=trade_df.loc[mask_close_short, 'price'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='orange'),
        name='Close Short'
    ), row=row, col=col)

@app.callback(
    Output("kline-graph-1", "figure"),
    Output("status-msg-1", "children"),
    Input("interval-1", "n_intervals")
)
def update_graph_1(n):
    try:
        # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
        before = time.time()
        df = read_and_sort_df(client, LIMIT_K_N)
        df = resample_to(df, '5min')
        print('data time range', df.index[0], df.index[-1])

        # df = resample_to_15m(df)
        print("read df and convert takes time:", time.time() - before)

        before = time.time()
        multiVwap = LHFrameStd.MultiTFVWAP(
            window_LFrame=windowConfig.window_tau_l, 
            window_HFrame=windowConfig.window_tau_h,
            window_SFrame=windowConfig.window_tau_s
        )
        # multiVwap.calculate_SFrame_vwap_poc_and_std(df.iloc[-(LIMIT_K_N - TREND_LENGTH):], DEBUG)
        multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)
        print("calc vwap-1 takes time:", time.time() - before)

        # 截断到第一个有效 SFrame_vwap_poc
        start = multiVwap.SFrame_vwap_poc.first_valid_index()
        if start is None:
            return go.Figure(), "暂无有效数据"
        # df = df.loc[start:].copy()
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
            len_er=200,
            fast=15,
            second2first_times=2.0,
            slow=1800,
            intervalP=0.01,
            minLen=10,
            maxLen=60,
            volLen=30
        )
        before = time.time()
        df_kama = compute_dynamic_kama(df, **kama_params)
        # df_kama.reindex()
        
        print("compute_dynamic_kama-1 takes time:", time.time() - before)

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
            # **{k:'deeppink'    for k in ["SFrame_vwap_up_getin","SFrame_vwap_down_getin"]},
            # **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
            # **{k:'blue'        for k in ["SFrame_vwap_up_sl"]},
            # **{k:'blue'        for k in ["SFrame_vwap_down_sl"]},
            **{k:'black'        for k in ["SFrame_center"]},
            **{k:'darkslategray' for k in ["SFrame_vwap_up_sl2"]},
            **{k:'darkslategray' for k in ["SFrame_vwap_down_sl2"]},
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


        # 这里插入交易信号叠加
        from trade_log_manager import TradeLogManager
        manager = TradeLogManager(base_path='./my_trade_data')
        trade_df = manager.load_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe='5min')
        add_trade_signals_to_fig(fig, trade_df, row=1, col=1)

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
        # mom_df = multiVwap.momentum_df.reindex(df.index)
        mom_df = anchored_momentum_via_kama(kama1=df_kama['kama1'], kama2=df_kama['kama2'], signal_period=140)
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
    

@app.callback(
    Output("kline-graph-2", "figure"),
    Output("status-msg-2", "children"),
    Input("interval-2", "n_intervals")
)
def update_graph_2(n):
    try:
        # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
        before = time.time()
        # from db_read import read_and_sort_df
        df = read_and_sort_df(client, LIMIT_K_N)

        print('data time range', df.index[0], df.index[-1] )
        df = resample_to(df.copy(deep=True), '15min')
        print("read df and convert takes time:", time.time() - before)

        before = time.time()
        multiVwap = LHFrameStd.MultiTFVWAP(
            window_LFrame=windowConfig.window_tau_l, 
            window_HFrame=windowConfig.window_tau_h,
            window_SFrame=windowConfig.window_tau_s
        )
        # multiVwap.calculate_SFrame_vwap_poc_and_std(df.iloc[-(LIMIT_K_N - TREND_LENGTH):], DEBUG)
        multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)
        print("calc vwap-2 takes time:", time.time() - before)

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
            len_er=200,
            fast=15,
            second2first_times=2.0,
            slow=1800,
            intervalP=0.01,
            minLen=10,
            maxLen=60,
            volLen=30
        )
        before = time.time()

        
        df_kama = compute_dynamic_kama(df, **kama_params)
        
        print("compute_dynamic_kama-2 takes time:", time.time() - before)

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
            # **{k:'deeppink'    for k in ["SFrame_vwap_up_getin","SFrame_vwap_down_getin"]},
            # **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
            # **{k:'blue'        for k in ["SFrame_vwap_up_sl"]},
            # **{k:'blue'        for k in ["SFrame_vwap_down_sl"]},
            **{k:'black'        for k in ["SFrame_center"]},
            **{k:'darkslategray' for k in ["SFrame_vwap_up_sl2"]},
            **{k:'darkslategray' for k in ["SFrame_vwap_down_sl2"]},
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
        # mom_df = multiVwap.momentum_df.reindex(df.index)
        mom_df = anchored_momentum_via_kama(kama1=df_kama['kama1'], kama2=df_kama['kama2'], signal_period=140)
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



# @app.callback(
#     Output("kline-graph-3", "figure"),
#     Output("status-msg-3", "children"),
#     Input("interval-3", "n_intervals")
# )
# def update_graph_3(n):
#     try:
#         # --- 1. 读数据 & 计算 vp_poc/VWAP/STD ---
#         before = time.time()
#         # from db_read import read_and_sort_df
#         df = read_and_sort_df(client, LIMIT_K_N)

#         print(df.head)

#         df = resample_to(df.copy(deep=True), '60min')
#         print("read df and convert takes time:", time.time() - before)

#         before = time.time()
#         multiVwap = LHFrameStd.MultiTFVWAP(
#             window_LFrame=windowConfig.window_tau_l, 
#             window_HFrame=windowConfig.window_tau_h,
#             window_SFrame=windowConfig.window_tau_s
#         )
#         # multiVwap.calculate_SFrame_vwap_poc_and_std(df.iloc[-(LIMIT_K_N - TREND_LENGTH):], DEBUG)
#         multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)
#         print("calc vwap-3 takes time:", time.time() - before)

#         # 截断到第一个有效 SFrame_vwap_poc
#         start = multiVwap.SFrame_vwap_poc.first_valid_index()
#         if start is None:
#             return go.Figure(), "暂无有效数据"
#         df = df.loc[start:].copy()
#         # 对 multiVwap 系列做同样截断
#         for var in vars(multiVwap):
#             ser = getattr(multiVwap, var)
#             if isinstance(ser, pd.Series):
#                 setattr(multiVwap, var, ser.loc[start:])

#         # 确保 datetime 列存在
#         if "datetime" not in df:
#             df["datetime"] = pd.to_datetime(df.index, unit="s")

#         # --- 1.5 计算 Dynamic KAMA ---
#         kama_params = dict(
#             src_col="close",
#             len_er=200,
#             fast=15,
#             second2first_times=2.0,
#             slow=1800,
#             intervalP=0.01,
#             minLen=10,
#             maxLen=60,
#             volLen=30
#         )
#         before = time.time()

        
#         df_kama = compute_dynamic_kama(df, **kama_params)
        
        
#         print("compute_dynamic_kama-3 takes time:", time.time() - before)

#         # --- 2. 子图：3 行 ---
#         fig = make_subplots(
#             rows=3, cols=1,
#             shared_xaxes=True, shared_yaxes=False,
#             vertical_spacing=0.08,
#             row_heights=[0.6, 0.2, 0.2],
#             subplot_titles=("K-line + vp_poc/VWAP + KAMA", "Volume", "Anchored Momentum")
#         )

#         # (A) 行 1: K 线 + vp/VWAP 系列
#         fig.add_trace(go.Candlestick(
#             x=df["datetime"], open=df["open"], high=df["high"],
#             low=df["low"], close=df["close"],
#             name="K-line"
#         ), row=1, col=1)

#         for name, color in {
#             # **{k:'purple'      for k in ["SFrame_vwap_poc"]},
#             # **{k:'deeppink'    for k in ["SFrame_vwap_up_getin","SFrame_vwap_down_getin"]},
#             **{k:'turquoise'   for k in ["SFrame_vwap_up_poc","SFrame_vwap_down_poc"]},
#             **{k:'blue'        for k in ["SFrame_vwap_up_sl"]},
#             **{k:'blue'        for k in ["SFrame_vwap_down_sl"]},
#             **{k:'black'        for k in ["SFrame_center"]},
#             **{k:'darkslategray' for k in ["SFrame_vwap_up_sl2"]},
#             **{k:'darkslategray' for k in ["SFrame_vwap_down_sl2"]},
#         }.items():
#             series = getattr(multiVwap, name, None)
#             if isinstance(series, pd.Series):
#                 fig.add_trace(go.Scatter(
#                     x=series.index.map(lambda ts: pd.to_datetime(ts,unit="s")),
#                     y=series.values,
#                     mode="lines", name=name,
#                     line=dict(color=color, width=1.5)
#                 ), row=1, col=1)

#         # (A.5) 行1: 绘制 KAMA1/KAMA2 及区域填充
#         # 线条
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"], y=df_kama["kama1"],
#             mode="lines", name="KAMA1", line=dict(color="green", width=1)
#         ), row=1, col=1)
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"], y=df_kama["kama2"],
#             mode="lines", name="KAMA2", line=dict(color="blue", width=2)
#         ), row=1, col=1)
#         # 填充区域
#         mask_up = df_kama["kama1"] >= df_kama["kama2"]
#         mask_dn = ~mask_up
#         # 绿色填充
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"][mask_up],
#             y=df_kama["kama2"][mask_up],
#             mode="lines", line=dict(width=0), showlegend=False
#         ), row=1, col=1)
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"][mask_up],
#             y=df_kama["kama1"][mask_up],
#             mode="lines", fill='tonexty',
#             fillcolor='rgba(0,255,0,0.2)',
#             line=dict(width=0), name="KAMA1≥KAMA2"
#         ), row=1, col=1)
#         # 红色填充
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"][mask_dn],
#             y=df_kama["kama1"][mask_dn],
#             mode="lines", line=dict(width=0), showlegend=False
#         ), row=1, col=1)
#         fig.add_trace(go.Scatter(
#             x=df_kama["datetime"][mask_dn],
#             y=df_kama["kama2"][mask_dn],
#             mode="lines", fill='tonexty',
#             fillcolor='rgba(255,0,0,0.2)',
#             line=dict(width=0), name="KAMA1<KAMA2"
#         ), row=1, col=1)

#         # (B) 行 2: 成交量柱 + 通道
#         vol_df = multiVwap.vol_df.loc[df.index]

#         def get_alpha(vol, low, high):
#             if vol < low:
#                 return 1
#             elif vol > high:
#                 return 1
#             else:
#                 return 1

#         vols  = vol_df['vol'].values
#         lows  = vol_df['lower'].values
#         highs = vol_df['upper'].values
#         alphas = [get_alpha(v, l, h) for v, l, h in zip(vols, lows, highs)]

#         opens  = df['open'].values
#         closes = df['close'].values
#         marker_colors = []
#         for c, o, a in zip(closes, opens, alphas):
#             if c > o:
#                 marker_colors.append(f"rgba(0,200,0,{a})")
#             elif c < o:
#                 marker_colors.append(f"rgba(200,0,0,{a})")
#             else:
#                 marker_colors.append(f"rgba(100,100,200,{a})")

#         fig.add_trace(go.Bar(
#             x=df["datetime"],
#             y=vol_df["vol_scaled"],
#             marker_color=marker_colors,
#             name="Scaled Volume"
#         ), row=2, col=1)
#         fig.add_trace(go.Scatter(
#             x=df["datetime"], y=vol_df["sma_scaled"],
#             mode="lines", line=dict(color="gray", width=1),
#             name="Volume MA"
#         ), row=2, col=1)
#         fig.add_trace(go.Scatter(
#             x=df["datetime"], y=vol_df["lower_scaled"],
#             mode="lines", line=dict(width=0), showlegend=False
#         ), row=2, col=1)
#         fig.add_trace(go.Scatter(
#             x=df["datetime"], y=vol_df["upper_scaled"],
#             mode="lines", line=dict(width=0), fill="tonexty",
#             fillcolor="rgba(173,216,230,0.2)", name="Volume Band"
#         ), row=2, col=1)

#         # (C) 行 3: 动能指标
#         # mom_df = multiVwap.momentum_df.reindex(df.index)
#         mom_df = anchored_momentum_via_kama(kama1=df_kama['kama1'], kama2=df_kama['kama2'], signal_period=140)
#         fig.add_trace(go.Bar(
#             x=df["datetime"], y=mom_df["hl"],
#             marker_color=mom_df["hlc"], name="Hist", showlegend=False
#         ), row=3, col=1)
#         fig.add_trace(go.Scatter(
#             x=df["datetime"], y=mom_df["amom"],
#             mode="lines", line=dict(color="red", width=1), name="AMOM"
#         ), row=3, col=1)
#         fig.add_trace(go.Scatter(
#             x=df["datetime"], y=mom_df["amoms"],
#             mode="lines", line=dict(color="green", width=1), name="Signal"
#         ), row=3, col=1)
#         fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=3, col=1)

#         # ========== 布局 ==========
#         fig.update_layout(
#             xaxis_rangeslider_visible=False,
#             height=900,
#             margin={"t":40, "b":60, "l":20, "r":20},
#             legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top")
#         )
#         fig.update_yaxes(title_text="Price", row=1, col=1)
#         fig.update_yaxes(title_text="Vol",   row=2, col=1)
#         fig.update_yaxes(title_text="Mom",   row=3, col=1)

#         return fig, "数据读取正常，自动刷新"
#     except Exception as e:
#         return go.Figure(), f"渲染错误: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)
