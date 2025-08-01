
import os
import pandas as pd
import numpy as np
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import LHFrameStd
from db_client import SQLiteWALClient
from db_read import read_and_sort_df, resample_to_7_5m
from dynamic_kama import compute_dynamic_kama

# -------- 用户可调参数 --------
BASIC_INTERVAL = 5
symbol = "ETH-USDT-SWAP"
DB_PATH = f'{symbol}.db'
LIMIT_K_N = 500 + max(LHFrameStd.WindowConfig().window_tau_s, 310) + 60000

# 初始化数据库 client
client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")

# 两个多周期 VWAP/POC 处理器 (2.5m 和 10m)
windowConfig = LHFrameStd.WindowConfig()
multiVwap1 = LHFrameStd.MultiTFvp_poc(windowConfig.window_tau_l,
                                     windowConfig.window_tau_h,
                                     windowConfig.window_tau_s)
multiVwap2 = LHFrameStd.MultiTFvp_poc(windowConfig.window_tau_l,
                                     windowConfig.window_tau_h,
                                     windowConfig.window_tau_s)

# Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H2(f"{symbol} 多周期 K 线对比（左 2.5m，右 10m）"),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(id='interval', interval=30*BASIC_INTERVAL*1000, n_intervals=0)
])

@app.callback(
    Output("kline-graph", "figure"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    try:
        # --- 1) 读取两次数据（后面你改成 10m 数据源） ---
        df1 = read_and_sort_df(client, LIMIT_K_N)    # 2.5m
        df2 = read_and_sort_df(client, LIMIT_K_N, resample=True)    # 2.5m  resample_to_7_5m(df1.copy(deep=True))    # TODO: 换成 10m 数据

        # --- 2) VWAP/POC 计算（示例复用同一逻辑） ---
        multiVwap1.calculate_SFrame_vwap_poc_and_std(df1, False)
        multiVwap2.calculate_SFrame_vwap_poc_and_std(df2, False)

        start1 = multiVwap1.SFrame_vwap_poc.first_valid_index()
        start2 = multiVwap2.SFrame_vwap_poc.first_valid_index()
        if start1 is None or start2 is None:
            return go.Figure(), "暂无有效数据"

        df1 = df1.loc[start1:].copy()
        df2 = df2.loc[start2:].copy()

        for obj, start in [(multiVwap1, start1), (multiVwap2, start2)]:
            for var in vars(obj):
                ser = getattr(obj, var)
                if isinstance(ser, pd.Series):
                    setattr(obj, var, ser.loc[start:])

        for df in (df1, df2):
            if "datetime" not in df:
                df["datetime"] = pd.to_datetime(df.index, unit="s")

        # --- 3) 计算 Dynamic KAMA ---
        kama_kwargs = dict(src_col="close", len_er=30, fast=6,
                          second2first_times=2.0, slow=120,
                          intervalP=0.01, minLen=10, maxLen=60, volLen=30)
        df_kama1 = compute_dynamic_kama(df1, **kama_kwargs)
        df_kama2 = compute_dynamic_kama(df2, **kama_kwargs)

        # --- 4) 搭建 3×2 子图 ---
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.08,
            column_widths=[0.5, 0.5],
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[
                "2.5m K 线 + VWAP/POC + KAMA",
                "10m  K 线 + VWAP/POC + KAMA",
                "2.5m 成交量",
                "10m  成交量",
                "2.5m 动能",
                "10m  动能"
            ]
        )

        def plot_column(df, df_kama, mvwap, col):
            # A) K线 + VWAP/POC + KAMA
            fig.add_trace(go.Candlestick(
                x=df["datetime"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name=f"K-line ({col})"
            ), row=1, col=col)

            # VWAP/POC 系列
            for name, color in [
                ("SFrame_vwap_poc", "purple"),
                ("HFrame_vwap_poc", "magenta"),
                ("SFrame_vwap_up_getin", "deeppink"),
                ("SFrame_vwap_down_getin", "deeppink"),
                ("SFrame_vwap_up_poc", "turquoise"),
                ("SFrame_vwap_down_poc", "turquoise")
            ]:
                ser = getattr(mvwap, name, None)
                if isinstance(ser, pd.Series):
                    fig.add_trace(go.Scatter(
                        x=ser.index.map(lambda ts: pd.to_datetime(ts, unit="s")),
                        y=ser.values, mode="lines", name=name+f"({col})",
                        line=dict(color=color, width=1.5)
                    ), row=1, col=col)

            # KAMA1/KAMA2 + 填色
            for nm, dfk, cols, width in [
                ("kama1", df_kama, "green", 1),
                ("kama2", df_kama, "blue", 2)
            ]:
                fig.add_trace(go.Scatter(
                    x=df_kama["datetime"], y=df_kama[nm],
                    mode="lines", name=f"{nm}({col})",
                    line=dict(color=cols, width=width)
                ), row=1, col=col)

            mask_up = df_kama["kama1"] >= df_kama["kama2"]
            mask_dn = ~mask_up
            # 上升填充
            fig.add_trace(go.Scatter(
                x=df_kama["datetime"][mask_up], y=df_kama["kama2"][mask_up],
                mode="lines", line=dict(width=0), showlegend=False
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=df_kama["datetime"][mask_up], y=df_kama["kama1"][mask_up],
                mode="lines", fill='tonexty',
                fillcolor='rgba(0,255,0,0.2)', line=dict(width=0),
                name=f"KAMA1≥KAMA2({col})"
            ), row=1, col=col)
            # 下降填充
            fig.add_trace(go.Scatter(
                x=df_kama["datetime"][mask_dn], y=df_kama["kama1"][mask_dn],
                mode="lines", line=dict(width=0), showlegend=False
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=df_kama["datetime"][mask_dn], y=df_kama["kama2"][mask_dn],
                mode="lines", fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)', line=dict(width=0),
                name=f"KAMA1<KAMA2({col})"
            ), row=1, col=col)

            # B) 成交量
            vol_df = mvwap.vol_df.loc[df.index]
            colors = [
                "rgba(0,200,0,1)" if c>o else "rgba(200,0,0,1)"
                for c,o in zip(df["close"], df["open"])
            ]
            fig.add_trace(go.Bar(
                x=df["datetime"], y=vol_df["vol_scaled"],
                marker_color=colors, name=f"Vol({col})"
            ), row=2, col=col)
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=vol_df["sma_scaled"],
                mode="lines", line=dict(color="gray", width=1),
                name=f"VolMA({col})"
            ), row=2, col=col)

            # C) 动能
            mom = mvwap.momentum_df.reindex(df.index)
            fig.add_trace(go.Bar(
                x=df["datetime"], y=mom["hl"],
                marker_color=mom["hlc"], showlegend=False, name=f"Hist({col})"
            ), row=3, col=col)
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=mom["amom"],
                mode="lines", line=dict(color="red", width=1),
                name=f"AMOM({col})"
            ), row=3, col=col)
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=mom["amoms"],
                mode="lines", line=dict(color="green", width=1),
                name=f"Signal({col})"
            ), row=3, col=col)
            fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=3, col=col)

        # 左侧 = 2.5m，右侧 = 10m
        plot_column(df1, df_kama1, multiVwap1, col=1)
        plot_column(df2, df_kama2, multiVwap2, col=2)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=900, width=1400,
            margin={"t":40, "b":60, "l":20, "r":20},
            legend=dict(orientation="h", x=0.5, xanchor="center", y=0)
        )
        for r, txt in zip([1,2,3], ["Price", "Vol", "Mom"]):
            fig.update_yaxes(title_text=txt, row=r, col=1)
            fig.update_yaxes(title_text=txt, row=r, col=2)

        return fig

    except Exception as e:
        return go.Figure(), f"渲染错误: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)
