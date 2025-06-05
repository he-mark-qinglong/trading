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
from yyyyy2_okx_5m import trade_coin

from db_client import SQLiteWALClient



basic_time_interval = 5
use4x = True
symbol = "ETH-USDT-SWAP"


DB_PATH = f'{symbol}.db'
client = SQLiteWALClient(db_path=DB_PATH, table="ohlcv_4x" if use4x else "ohlcv")

trade_client = None

window_tau_l = int(12) 
window_tau_h = window_tau_l * 5
window_tau_s = window_tau_h * 5
multiVwap = LHFrameStd.MultiTFvp_poc(window_LFrame=window_tau_l, window_HFrame=window_tau_h, window_SFrame=window_tau_s)

app = Dash(__name__)
app.layout = html.Div([
    html.H2(f"OKX {4*basic_time_interval if use4x else basic_time_interval}s K-line OHLCV (Auto-refresh)"),
    dcc.ConfirmDialogProvider(
        children=html.Button("一键平仓", id="btn-close", n_clicks=0),
        id="confirm-close",
        message="⚠️ 确认要全部平仓？此操作不可撤销！"
    ),
    html.Div(id="close-status", style={"marginTop": "5px", "color": "green"}),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(id='interval', interval=(4 * basic_time_interval if use4x else basic_time_interval)*1000, n_intervals=0),
    html.Div(id="status-msg", style={"color": "red", "marginTop": 10})
])

# 颜色映射 & 要画的属性列表（包含 SFrame 和 HFrame 的所有线）
colors = {
    'LFrame_vp_poc_series':     'yellow',
    'SFrame_vp_poc':            'purple',

    'SFrame_vwap_up_poc':          'red',
    # 'SFrame_vwap_up_getin':    'orange',
    # 'SFrame_vwap_up_sl':       'firebrick',
    'SFrame_vwap_down_poc':        'blue',
    # 'SFrame_vwap_down_getin':  'deepskyblue',
    # 'SFrame_vwap_down_sl':     'seagreen',

    # 'HFrame_vwap_up_poc':          'magenta',
    'HFrame_vwap_up_getin':    'deeppink',
    'HFrame_vwap_up_sl':       'orangered',
    # 'HFrame_vwap_down_poc':        'teal',
    'HFrame_vwap_down_getin':  'turquoise',
    'HFrame_vwap_down_sl':     'darkslategray',
}
vars_to_plot = list(colors.keys())

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    try:
        # 1. 从 SQLite 读最新 2000 条
        try:
            # 先拿最新 2000 条（倒序）
            df = client.read_df(limit=1800, order_by="ts DESC")
            if df.empty:
                return go.Figure(), "暂无数据"

            # 再正序排列
            df = df.sort_values("ts", ascending=True)
        except Exception as e:
            return go.Figure(), f"读取数据库错误：{e}"
        if df.empty:
            return go.Figure(), "暂无数据"

        # 2. 检查必须列
        required = {"ts","open","high","low","close","vol"}
        if not required.issubset(df.columns):
            miss = required - set(df.columns)
            return go.Figure(), f"CSV 缺失列: {miss}"

        # 3. 转换时间
        df["ts"] = df["ts"].astype(int)
        df = df.drop_duplicates("ts").sort_values("ts")
        df["datetime"] = pd.to_datetime(df["ts"], unit="s")

        # 4. 计算所有 vp_poc / VWAP / STD 系列
        before_cal = time.time()
        
        multiVwap.calculate_SFrame_vp_poc_and_std(df)
        after_calc = time.time()
        print(f'{"4x" if use4x else "1x"}time consumed:{after_calc - before_cal}')

        # 5. 找到第一个非 NaN 的 HFrame 下轨止损线索引，同步截断
        start = multiVwap.HFrame_vwap_down_sl.first_valid_index()
        if start is not None:
            df = df.loc[start:].copy()
            for var in vars_to_plot:
                s = getattr(multiVwap, var, None)
                if isinstance(s, pd.Series):
                    setattr(multiVwap, var, s.loc[start:])
        else:
            df = df.iloc[0:0]

        # 6. 构建子图
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1, row_heights=[0.7, 0.3],
            subplot_titles=("K-line + vp_poc/VWAP", "Volume")
        )

        # 7. 添加 K 线
        fig.add_trace(go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name=f"{4*basic_time_interval if use4x else basic_time_interval}s-K"
        ), row=1, col=1)

        # 8. 添加所有 vp_poc/VWAP 系列
        for var in vars_to_plot:
            series = getattr(multiVwap, var, None)
            if not isinstance(series, pd.Series) or series.isna().all():
                continue
            fig.add_trace(go.Scatter(
                x=df["datetime"],
                y=series.values,
                mode="lines",
                name=var,
                line=dict(
                    color=colors[var],
                    width=1 if "HFrame" in var else 2,
                    dash="dot" if ("HFrame" in var and not "_poc" in var )else "solid" if '_poc' in var else "dash"
                )
            ), row=1, col=1)

        # 9. 添加成交量
        df["vol_color"] = np.where(
            df["close"] > df["open"],  "rgba(0,200,0,0.6)",
            np.where(df["close"] < df["open"], "rgba(200,0,0,0.6)", "rgba(100,100,200,0.6)")
        )
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["vol"],
                marker_color=df["vol_color"],
                name="Volume"
            ),
            row=2, col=1
        )

        # … 前面已经 add_trace 画完所有线后 …

        # --------- 填充 HFrame 上轨 getin 到 sl 之间的色带 ---------
        # 1) 先画下边界（getin），线宽设为 0，不显示图例
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=multiVwap.HFrame_vwap_up_getin.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ), row=1, col=1)

        # 2) 再画上边界（sl），并填充到前一条 trace
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=multiVwap.HFrame_vwap_up_sl.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,105,180,0.2)",  # 热粉色半透明
            name="HFrame_up_band"
        ), row=1, col=1)


        # --------- 填充 HFrame 下轨 sl 到 getin 之间的色带 ---------
        # 1) 先画下边界（sl）
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=multiVwap.HFrame_vwap_down_sl.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ), row=1, col=1)

        # 2) 再画上边界（getin），并填充到前一条 trace
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=multiVwap.HFrame_vwap_down_getin.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(173,216,230,0.2)",  # 淡天蓝色半透明
            name="HFrame_down_band"
        ), row=1, col=1)

        # 10. 布局调整
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            margin={"t":40, "b":60, "l":20, "r":20},  # 底部留点空间给图例
            legend=dict(
                orientation='h',      # 水平排列
                x=0.5,                # x=0.5 居中
                xanchor='center',     # 以图宽中心为对齐点
                y=0,                  # y=0 紧贴绘图区底部
                yanchor='top'         # 把 legend 的 “上边” 对齐到 y=0
            )
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)


        return fig, "数据读取正常，自动刷新"
    except Exception as e:
        return go.Figure(), f"渲染错误：{e}"

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
def on_close(submit_n, disabled):
    if not submit_n or disabled:
        return '', False

    status = "平仓中…"
    result = execute_close_position()
    if result.get("success"):
        status = f"✔ 平仓完成，详情：{result}"
    else:
        status = f"✘ 平仓失败：{result.get('errmsg','未知错误')}"
    return status, True

if __name__ == '__main__':
    app.run(debug=True, port=8050 if use4x else 8051)