import os
import pandas as pd
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import LHFrameStd
from yyyyy2_okx_5m import trade_coin

symbol = "ETH-USDT-SWAP"
csv_path = f"{symbol}_4s_ohlcv.csv"

trade_client = None

app = Dash(__name__)
app.layout = html.Div([
    html.H2("OKX 4s K-line OHLCV (Auto-refresh)"),
    dcc.ConfirmDialogProvider(
        children=html.Button("一键平仓", id="btn-close", n_clicks=0),
        id="confirm-close",
        message="⚠️ 确认要全部平仓？此操作不可撤销！"
    ),
    html.Div(id="close-status", style={"marginTop": "5px", "color": "green"}),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(id='interval', interval=5*1000, n_intervals=0),
    html.Div(id="status-msg", style={"color": "red", "marginTop": 10})
])

# 颜色映射 & 要画的属性列表（包含 SFrame 和 HFrame 的所有线）
colors = {
    'LFrame_vpPOC_series':     'yellow',
    'LFrame_ohlc5_series':     'green',
    'SFrame_vpPOC':            'purple',
    'SFrame_vwap_up':          'red',
    'SFrame_vwap_up_getin':    'orange',
    'SFrame_vwap_up_getout':   'chocolate',
    'SFrame_vwap_up_sl':       'firebrick',
    'SFrame_vwap_down':        'blue',
    'SFrame_vwap_down_getin':  'deepskyblue',
    'SFrame_vwap_down_getout': 'cyan',
    'SFrame_vwap_down_sl':     'seagreen',
    'HFrame_vwap_up':          'magenta',
    'HFrame_vwap_up_getin':    'deeppink',
    'HFrame_vwap_up_getout':   'hotpink',
    'HFrame_vwap_up_sl':       'orangered',
    'HFrame_vwap_down':        'teal',
    'HFrame_vwap_down_getin':  'turquoise',
    'HFrame_vwap_down_getout': 'lightseagreen',
    'HFrame_vwap_down_sl':     'darkslategray',
}
vars_to_plot = list(colors.keys())

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    if not os.path.exists(csv_path):
        return go.Figure(), "CSV文件不存在"

    try:
        # 1. 读取 CSV（带锁重试）
        while True:
            try:
                df = pd.read_csv(csv_path)
                break
            except Exception:
                time.sleep(0.1)

        # 2. 检查必须列
        required = {"ts","open","high","low","close","vol"}
        if not required.issubset(df.columns):
            miss = required - set(df.columns)
            return go.Figure(), f"CSV 缺失列: {miss}"

        # 3. 转换时间，截取最后 2000 行
        df["ts"] = df["ts"].astype(int)
        df = df.drop_duplicates("ts").sort_values("ts").iloc[-2000:].copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="s")

        # 4. 计算所有 vpPOC / VWAP / STD 系列
        multiVwap = LHFrameStd.MultiTFvpPOC()
        multiVwap.calculate_SFrame_vpPOC_and_std(df)

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
            subplot_titles=("K-line + vpPOC/VWAP", "Volume")
        )

        # 7. 添加 K 线
        fig.add_trace(go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="4s-K"
        ), row=1, col=1)

        # 8. 添加所有 vpPOC/VWAP 系列
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
                    width=1.5,
                    dash="dash" if "getin" in var or "getout" in var else "solid"
                )
            ), row=1, col=1)

        # 9. 添加成交量
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["vol"],
            marker_color="rgba(150,150,255,0.7)",
            name="Volume"
        ), row=2, col=1)

        # 10. 布局调整
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            margin={"t":40,"b":30,"l":20,"r":20}
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
    app.run(debug=True)