import os
import pandas as pd
import time
from datetime import datetime
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import LHFrameStd
from yyyyy2_okx_5m import trade_coin

symbol = "ETH-USDT-SWAP"
csv_path = symbol + "_4s_ohlcv.csv"


trade_client = None

app = Dash(__name__)
app.layout = html.Div([
    html.H2("OKX 4s K-line OHLCV (Auto-refresh)"),
    # 这里是一键平仓部分
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

# 颜色映射 & 要画的属性列表
colors = {
    'LFrame_vpPOC_series': 'yellow',
    'LFrame_ohlc5_series': 'green',
    'SFrame_vpPOC': 'purple',
    'HFrame_vwap_up': 'red',
    'HFrame_vwap_up_getin': 'orange',
    'HFrame_vwap_up_getout': 'chocolate',
    'HFrame_vwap_down': 'blue',
    'HFrame_vwap_down_getin': 'deepskyblue',
    'HFrame_vwap_down_getout': 'cyan',
    'HFrame_vwap_up_sl': 'red',
    'HFrame_vwap_down_sl': 'green',
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
        # 1. 读取 + 验证
        df = None
        while True:
            try:
                df = pd.read_csv(csv_path)
                break
            except Exception as e:
                print(f'error:{e}')
                time.sleep(1)

        req = {"ts","open","high","low","close","vol"}
        if not req.issubset(df.columns):
            miss = req - set(df.columns)
            return go.Figure(), f"CSV 缺失列: {miss}"

        # 2. ts→datetime，截取后500条
        df["ts"] = df["ts"].astype(int)
        df = df.sort_values("ts").drop_duplicates("ts")
        df["datetime"] = pd.to_datetime(df["ts"], unit="s")

         # 3. 计算 vpPOC/VWAP 系列（与 df 同长度同索引）
        df = df.iloc[-min(1000, len(df)):]
        import time
        before_cal = time.time()
        multiVwap = LHFrameStd.MultiTFvpPOC()
        multiVwap.calculate_SFrame_vpPOC_and_std(df)
        after_calc = time.time()
        print(f'time consumed:{after_calc - before_cal}')


        # 找到第一个非 NaN 的索引标签
        start_idx = multiVwap.HFrame_vwap_down_sl.first_valid_index()

        if start_idx is not None:
            # 用 .loc 截断 df
            df = df.loc[start_idx:].copy()
            
            # 同步截断 Dash 里要画的所有 series
            for var in vars_to_plot:  # vars_to_plot 要与 Dash callback 里一致
                series = getattr(multiVwap, var, None)
                if isinstance(series, pd.Series):
                    # 截断后赋回去
                    setattr(multiVwap, var, series.loc[start_idx:])
        else:
            # 如果全是 NaN，就清空 df，下面会绘制空图
            df = df.iloc[0:0]

        # 之后继续你的绘图逻辑，用截断后的 df 与 multiVwap.series.values 渲染

        # 4. 构建 subplot
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1, row_heights=[0.7, 0.3],
            subplot_titles=("K-line + vpPOC/VWAP", "Volume")
        )

        # 5. 主图：4s K 线
        fig.add_trace(go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="4s-K"
        ), row=1, col=1)

        # 6. 主图：遍历 vpPOC/VWAP 线
        for var in vars_to_plot:
            series = getattr(multiVwap, var, None)
            if series is None: 
                continue
            # 假设 series 是 pandas.Series，索引与 df 一致
            fig.add_trace(go.Scatter(
                x=df["datetime"],
                y=series.values,
                mode="lines",
                name=var,
                line=dict(
                    color=colors[var],
                    width=2 if "vwap" not in var else 1.5,
                    dash="dash" if ("getin" in var or "getout" in var) else "solid"
                )
            ), row=1, col=1)

        # 7. 底图：成交量
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["vol"],
            marker_color="rgba(150,150,255,0.7)",
            name="Volume"
        ), row=2, col=1)

        # 8. Layout 微调
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700, margin={"t":40,"b":30,"l":20,"r":20}
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig, "数据读取正常，自动刷新"
    except Exception as e:
        return go.Figure(), f"渲染错误：{str(e)}"



# 占位函数：点击后会触发这里，后续填你真实的 API 调用逻辑
def execute_close_position():
    try:
        if trade_client == None:
            trade_client = trade_coin(symbol, 'yyyyy2_okx', 1500)
        res = trade_client.close_all_positions()
        return {"success": True, "detail": res}
    except Exception as e:
        return {"success": False, "errmsg": str(e)}

# ----------------- 平仓回调 -----------------
@app.callback(
    Output("close-status", "children"),
    Output("btn-close", "disabled"),
    Input("confirm-close", "submit_n_clicks"),
    State("btn-close", "disabled"),
)
def on_close(submit_n, disabled):
    if not submit_n or disabled:
        # 没确认或已禁用，跳过更新
        return 'waiting', False

    # 先给前端一个“正在平仓”提示，并立即禁用按钮
    status = "平仓中…"
    # 真正调用你的平仓接口
    res = execute_close_position()
    if res.get("success"):
        status = f"✔ 平仓完成，成交价={res['filled_px']}，成交量={res['filled_sz']}"
    else:
        status = f"✘ 平仓失败：{res.get('errmsg','未知错误')}"

    return status, True

if __name__ == '__main__':
    app.run(debug=True)