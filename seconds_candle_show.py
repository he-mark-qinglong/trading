import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

csv_path = "btc_10s_ohlcv.csv"

app = Dash(__name__)

app.layout = html.Div([
    html.H2("OKX 10s K-line OHLCV (Auto-refresh)"),
    dcc.Graph(id="kline-graph"),
    dcc.Interval(id='interval', interval=5*1000, n_intervals=0),
    html.Div(id="status-msg", style={"color": "red", "marginTop": 10})
])

@app.callback(
    Output("kline-graph", "figure"),
    Output("status-msg", "children"),
    Input("interval", "n_intervals")
)
def update_graph(n):
    if not os.path.exists(csv_path):
        return go.Figure(), "CSV文件不存在"
    try:
        df = pd.read_csv(csv_path)
        expected_cols = {"ts", "open", "high", "low", "close", "vol"}
        if not expected_cols.issubset(df.columns):
            return go.Figure(), f"CSV 缺失必需列: {expected_cols - set(df.columns)}"

        df["ts"] = df["ts"].astype(int)
        df = df.sort_values("ts").drop_duplicates("ts")
        df["datetime"] = pd.to_datetime(df["ts"], unit='s')
        if len(df) > 500:
            df = df.iloc[-500:]

        # 创建上下子图（共享 x 轴）
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("K-line", "Volume")
        )

        # K线
        fig.add_trace(go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="10s-K"
        ), row=1, col=1)

        # 成交量
        fig.add_trace(go.Bar(
            x=df["datetime"],
            y=df["vol"],
            marker_color="rgba(150,150,255,0.7)",
            name="Volume"
        ), row=2, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis_title="Time",
            yaxis_title="Price",
            height=600,
            margin={"t": 30, "b": 30, "l": 20, "r": 20}
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig, "数据读取正常，自动刷新"
    except Exception as e:
        return go.Figure(), f"渲染错误：{str(e)}"

if __name__ == '__main__':
    app.run(debug=True)