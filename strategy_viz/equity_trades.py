
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def _ensure_dt_index(df: pd.DataFrame, prefer_col: str = "datetime") -> pd.DataFrame:
    """
    Ensure DataFrame index is datetime.
    If 'prefer_col' exists, use it as the datetime index after conversion.
    """
    df = df.copy()
    if prefer_col in df.columns:
        df[prefer_col] = pd.to_datetime(df[prefer_col])
        df = df.set_index(prefer_col)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        # try best effort conversion for index
        try:
            # try epoch seconds/nanos or date-like strings
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("Failed to convert index to datetime and no 'datetime' column present.")
    return df

def render_trades_with_price(
    df_price: pd.DataFrame,
    trade_df,
    multiVwap,
    df_kama,
    use_trade_price: bool = True,   # True: 用交易日志里的 price；False: 用历史K线近似（不推荐）
    size_by_amount: bool = True,    # True: 根据 amount 缩放点大小
    show_vlines: bool = False,      # True: 在交易时间画竖线
    annotate: bool = False          # True: 在点旁标注 action 或 amount
) -> None:
    """
    在价格曲线上渲染交易点。优先使用交易日志中的实际成交/挂单价格（price）。

    要求：
    - df_price 包含 'close' 列（或至少一列可做主曲线，默认用 'close'）
    - trade_log 至少包含列：['action', 'price']，推荐包含 ['amount']（用于点大小）
    - TradeLogManager 需实现 manager.load_trade_log(exchange_id, symbol, timeframe) -> DataFrame

    交易标注规则（基于 action 字段）：
    - 'open_long'   : 绿色上三角
    - 'close_long' | 'stop_loss_long' | 'decay_long' : 红色下三角
    - 'open_short'  : 蓝色上三角
    - 'close_short' | 'stop_loss_short' | 'decay_short' : 橙色下三角
    """

    SFrame_vwap_up_sl2 = multiVwap.SFrame_vwap_up_sl2
    SFrame_vwap_down_sl2 = multiVwap.SFrame_vwap_down_sl2

    kama1 = df_kama['kama1']
    kama2 = df_kama['kama2']
    
    if trade_df is None or len(trade_df) == 0:
        raise ValueError("交易日志为空，无法渲染交易信号。")

    # 2) 统一时间索引
    df_price = _ensure_dt_index(df_price, prefer_col="datetime")
    trade_df = _ensure_dt_index(trade_df, prefer_col="datetime")

    # 3) 校验字段
    if "action" not in trade_df.columns:
        raise ValueError("trade_df 缺少 'action' 列。")
    if "price" not in trade_df.columns:
        raise ValueError("trade_df 缺少 'price' 列，无法按成交/挂单价格绘制。")
    trade_df["price"] = pd.to_numeric(trade_df["price"], errors="coerce")

    # 4) 点大小（按 amount）
    if size_by_amount and "amount" in trade_df.columns:
        amt = pd.to_numeric(trade_df["amount"], errors="coerce").fillna(0).abs()
        sizes = (40 + 10 * np.sqrt(amt)).clip(20, 200)
    else:
        sizes = pd.Series(60, index=trade_df.index)

    # 5) 价格主曲线
    price_col = "close" if "close" in df_price.columns else df_price.columns[0]
    plt.figure(figsize=(18, 7))
    plt.plot(df_price.index, df_price[price_col], label=f"{price_col.title()} Price", lw=1, color="#444")

    # 6) 历史价格近似（备用）
    def price_from_history(times: pd.DatetimeIndex) -> pd.Series:
        return df_price[price_col].reindex(times, method="nearest")

    # 7) 类别掩码
    actions = trade_df["action"].astype(str)
    m_open_long   = actions.str.contains("open_long", na=False, regex=False)
    m_close_long  = actions.str.contains("close_long|stop_loss_long|decay_long", na=False, regex=True)
    m_open_short  = actions.str.contains("open_short", na=False, regex=False)
    m_close_short = actions.str.contains("close_short|stop_loss_short|decay_short", na=False, regex=True)

    # 8) 绘制函数
    def scatter_trades(mask: pd.Series, marker: str, color: str, label: str):
        if mask.sum() == 0:
            return
        times = trade_df.index[mask]
        if use_trade_price:
            prices = trade_df.loc[mask, "price"]
        else:
            prices = price_from_history(times)

        # 清洗无效价格
        valid = prices.notna() & np.isfinite(prices.to_numpy())
        times = times[valid]
        prices = prices[valid]
        if len(times) == 0:
            return

        s = 60  #sizes.loc[times] if isinstance(sizes, pd.Series) else sizes
        plt.scatter(times, prices, marker=marker, color=color, s=s, label=label, alpha=0.9, edgecolor="none")

        if show_vlines:
            for t in times:
                plt.axvline(t, color=color, alpha=0.15, lw=1)

        if annotate:
            # 标注少量信息，避免过密
            for t, p in zip(times, prices):
                text = trade_df.loc[t, "action"]
                if isinstance(text, pd.Series):
                    text = text.iloc[0]
                plt.annotate(
                    text,
                    xy=(t, p),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    ha="center"
                )

    # 9) 绘制四类交易点
    scatter_trades(m_open_long,   marker="^", color="green",     label="Open Long")
    scatter_trades(m_close_long,  marker="v", color="red",       label="Close Long/Stop/Decay")
    scatter_trades(m_open_short,  marker="^", color="royalblue", label="Open Short")
    scatter_trades(m_close_short, marker="v", color="orange",    label="Close Short/Stop/Decay")

    # 6) 辅助：将索引安全转为 datetime（series.index 可能是 epoch 秒）
    def _to_dt_index(idx):
        if pd.api.types.is_datetime64_any_dtype(idx):
            return pd.to_datetime(idx)
        # 尝试按 epoch 秒
        try:
            return pd.to_datetime(idx, unit="s")
        except Exception:
            # 最后兜底尝试
            return pd.to_datetime(idx)
    # 7) 画四条指标线
    # 如果传了 fig（Plotly），用 add_trace；否则用 Matplotlib
    def _plot_one_series(name: str, series: pd.Series, color: str, linestyle="--"):
        if not isinstance(series, pd.Series) or series.empty:
            return
        x = _to_dt_index(series.index)
        y = series.values
        plt.plot(x, y, label=name, lw=1.5, color=color, linestyle=linestyle, alpha=0.8)

    # 对应你的配色和名称
    _plot_one_series("SFrame_vwap_up_sl2", SFrame_vwap_up_sl2, color="darkslategray", linestyle="--")
    _plot_one_series("SFrame_vwap_down_sl2", SFrame_vwap_down_sl2, color="darkslategray", linestyle="--")
    _plot_one_series("kama1", kama1, color="purple", linestyle="-")
    _plot_one_series("kama2", kama2, color="green", linestyle="-")

    title_suffix = "fill price from trade log" if use_trade_price else "nearest bar price (approx)"
    plt.title(f"Price and Trade Signals ({title_suffix})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

def render_equity_change(history):
    """
    绘制权益变化曲线。history 为 [(datetime, total_asset), ...] 或已包含上述两列的 DataFrame。
    """
    if history is None:
        raise ValueError("history 为空")

    if isinstance(history, (list, tuple)):
        df_hist = pd.DataFrame(history, columns=["datetime", "total_asset"])
        # 新增：先统一移除时区（处理混合 aware/naive）
        df_hist["datetime"] = df_hist["datetime"].apply(
            lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) and x.tzinfo is not None else x
        )
        df_hist["datetime"] = pd.to_datetime(df_hist["datetime"])
        df_hist = df_hist.set_index("datetime")
    elif isinstance(history, pd.DataFrame):
        df_hist = history.copy()
        if "datetime" in df_hist.columns:
            # 新增：先统一移除时区（处理混合 aware/naive）
            df_hist["datetime"] = df_hist["datetime"].apply(
                lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) and x.tzinfo is not None else x
            )
            df_hist["datetime"] = pd.to_datetime(df_hist["datetime"])
            df_hist = df_hist.set_index("datetime")
        elif not pd.api.types.is_datetime64_any_dtype(df_hist.index):
            df_hist.index = pd.to_datetime(df_hist.index)
        # 统一列名
        if "total_asset" not in df_hist.columns:
            raise ValueError("history DataFrame 需要包含 'total_asset' 列。")
    else:
        raise TypeError("history 需要为 list/tuple 或 pandas.DataFrame")

    # 新增：如果索引仍有 tz-aware，移除它（额外防护）
    if df_hist.index.tz is not None:
        df_hist.index = df_hist.index.tz_localize(None)  # 转换为 naive

    plt.figure(figsize=(14, 5))
    df_hist["total_asset"].plot(color="#2c7fb8", lw=1.2)
    plt.title("Backtest Account Equity")
    plt.xlabel("Time")
    plt.ylabel("Total Asset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# def render_equity_change(history):
#     """
#     绘制权益变化曲线。history 为 [(datetime, total_asset), ...] 或已包含上述两列的 DataFrame。
#     """
#     if history is None:
#         raise ValueError("history 为空")

#     if isinstance(history, (list, tuple)):
#         df_hist = pd.DataFrame(history, columns=["datetime", "total_asset"])
#         df_hist["datetime"] = pd.to_datetime(df_hist["datetime"])
#         df_hist = df_hist.set_index("datetime")
#     elif isinstance(history, pd.DataFrame):
#         df_hist = history.copy()
#         if "datetime" in df_hist.columns:
#             df_hist["datetime"] = pd.to_datetime(df_hist["datetime"])
#             df_hist = df_hist.set_index("datetime")
#         elif not pd.api.types.is_datetime64_any_dtype(df_hist.index):
#             df_hist.index = pd.to_datetime(df_hist.index)
#         # 统一列名
#         if "total_asset" not in df_hist.columns:
#             raise ValueError("history DataFrame 需要包含 'total_asset' 列。")
#     else:
#         raise TypeError("history 需要为 list/tuple 或 pandas.DataFrame")
#     # 移除时区信息，使索引为 naive DatetimeIndex（修复 Matplotlib 渲染问题）
#     if df_hist.index.tz is not None:  # 检查是否有时区
#         df_hist.index = df_hist.index.tz_localize(None)  # 移除时区，转换为 naive

#     plt.figure(figsize=(14, 5))
#     df_hist["total_asset"].plot(color="#2c7fb8", lw=1.2)
#     plt.title("Backtest Account Equity")
#     plt.xlabel("Time")
#     plt.ylabel("Total Asset")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
