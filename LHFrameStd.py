import numpy as np  
import pandas as pd  
import os   
import pandas_ta as ta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import vwap_calc

class MultiTFvp_poc:  
    def __init__(self,  
                 lambd=0.03,  
                 window_LFrame=12,  
                 window_HFrame=12*10,
                 window_SFrame=12*10 * 4,
  
                 std_window_LFrame=15):  
        self.lambd = lambd  
        self.rma_smooth_window = 9
        self.window_LFrame = window_LFrame  
        self.window_HFrame = window_HFrame  
        self.window_SFrame = window_SFrame
        self.std_window_LFrame = std_window_LFrame  


        self.golden_split_factor = 1.618 

        # 预定义所有结果属性为None  
        self.LFrame_vp_poc_series = None  
        self.LFrame_ohlc5_series = None  
       

        self.SFrame_vp_poc = None  
        self.HFrame_vp_poc = None  
        self.HFrame_price_std = None  

        self.SFrame_vwap_up = None
        self.SFrame_vwap_up_getin = None
        self.SFrame_vwap_up_getout = None

        self.SFrame_vwap_down = None
        self.SFrame_vwap_down_getin = None
        self.SFrame_vwap_down_getout = None
        
        self.SFrame_vwap_down_sl = None
        self.SFrame_vwap_up_sl = None
    
    def calculate_SFrame_vp_poc_and_std(self, coin_date_df, debug=False):
        open_  = coin_date_df['open']
        high   = coin_date_df['high']
        low    = coin_date_df['low']
        close  = coin_date_df['close']
        vol    = coin_date_df['vol']

        # 并行执行 LFrame 和 SFrame 的中心 VWAP 计算
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_L = executor.submit(
                vwap_calc.vpvr_center_vwap_log_decay,
                open_, close, vol,
                self.window_LFrame, 40, 0.995, 0.99,
                debug=debug,
            )
            future_H = executor.submit(
                vwap_calc.vpvr_center_vwap_log_decay,
                open_, close, vol,
                self.window_HFrame, 40, 1 - (0.07 * 2), 0.99,
                debug=debug,
            )
            future_S = executor.submit(
                vwap_calc.vpvr_center_vwap_log_decay,
                open_, close, vol,
                self.window_SFrame, 40, 1 - (0.07 * 1), 0.99,
                debug=debug,
            )

            hframe_vp = future_H.result()

            # 等待 SFrame 完成后再启动 Band 计算
            sframe_vp = future_S.result()
            sfuture_band = executor.submit(
                vwap_calc.vpvr_pct_band_vwap_log_decay,
                open_prices=open_,
                close_prices=close,
                vol=vol,
                length=self.window_SFrame,
                bins=40,
                pct=0.07,
                decay=0.995,
                vwap_series=sframe_vp,
                debug=debug,
            )
            hfuture_band = executor.submit(
                vwap_calc.vpvr_pct_band_vwap_log_decay,
                open_prices=open_,
                close_prices=close,
                vol=vol,
                length=self.window_HFrame,
                bins=40,
                pct=0.07,
                decay=0.995,
                vwap_series=hframe_vp,
                debug=debug,
            )

            # 取回结果
            self.LFrame_vp_poc_series = future_L.result()
            self.SFrame_vp_poc       = sframe_vp
            self.HFrame_vp_poc       = hframe_vp
            slow_poc, shigh_poc       = sfuture_band.result()
            hlow_poc, hhigh_poc       = hfuture_band.result()

        # 以下全为向量化运算
        self.LFrame_ohlc5_series = pd.Series(close.values, index=coin_date_df.index)

        # 对 SFrame_vp_poc 做 RMA 平滑
        self.SFrame_vp_poc = ta.rma(self.SFrame_vp_poc, length=self.rma_smooth_window)

        # 计算 HFrame 的最大摆幅
        delta_high = np.maximum(close - self.HFrame_vp_poc, 0)
        max_delta_high = delta_high.rolling(240).max().abs()
        delta_low  = np.minimum(close - self.HFrame_vp_poc, 0)
        min_delta_low = delta_low.rolling(240).min().abs()
        HFrame_max_swing = np.maximum(max_delta_high, min_delta_low)
        
        # 计算 SFrame 的最大摆幅
        delta_high = np.maximum(close - self.SFrame_vp_poc, 0)
        max_delta_high = delta_high.rolling(240).max().abs()
        delta_low  = np.minimum(close - self.SFrame_vp_poc, 0)
        min_delta_low = delta_low.rolling(240).min().abs()
        SFrame_max_swing = np.maximum(max_delta_high, min_delta_low)

        # 1. 计算 SFrame 的价格标准差
        # HFrame 的价格标准差
        self.SFrame_price_std = (
            close.rolling(self.window_HFrame).std() * 0.9
            + SFrame_max_swing * 0.1
        ) * self.golden_split_factor
        self.SFrame_price_std.index = coin_date_df.index
        
        # 2. SFrame 上下边界及进出场线（用 ta.rma）
        self.SFrame_vwap_up_poc        = ta.rma(shigh_poc, length=self.rma_smooth_window)
        self.SFrame_vwap_up_getin  = ta.rma(shigh_poc + self.SFrame_price_std, length=self.rma_smooth_window)
        self.SFrame_vwap_up_getout = ta.rma(shigh_poc - self.SFrame_price_std, length=self.rma_smooth_window)
        self.SFrame_vwap_up_sl     = ta.rma(shigh_poc + 2 * self.SFrame_price_std, length=self.rma_smooth_window)

        self.SFrame_vwap_down_poc        = ta.rma(slow_poc, length=self.rma_smooth_window)
        self.SFrame_vwap_down_getin  = ta.rma(slow_poc - self.SFrame_price_std, length=self.rma_smooth_window)
        self.SFrame_vwap_down_getout = ta.rma(slow_poc + self.SFrame_price_std, length=self.rma_smooth_window)
        self.SFrame_vwap_down_sl     = ta.rma(slow_poc - 2 * self.SFrame_price_std, length=self.rma_smooth_window)

        # 3. 计算 HFrame 的价格标准差，并对齐索引
        h_std = close.rolling(self.window_HFrame).std() * 0.9 + HFrame_max_swing * 0.1
        self.HFrame_price_std = h_std.reindex(coin_date_df.index) * self.golden_split_factor

        # 4. HFrame 上下边界及进出场线（元素级取最大/最小）
        rma = lambda series: ta.rma(series, length=self.rma_smooth_window)

        self.HFrame_vwap_up_poc        = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc))
        self.HFrame_vwap_up_getin  = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc + self.HFrame_price_std))
        self.HFrame_vwap_up_sl     = np.maximum(self.SFrame_vwap_up_sl,    rma(hhigh_poc + 2 * self.HFrame_price_std))

        self.HFrame_vwap_down_poc        = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc))
        self.HFrame_vwap_down_getin  = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc - self.HFrame_price_std))
        self.HFrame_vwap_down_sl     = np.minimum(self.SFrame_vwap_down_sl,  rma(hlow_poc - 2 * self.HFrame_price_std))
        
import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt


def plot_all_multiftfpoc_vars(multFramevp_poc, symbol='', is_trading=False, save_to_file=True):
    import matplotlib.dates as mdates
    from datetime import datetime

    fig, (ax_k, ax_vol) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [7, 3]},
                                       constrained_layout=True)
    fig.patch.set_facecolor('black')
    ax_k.set_facecolor('black')
    ax_vol.set_facecolor('black')

    # 颜色与变量名列表——和 Dash 一致
    colors = {
        'LFrame_vp_poc_series':     'yellow',
        'SFrame_vp_poc':            'purple',
        'SFrame_vwap_up_poc':       'red',
        'SFrame_vwap_up_sl':        'firebrick',
        'SFrame_vwap_down_poc':     'blue',
        'SFrame_vwap_down_sl':      'seagreen',
        'HFrame_vwap_up_getin':     'deeppink',
        'HFrame_vwap_up_sl':        'orangered',
        'HFrame_vwap_down_getin':   'turquoise',
        'HFrame_vwap_down_sl':      'darkslategray',
    }
    vars_to_plot = list(colors.keys())

    # === 参考DASH裁剪法：找SFrame_vwap_up_poc第一个有效点 ===
    base_var = 'SFrame_vwap_up_poc'
    base_series = getattr(multFramevp_poc, base_var, None)
    start = None
    if isinstance(base_series, (pd.Series, np.ndarray)) and hasattr(base_series, "first_valid_index"):
        start = base_series.first_valid_index()

    # 主 DataFrame
    df = getattr(multFramevp_poc, 'df', None)
    if start is not None:
        # DataFrame裁剪
        if isinstance(df, pd.DataFrame) and start in df.index:
            df = df.loc[start:].copy()
        # 所有序列统一loc[start:]，完全和Dash那段一样
        for var in vars_to_plot + ["HFrame_vwap_up_getin", "HFrame_vwap_up_sl", "HFrame_vwap_down_sl", "HFrame_vwap_down_getin"]:
            s = getattr(multFramevp_poc, var, None)
            if isinstance(s, pd.Series) and start in s.index:
                setattr(multFramevp_poc, var, s.loc[start:])

    # X轴时间
    datetime_index = None
    if df is not None and 'datetime' in df.columns:
        datetime_index = df['datetime']
    else:
        for var in vars_to_plot:
            s = getattr(multFramevp_poc, var, None)
            if isinstance(s, pd.Series):
                datetime_index = s.index
                break

    # ---主K线/close---
    if df is not None and set(['open', 'high', 'low', 'close']).issubset(df.columns):
        o,h,l,c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        times = datetime_index if datetime_index is not None else df.index
        try:
            from mplfinance.original_flavor import candlestick_ohlc
            quotes = np.column_stack([mdates.date2num(times), o,h,l,c])
            candlestick_ohlc(ax_k, quotes, width=0.0015, colorup='lime', colordown='red', alpha=0.7)
        except ImportError:
            ax_k.plot(times, c, color='white', label='Close', linewidth=1)

    elif datetime_index is not None:
        for var in ['LFrame_ohlc5_series', 'close']:
            cval = getattr(multFramevp_poc, var, None)
            if isinstance(cval, pd.Series):
                ax_k.plot(cval.index, cval.values, color='lightgray', linewidth=1, label=var)
                break

    # ---所有系列线---
    for var in vars_to_plot:
        series = getattr(multFramevp_poc, var, None)
        if not isinstance(series, pd.Series) or series.isna().all():
            continue
        ax_k.plot(series.index, series.values,
                  label=var, color=colors.get(var, 'white'),
                  linewidth=2 if 'HFrame' not in var else 1,
                  linestyle='-' if '_poc' in var else 'dotted')

    # -------- 色带填充（严格和Dash同步！！）--------
    # HFrame上轨
    h_getin = getattr(multFramevp_poc, "HFrame_vwap_up_getin", None)
    h_sl    = getattr(multFramevp_poc, "HFrame_vwap_up_sl", None)
    if isinstance(h_getin, pd.Series) and isinstance(h_sl, pd.Series):
        ax_k.fill_between(h_getin.index, h_getin.values, h_sl.values,
                          color="hotpink", alpha=0.20, label="HFrame_up_band")

    # HFrame下轨
    h_down_sl = getattr(multFramevp_poc, "HFrame_vwap_down_sl", None)
    h_down_getin = getattr(multFramevp_poc, "HFrame_vwap_down_getin", None)
    if isinstance(h_down_sl, pd.Series) and isinstance(h_down_getin, pd.Series):
        ax_k.fill_between(h_down_sl.index, h_down_sl.values, h_down_getin.values,
                          color="deepskyblue", alpha=0.20, label="HFrame_down_band")

    # 成交量柱状图
    if df is not None and 'vol' in df.columns:
        times = datetime_index if datetime_index is not None else df.index
        ax_vol.bar(times, df['vol'].values, width=0.003, color='dodgerblue', alpha=0.6, label='Volume')

    # y轴自适应
    try:
        all_y = []
        for var in vars_to_plot:
            series = getattr(multFramevp_poc, var, None)
            if isinstance(series, pd.Series):
                all_y.extend(series.dropna().values)
        if all_y:
            ymin = min(all_y) * 0.99
            ymax = max(all_y) * 1.01
            ax_k.set_ylim(ymin, ymax)
    except:
        pass

    ax_k.set_title(f"OKX 4s K-line + vp_poc/VWAP Derived Series - {symbol}", color='w')
    ax_k.set_ylabel("Price/Value", color='w')
    ax_vol.set_ylabel("Volume", color='w')
    for spine in ax_k.spines.values():
        spine.set_color('white')
    for spine in ax_vol.spines.values():
        spine.set_color('white')
    ax_k.tick_params(axis='x', labelrotation=15, colors='white')
    ax_k.tick_params(axis='y', colors='white')
    ax_vol.tick_params(axis='x', labelrotation=15, colors='white')
    ax_vol.tick_params(axis='y', colors='white')
    ax_k.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,  # 可自行调节，一行排4~5个
        fontsize="small",
        facecolor="black",
        labelcolor="white"
    )

    ax_k.grid(True, alpha=0.2)
    ax_vol.grid(True, alpha=0.15)
    # X轴日期格式
    if datetime_index is not None:
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate(rotation=10)

    if save_to_file:
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S-%f")[:-3]
        prefix = f"{symbol}_" if symbol else ""
        if is_trading:
            prefix = f"trade_{prefix}"
        filename = os.path.join(save_dir, f"{prefix}multFramevp_poc_combined_plot_{timestamp}.png")
        fig.savefig(filename, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Plot saved to file: {filename}")
    else:
        return fig

def calc_atr(df, period=14, high_col="high", low_col="low", close_col="close"):
    """
    计算ATR并返回Series，可自动识别DataFrame列名
    Params:
        df      -- 带有high/low/close列的DataFrame
        period  -- ATR窗口期（默认14）
        high_col, low_col, close_col -- 列名，如自定义表头可改
    Returns:
        Series: ATR序列。如需直接加列可用 df['ATR'] = ...
    """
    high = df[high_col]
    low  = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def rsi_with_ema_smoothing(coin_date_df, length=13):  
    close = coin_date_df.iloc[:, 4]  

    delta = close.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain.iloc[0] = 0
    loss.iloc[0] = 0

    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()

    # 将初始值赋到第length-1的位置，前面都是NaN
    avg_gain = avg_gain.to_numpy()
    avg_loss = avg_loss.to_numpy()
    gain = gain.to_numpy()
    loss = loss.to_numpy()

    # 从length位置开始迭代计算后续avg_gain和avg_loss
    for i in range(length, len(close)):  
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain[i]) / length  
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss[i]) / length  

    rs = avg_gain / avg_loss
    # 转回pd.Series，并赋予index
    rsi_raw = pd.Series(100 - 100 / (1 + rs), index=close.index)

    # 处理除零及特殊情况
    rsi_raw[avg_loss == 0] = 100
    rsi_raw[(avg_gain == 0) & (avg_loss == 0)] = 0

    # EMA平滑
    rsi_ema = rsi_raw.ewm(alpha=2/(length+1), adjust=False, min_periods=length).mean()
    
    return rsi_ema

