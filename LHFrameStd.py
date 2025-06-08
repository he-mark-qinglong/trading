import numpy as np  
import pandas as pd  
import os   
import pandas_ta as ta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import vwap_calc

class WindowConfig:
    def __init__(self):
        self.window_tau_l = int(12)
        self.window_tau_h = self.window_tau_l * 5
        self.window_tau_s = self.window_tau_h * 5

class MultiTFvp_poc:
    def __init__(self,
                 lambd=0.03,
                 window_LFrame=12,
                 window_HFrame=12*10,
                 window_SFrame=12*10*4,
                 std_window_LFrame=15):
        self.lambd               = lambd
        self.window_LFrame       = window_LFrame
        self.window_HFrame       = window_HFrame
        self.window_SFrame       = window_SFrame
        self.std_window_LFrame   = std_window_LFrame
        self.rma_smooth_window   = 9
        self.golden_split_factor = 1.618

        # 原始 OHLCV 全量存储
        self.df = None

        # 各种结果的全量 Series（索引与 self.df 保持一致）
        self.LFrame_vp_poc_series = pd.Series(dtype=float)
        self.HFrame_vp_poc       = pd.Series(dtype=float)
        self.SFrame_vp_poc       = pd.Series(dtype=float)
        self.slow_poc            = pd.Series(dtype=float)
        self.shigh_poc           = pd.Series(dtype=float)
        self.hlow_poc            = pd.Series(dtype=float)
        self.hhigh_poc           = pd.Series(dtype=float)

        # 向量化后的一些 Series
        self.LFrame_ohlc5_series = pd.Series(dtype=float)
        self.SFrame_price_std    = pd.Series(dtype=float)
        self.HFrame_price_std    = pd.Series(dtype=float)

        # 上下轨
        for attr in [
            'SFrame_vwap_up_poc','SFrame_vwap_up_getin','SFrame_vwap_up_getout',
            'SFrame_vwap_up_sl','SFrame_vwap_up_sl2','SFrame_vwap_down_poc',
            'SFrame_vwap_down_getin','SFrame_vwap_down_getout',
            'SFrame_vwap_down_sl','SFrame_vwap_down_sl2',
            'HFrame_vwap_up_poc','HFrame_vwap_up_getin','HFrame_vwap_up_sl',
            'HFrame_vwap_up_sl2','HFrame_vwap_down_poc','HFrame_vwap_down_getin',
            'HFrame_vwap_down_sl','HFrame_vwap_down_sl2'
        ]:
            setattr(self, attr, pd.Series(dtype=float))

    def _run_heavy(self, df_block, debug=False):
        """在一段 df_block 上并行计算 vp_poc 和 band，返回带 index 的 Series"""
        o, c, v = df_block['open'], df_block['close'], df_block['vol']
        with ThreadPoolExecutor(max_workers=3) as ex:
            fL = ex.submit(vwap_calc.vpvr_center_vwap_log_decay,
                           o, c, v,
                           self.window_LFrame, 40, 0.995, 0.99,
                           debug=debug)
            fH = ex.submit(vwap_calc.vpvr_center_vwap_log_decay,
                           o, c, v,
                           self.window_HFrame, 40, 1-0.14, 0.99,
                           debug=debug)
            fS = ex.submit(vwap_calc.vpvr_center_vwap_log_decay,
                           o, c, v,
                           self.window_SFrame, 40, 1-0.07, 0.99,
                           debug=debug)
            hvp = fH.result()
            svp = fS.result()
            fbH = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_HFrame, bins=40,
                            pct=0.07, decay=0.995, vwap_series=hvp,
                            debug=debug)
            fbS = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_SFrame, bins=40,
                            pct=0.07, decay=0.995, vwap_series=svp,
                            debug=debug)

        idx = df_block.index
        slow_arr, shigh_arr = fbS.result()
        hlow_arr,  hhigh_arr = fbH.result()
        return {
            'L':     pd.Series(fL.result(),   index=idx),
            'H':     pd.Series(hvp,           index=idx),
            'S':     pd.Series(svp,           index=idx),
            'slow':  pd.Series(slow_arr,      index=idx),
            'shigh': pd.Series(shigh_arr,     index=idx),
            'hlow':  pd.Series(hlow_arr,      index=idx),
            'hhigh': pd.Series(hhigh_arr,     index=idx),
        }

    def append_df(self, new_df: pd.DataFrame, debug=False):
        """
        增量更新：new_df 已保证索引是 DatetimeIndex，
        只 append 时间戳 > old_last_ts 的那部分结果。
        """
        # 1) 记录旧数据最后一个时间戳
        old_last_ts = None if self.df is None else self.df.index[-1]

        # 2) 合并 self.df 与 new_df，按 index 去重（保 new_df 的新/更新行）
        old_df = None if self.df is None else self.df.copy()
        if self.df is None:
            self.df = new_df.copy()
        else:
            tmp     = pd.concat([self.df, new_df])
            self.df = tmp[~tmp.index.duplicated(keep='last')]

        # 3) 构造 block = overlap 上下文 + new_df
        overlap = max(self.window_LFrame,
                      self.window_HFrame,
                      self.window_SFrame)
        if old_df is None:
            block = new_df
        else:
            head  = old_df.iloc[-overlap:] if len(old_df) >= overlap else old_df
            block = pd.concat([head, new_df], axis=0)

        # 4) 在 block 上跑最耗时部分
        out = self._run_heavy(block, debug=debug)

        # 5) 只取 > old_last_ts 的那段新结果，dropna 再 concat
        def take_new(old: pd.Series, new: pd.Series) -> pd.Series:
            if old_last_ts is not None:
                new = new[new.index > old_last_ts]
            new = new.dropna()
            if new.empty:
                return old
            return pd.concat([old, new])

        self.LFrame_vp_poc_series = take_new(self.LFrame_vp_poc_series, out['L'])
        self.HFrame_vp_poc       = take_new(self.HFrame_vp_poc,       out['H'])
        self.SFrame_vp_poc       = take_new(self.SFrame_vp_poc,       out['S'])
        self.slow_poc            = take_new(self.slow_poc,            out['slow'])
        self.shigh_poc           = take_new(self.shigh_poc,           out['shigh'])
        self.hlow_poc            = take_new(self.hlow_poc,            out['hlow'])
        self.hhigh_poc           = take_new(self.hhigh_poc,           out['hhigh'])

        # 6) 向量化后续计算
        self._run_vectorized()
        
        return self.df

    def _run_vectorized(self):
        df    = self.df
        close = df['close']

        # 1) OHLC5
        self.LFrame_ohlc5_series = pd.Series(close.values, index=close.index)

        # 2) RMA 平滑
        self.SFrame_vp_poc = ta.rma(self.SFrame_vp_poc,
                                    length=self.rma_smooth_window)

        # 3) swing 用于 std
        def swing(vp_poc):
            d_high = np.maximum(close - vp_poc, 0)
            dh_max = d_high.rolling(240).max().abs()
            d_low  = np.minimum(close - vp_poc, 0)
            dl_min = d_low.rolling(240).min().abs()
            return np.maximum(dh_max, dl_min)

        H_swing = swing(self.HFrame_vp_poc)
        S_swing = swing(self.SFrame_vp_poc)

        # 4) 价格 std
        self.SFrame_price_std = (
            close.rolling(self.window_SFrame).std() * 0.9
            + S_swing * 0.1
        ) * self.golden_split_factor

        h_std = (
            close.rolling(self.window_HFrame).std() * 0.9
            + H_swing * 0.1
        ) * self.golden_split_factor
        self.HFrame_price_std = pd.Series(h_std.values, index=close.index)

        # 5) 计算上下轨（RMA + max/min 合并逻辑）
        rma = lambda s: ta.rma(s, length=self.rma_smooth_window)
        slow, shigh = self.slow_poc, self.shigh_poc
        hlow, hhigh = self.hlow_poc, self.hhigh_poc

        # SFrame
        self.SFrame_vwap_up_poc    = rma(shigh)
        self.SFrame_vwap_up_getin  = rma(shigh + self.SFrame_price_std)
        self.SFrame_vwap_up_getout = rma(shigh - self.SFrame_price_std)
        self.SFrame_vwap_up_sl     = rma(shigh + 2*self.SFrame_price_std)
        self.SFrame_vwap_up_sl2    = rma(shigh + 4*self.SFrame_price_std)

        self.SFrame_vwap_down_poc    = rma(slow)
        self.SFrame_vwap_down_getin  = rma(slow - self.SFrame_price_std)
        self.SFrame_vwap_down_getout = rma(slow + self.SFrame_price_std)
        self.SFrame_vwap_down_sl     = rma(slow - 2*self.SFrame_price_std)
        self.SFrame_vwap_down_sl2    = rma(slow - 4*self.SFrame_price_std)

        # HFrame（取更保守的 max/min）
        self.HFrame_vwap_up_poc    = np.maximum(self.SFrame_vwap_up_getin,
                                                rma(hhigh))
        self.HFrame_vwap_up_getin  = np.maximum(self.SFrame_vwap_up_getin,
                                                rma(hhigh + self.HFrame_price_std))
        self.HFrame_vwap_up_sl     = np.maximum(self.SFrame_vwap_up_sl,
                                                rma(hhigh + 2*self.HFrame_price_std))
        self.HFrame_vwap_up_sl2    = np.maximum(self.SFrame_vwap_up_sl2,
                                                rma(hhigh + 4*self.HFrame_price_std))

        self.HFrame_vwap_down_poc    = np.minimum(self.SFrame_vwap_down_getin,
                                                  rma(hlow))
        self.HFrame_vwap_down_getin  = np.minimum(self.SFrame_vwap_down_getin,
                                                  rma(hlow - self.HFrame_price_std))
        self.HFrame_vwap_down_sl     = np.minimum(self.SFrame_vwap_down_sl,
                                                  rma(hlow - 2*self.HFrame_price_std))
        self.HFrame_vwap_down_sl2    = np.minimum(self.SFrame_vwap_down_sl2,
                                                  rma(hlow - 4*self.HFrame_price_std))

    def calculate_SFrame_vp_poc_and_std(self, coin_date_df, debug=False):
        """
        旧接口兼容：清空后全量跑一次。
        """
        self.df = None
        for attr in [
            'LFrame_vp_poc_series','HFrame_vp_poc','SFrame_vp_poc',
            'slow_poc','shigh_poc','hlow_poc','hhigh_poc'
        ]:
            setattr(self, attr, pd.Series(dtype=float))
        self.append_df(coin_date_df, debug=debug)

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
        'HFrame_vwap_up_sl2':        'orangered',
        'HFrame_vwap_down_getin':   'turquoise',
        'HFrame_vwap_down_sl':      'darkslategray',
        'HFrame_vwap_down_sl2':      'darkslategray',
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

