import numpy as np  
import pandas as pd  
import os   
import pandas_ta as ta
import numpy as np
import pandas as pd

def vpvr_pct_band_vwap_boundary_series(src, vol, length, bins, pct, decay, vwap_series):
    """
    输入:
        src, vol, vwap_series: pd.Series, 价格、成交量、vwap等
        length: 滑窗长度
        bins: 分桶数
        pct: 累计包含的成交量比例，0~1
        decay: 时间加权衰减因子
    输出:
        pd.DataFrame：index与输入一致，含 band_low/band_high/vwap 三列
    """
    src = pd.Series(src)
    vol = pd.Series(vol)
    vwap_series = pd.Series(vwap_series)
    index = src.index
    band_low = np.full(len(src), np.nan)
    band_high = np.full(len(src), np.nan)
    for end in range(length-1, len(src)):
        slc = slice(end-length+1, end+1)
        s = src.iloc[slc].values
        v = vol.iloc[slc].values
        vv = vwap_series.iloc[end]
        lowv = np.min(s)
        highv = np.max(s)
        binsize = max((highv - lowv) / bins, 1e-8)
        accum = np.zeros(bins)
        for j in range(length):
            pricej = s[j]
            volj   = v[j]
            weightj = decay**(length - 1 - j)
            bini = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bini = max(0, min(bins - 1, bini))
            accum[bini] += volj * weightj
        total_vol = np.sum(accum)
        if total_vol == 0:
            continue
        # 找最大积点（POC）的位置
        center_idx = np.argmax(accum)
        sum_vol = accum[center_idx]
        left_idx = center_idx
        right_idx = center_idx
        while sum_vol < total_vol * pct:
            add_left = accum[left_idx - 1] if left_idx > 0 else -1
            add_right = accum[right_idx + 1] if right_idx < bins - 1 else -1
            if add_left >= add_right:
                if left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                elif right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                else:
                    break
            else:
                if right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                elif left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                else:
                    break
        # 计算边界加权均价
        # 下边界
        vlow = accum[left_idx]
        pricelow = lowv + binsize * (left_idx + 0.5)
        band_low[end] = (vlow * pricelow) / vlow if vlow > 0 else np.nan
        # 上边界
        vhigh = accum[right_idx]
        pricehigh = lowv + binsize * (right_idx + 0.5)
        band_high[end] = (vhigh * pricehigh) / vhigh if vhigh > 0 else np.nan

    return pd.Series(band_low, index=index), pd.Series(band_high, index=index)
    
def vpvr_pct_band_vwap_decay_series(src, vol, length, bins, pct, decay):
    """
    输入:
        src, vol: pd.Series (或可转Series的一维结构)
        length: 窗口长度
        bins: 分桶数
        pct: 累计volume覆盖比例 (如0.7)
        decay: 衰减系数 (如0.995)
    输出:
        pd.Series: 每根K线vwap, index与输入对齐
    """
    src = pd.Series(src)
    vol = pd.Series(vol)
    index = src.index
    result = np.full(len(src), np.nan)

    for end in range(length - 1, len(src)):
        slc = slice(end - length + 1, end + 1)
        s = src.iloc[slc].values
        v = vol.iloc[slc].values

        lowv, highv = np.min(s), np.max(s)
        binsize = max((highv - lowv) / bins, 1e-8)
        accum = np.zeros(bins)

        for j in range(length):
            pricej = s[j]
            volj = v[j]
            weightj = decay ** (length - 1 - j)
            bini = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bini = max(0, min(bins - 1, bini))
            accum[bini] += volj * weightj

        total_vol = np.sum(accum)
        if total_vol == 0:
            continue

        center_idx = np.argmax(accum)
        sum_vol = accum[center_idx]
        left_idx = center_idx
        right_idx = center_idx

        while sum_vol < total_vol * pct:
            add_left = accum[left_idx - 1] if left_idx > 0 else -1
            add_right = accum[right_idx + 1] if right_idx < bins - 1 else -1
            if add_left >= add_right:
                if left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                elif right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                else:
                    break
            else:
                if right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                elif left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                else:
                    break

        wsum, pvsum = 0.0, 0.0
        for i in range(left_idx, right_idx + 1):
            v_i = accum[i]
            price_i = lowv + binsize * (i + 0.5)
            wsum += v_i
            pvsum += v_i * price_i
        result[end] = pvsum / wsum if wsum > 0 else np.nan

    return pd.Series(result, index=index)

class MultiTFvpPOC:  
    def __init__(self,  
                 lambd=0.03,  
                 window_LFrame=12,  
                 window_HFrame=12*10,
                 window_SFrame=12*10 * 4,
  
                 std_window_LFrame=15):  
        self.lambd = lambd  
        self.window_LFrame = window_LFrame  
        self.window_HFrame = window_HFrame  
        self.window_SFrame = window_SFrame
        self.std_window_LFrame = std_window_LFrame  


        self.golden_split_factor = 1.618 

        # 预定义所有结果属性为None  
        self.LFrame_vpPOC_series = None  
        self.LFrame_ohlc5_series = None  
       

        self.SFrame_vpPOC = None  
        self.HFrame_ohlc5_series = None  
        self.HFrame_price_std = None  

        self.HFrame_vwap_up = None
        self.HFrame_vwap_up_getin = None
        self.HFrame_vwap_up_getout = None

        self.HFrame_vwap_down = None
        self.HFrame_vwap_down_getin = None
        self.HFrame_vwap_down_getout = None
        
    
    
    def calculate_SFrame_vpPOC_and_std(self, coin_date_df):  
        self.LFrame_vpPOC_series = vpvr_pct_band_vwap_decay_series(coin_date_df['close'], coin_date_df['vol'], self.window_LFrame, 40, 0.995, 0.99)
        
        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  

        weights_l, weights_c, weights_h, weights_o = 1.5, 2, 1.5, 0.5  
        weight_sum = weights_l + weights_c + weights_h + weights_o  
        ohlc5_values = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
        # self.LFrame_ohlc5_series = pd.Series(ohlc5_values.values, index=coin_date_df.index)  
        self.LFrame_ohlc5_series = pd.Series(close.values, index=coin_date_df.index)  
        self.SFrame_vpPOC =  vpvr_pct_band_vwap_decay_series(coin_date_df['close'], coin_date_df['vol'], self.window_SFrame, 40, 0.995, 0.99)   
        self.SFrame_vpPOC = ta.rma(self.SFrame_vpPOC, length=self.window_LFrame)
        self.HFrame_ohlc5_series = self.LFrame_ohlc5_series  

        self.HFrame_price_std = self.HFrame_ohlc5_series.rolling(window=self.window_HFrame, min_periods=1).std()  
        self.HFrame_price_std.index = coin_date_df.index  


        low_poc, high_poc = vpvr_pct_band_vwap_boundary_series(
                    src=coin_date_df['close'],
                    vol=coin_date_df['vol'],
                    length=self.window_SFrame,         # 滑动窗口长度
                    bins=40,
                    pct=0.95,
                    decay=0.995,
                    vwap_series=self.LFrame_vpPOC_series   # vwap请提前自行计算、赋值
                )
        self.HFrame_vwap_up = ta.rma(high_poc, length=self.window_LFrame)
        self.HFrame_vwap_up_getin = ta.rma(high_poc + self.HFrame_price_std, length=self.window_LFrame)
        self.HFrame_vwap_up_getout = ta.rma(high_poc - self.HFrame_price_std, length=self.window_LFrame)

        self.HFrame_vwap_down = ta.rma(low_poc, length=self.window_LFrame)
        self.HFrame_vwap_down_getin = ta.rma(low_poc - self.HFrame_price_std, length=self.window_LFrame)
        self.HFrame_vwap_down_getout = ta.rma(low_poc + self.HFrame_price_std, length=self.window_LFrame)


import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt

def plot_all_multiftfpoc_vars(multFramevpPOC, symbol=''):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('black')

    # 颜色定义
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
    }

    # 依次绘制所有线
    vars_to_plot = [
        'LFrame_ohlc5_series',
        'LFrame_vpPOC_series',
        'SFrame_vpPOC',
        'HFrame_vwap_up',
        'HFrame_vwap_up_getin',
        'HFrame_vwap_up_getout',
        'HFrame_vwap_down',
        'HFrame_vwap_down_getin',
        'HFrame_vwap_down_getout',
    ]
    for var in vars_to_plot:
        val = getattr(multFramevpPOC, var, None)
        if val is not None and hasattr(val, 'index') and hasattr(val, 'values'):
            ax.plot(val.index, val.values, label=var, color=colors.get(var, 'black'), linewidth=2 if 'vwap' not in var else 1.5, linestyle='-' if 'getin' not in var and 'getout' not in var else '--')

    # 设置y轴自适应
    all_y_values = []
    for var in vars_to_plot:
        val = getattr(multFramevpPOC, var, None)
        if val is not None and hasattr(val, 'values'):
            all_y_values.extend(val.values)
    if all_y_values:
        ymin = min(all_y_values) * 0.99
        ymax = max(all_y_values) * 1.01
        ax.set_ylim(ymin, ymax)

    ax.set_title(f"Combined vpPOC and VWAP Derived Lines - {symbol}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price/Value")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()

    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    prefix = f"{symbol}_" if symbol else ""
    filename = os.path.join(save_dir, f"{prefix}multFramevpPOC_combined_plot_{timestamp}.png")
    fig.savefig(filename)
    plt.close(fig)
    print(f"Plot saved to file: {filename}")

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