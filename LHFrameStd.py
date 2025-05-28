import numpy as np  
import pandas as pd  
import os   

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
        self.std_window_LFrame = std_window_LFrame  


        self.golden_split_factor = 1.618 

        # 预定义所有结果属性为None  
        self.LFrame_vpPOC_series = None  
        self.LFrame_ohlc5_series = None  
       

        self.HFrame_vpPOC = None  
        self.HFrame_ohlc5_series = None  
        self.HFrame_price_std = None  

        # 定义HFrame标准差边界成员变量（up和down分开）  
        self.HFrame_std_0_5_up = None  
        self.HFrame_std_0_5_down = None  
        self.HFrame_std_1_0_up = None  
        self.HFrame_std_1_0_down = None  
        self.HFrame_std_1_5_up = None  
        self.HFrame_std_1_5_down = None  
        self.HFrame_std_2_0_up = None  
        self.HFrame_std_2_0_down = None  
        self.HFrame_std_3_0_up = None  
        self.HFrame_std_3_0_down = None  
        self.HFrame_std_3_5_up = None  
        self.HFrame_std_3_5_down = None 
    
    
    def calculate_HFrame_vpPOC_and_std(self, coin_date_df):  
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
        self.HFrame_vpPOC =  vpvr_pct_band_vwap_decay_series(coin_date_df['close'], coin_date_df['vol'], self.window_HFrame, 40, 0.995, 0.99)   
        self.HFrame_ohlc5_series = self.LFrame_ohlc5_series  

        self.HFrame_price_std = self.HFrame_ohlc5_series.rolling(window=self.window_HFrame, min_periods=1).std()  
        self.HFrame_price_std.index = coin_date_df.index  
        
        multipliers = [0.5, 1.0, 1.5, 2.0] 
        for m in multipliers:  
            upper = self.HFrame_vpPOC + m * self.HFrame_price_std * self.golden_split_factor
            lower = self.HFrame_vpPOC - m * self.HFrame_price_std * self.golden_split_factor
            setattr(self, f'HFrame_std_{str(m).replace(".", "_")}_up', upper)  
            setattr(self, f'HFrame_std_{str(m).replace(".", "_")}_down', lower)  


        low_poc, high_poc = vpvr_pct_band_vwap_boundary_series(
                    src=coin_date_df['close'],
                    vol=coin_date_df['vol'],
                    length=self.window_HFrame,         # 滑动窗口长度
                    bins=40,
                    pct=0.95,
                    decay=0.995,
                    vwap_series=self.LFrame_vpPOC_series   # vwap请提前自行计算、赋值
                )
        setattr(self, f'HFrame_vwap_up', high_poc)  
        setattr(self, f'HFrame_vwap_down', low_poc) 

    # def rsi_with_ema_smoothing(self, coin_date_df, length=13):  
    #     close = coin_date_df.iloc[:, 4]  

    #     delta = close.diff(1)
    #     gain = delta.clip(lower=0)
    #     loss = -delta.clip(upper=0)

    #     gain.iloc[0] = 0
    #     loss.iloc[0] = 0

    #     avg_gain = gain.rolling(window=length, min_periods=length).mean()
    #     avg_loss = loss.rolling(window=length, min_periods=length).mean()

    #     # 将初始值赋到第length-1的位置，前面都是NaN
    #     avg_gain = avg_gain.to_numpy()
    #     avg_loss = avg_loss.to_numpy()
    #     gain = gain.to_numpy()
    #     loss = loss.to_numpy()

    #     # 从length位置开始迭代计算后续avg_gain和avg_loss
    #     for i in range(length, len(close)):  
    #         avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain[i]) / length  
    #         avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss[i]) / length  

    #     rs = avg_gain / avg_loss
    #     # 转回pd.Series，并赋予index
    #     rsi_raw = pd.Series(100 - 100 / (1 + rs), index=close.index)

    #     # 处理除零及特殊情况
    #     rsi_raw[avg_loss == 0] = 100
    #     rsi_raw[(avg_gain == 0) & (avg_loss == 0)] = 0

    #     # EMA平滑
    #     rsi_ema = rsi_raw.ewm(alpha=2/(length+1), adjust=False, min_periods=length).mean()
        
    #     return rsi_ema

import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt
def plot_all_multiftfpoc_vars(multFramevpPOC, symbol=''):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('black')

    # LFrame颜色定义
    colors = {
        'LFrame_vpPOC_series': 'yellow',
        'LFrame_ohlc5_series': 'green',
        'LFrame_std_2_upper': 'cyan',
        'LFrame_std_2_lower': 'cyan',
        'HFrame_vpPOC': 'purple',
        'HFrame_ohlc5_series': 'orange',
        # 新增的两根线
        'HFrame_vwap_up': 'black',
        'HFrame_vwap_down': 'black'
    }

    # 绘制LFrame线
    for var in [
        'LFrame_ohlc5_series',
        'LFrame_std_2_upper', 'LFrame_std_2_lower',
    ]:
        val = getattr(multFramevpPOC, var, None)
        if val is not None and hasattr(val, 'index') and hasattr(val, 'values'):
            ax.plot(val.index, val.values, label=var, color=colors.get(var, 'black'), linewidth=1)

    lframe_vp = getattr(multFramevpPOC, 'LFrame_vpPOC_series', None)
    if lframe_vp is not None and hasattr(lframe_vp, 'index') and hasattr(lframe_vp, 'values'):
        ax.plot(lframe_vp.index, lframe_vp.values, label='LFrame vpPOC', color=colors['LFrame_vpPOC_series'], linewidth=2)

    # 绘制HFrame vpPOC
    hframe_vp = getattr(multFramevpPOC, 'HFrame_vpPOC', None)
    if hframe_vp is not None and hasattr(hframe_vp, 'index') and hasattr(hframe_vp, 'values'):
        ax.plot(hframe_vp.index, hframe_vp.values, label='HFrame vpPOC', color=colors['HFrame_vpPOC'], linewidth=3)

    # 新增 - HFrame上下边界线
    hframe_vwap_up = getattr(multFramevpPOC, 'HFrame_vwap_up', None)
    if hframe_vwap_up is not None and hasattr(hframe_vwap_up, 'index') and hasattr(hframe_vwap_up, 'values'):
        ax.plot(hframe_vwap_up.index, hframe_vwap_up.values, label='HFrame vwap upper', color=colors['HFrame_vwap_up'], linestyle='--', linewidth=1.5)

    hframe_vwap_down = getattr(multFramevpPOC, 'HFrame_vwap_down', None)
    if hframe_vwap_down is not None and hasattr(hframe_vwap_down, 'index') and hasattr(hframe_vwap_down, 'values'):
        ax.plot(hframe_vwap_down.index, hframe_vwap_down.values, label='HFrame vwap lower', color=colors['HFrame_vwap_down'], linestyle='--', linewidth=1.5)

    # HFrame标准差倍数和对应颜色
    hframe_std_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]
    # 颜色方案，1倍为红色，其他统一蓝色
    multiplier_colors = {
        1.0: 'red'
    }
    default_color = 'blue'

    for m in hframe_std_multipliers:
        upper = getattr(multFramevpPOC, f'HFrame_std_{str(m).replace(".", "_")}_up', None)
        lower = getattr(multFramevpPOC, f'HFrame_std_{str(m).replace(".", "_")}_down', None)
        if upper is None or lower is None:
            continue
        if not (hasattr(upper, 'index') and hasattr(upper, 'values') and hasattr(lower, 'values')):
            continue
        color = multiplier_colors.get(m, default_color)
        label_upper = f'HFrame +{m}σ' if m == 1.0 else None
        label_lower = f'HFrame -{m}σ' if m == 1.0 else None

        ax.plot(upper.index, upper.values, label=label_upper, color=color, linewidth=1)
        ax.plot(lower.index, lower.values, label=label_lower, color=color, linewidth=1)

    # 设置y轴范围以适应所有数据
    all_y_values = []
    for var in [
        'LFrame_ohlc5_series',
        'LFrame_std_2_upper', 'LFrame_std_2_lower',
        'LFrame_vpPOC_series',
        'HFrame_vpPOC',
        'HFrame_vwap_up', 'HFrame_vwap_down'   # 新增这两根线
    ]:
        val = getattr(multFramevpPOC, var, None)
        if val is not None and hasattr(val, 'values'):
            all_y_values.extend(val.values)

    for m in hframe_std_multipliers:
        upper = getattr(multFramevpPOC, f'HFrame_std_{str(m).replace(".", "_")}_up', None)
        lower = getattr(multFramevpPOC, f'HFrame_std_{str(m).replace(".", "_")}_down', None)
        if upper is not None and lower is not None:
            all_y_values.extend(upper.values)
            all_y_values.extend(lower.values)

    if all_y_values:
        ymin = min(all_y_values) * 0.99
        ymax = max(all_y_values) * 1.01
        ax.set_ylim(ymin, ymax)

    ax.set_title(f"Combined vpPOC and Std Lines - {symbol}")
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