import numpy as np  
import pandas as pd  
import os   
import pandas_ta as ta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import vwap_calc

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
        self.HFrame_vpPOC = None  
        self.HFrame_price_std = None  

        self.SFrame_vwap_up = None
        self.SFrame_vwap_up_getin = None
        self.SFrame_vwap_up_getout = None

        self.SFrame_vwap_down = None
        self.SFrame_vwap_down_getin = None
        self.SFrame_vwap_down_getout = None
        
        self.SFrame_vwap_down_sl = None
        self.SFrame_vwap_up_sl = None
    
    # def calculate_SFrame_vpPOC_and_std(self, coin_date_df):  
    
    #     open = coin_date_df['open']
    #     high = coin_date_df['high']
    #     low = coin_date_df['low']  
    #     close = coin_date_df['close']  
    #     vol = coin_date_df['vol']

    #     self.LFrame_vpPOC_series = vwap_calc.vpvr_center_vwap_log_decay(open, close, vol, self.window_LFrame, 40, 0.995, 0.99)
         
    #     self.SFrame_vpPOC =  vwap_calc.vpvr_center_vwap_log_decay(open, close, vol, self.window_SFrame, 40, 0.995, 0.99)   

    #     low_poc, high_poc = vwap_calc.vpvr_pct_band_vwap_log_decay(
    #                 open_prices=open,
    #                 close_prices=close,
    #                 vol=vol,
    #                 length=self.window_SFrame,         # 滑动窗口长度
    #                 bins=40,
    #                 pct=0.07,
    #                 decay=0.995,
    #                 vwap_series=self.SFrame_vpPOC   # vwap请提前自行计算、赋值
    #             )
    #     self.LFrame_ohlc5_series = pd.Series(close.values, index=coin_date_df.index) 
       

    #     # weights_l, weights_c, weights_h, weights_o = 1.5, 2, 1.5, 0.5  
    #     # weight_sum = weights_l + weights_c + weights_h + weights_o  
    #     # ohlc5_values = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
    #     # self.LFrame_ohlc5_series = pd.Series(ohlc5_values.values, index=coin_date_df.index)  
        
    #     self.SFrame_vpPOC = ta.rma(self.SFrame_vpPOC, length=self.window_LFrame)

    #     # 假设 close, SFrame_vpPOC, ohlc5 为 pd.Series
    #     delta_high = np.maximum(close - self.SFrame_vpPOC, 0)
    #     # 过去240根K线内的最大delta_high的绝对值
    #     max_delta_high = delta_high.rolling(240).max().abs()

    #     delta_low = np.minimum(close - self.SFrame_vpPOC, 0)
    #     # 过去240根K线内的最小delta_low的绝对值
    #     min_delta_low = delta_low.rolling(240).min().abs()

    #     # 取二者较大值
    #     HFrame_max_swing = np.maximum(max_delta_high, min_delta_low)

    #     # ohlc5标准差, 按HFrame_vpLen窗口计算
    #     self.HFrame_price_std = close.rolling(self.window_HFrame).std() * 0.9 + HFrame_max_swing * 0.1
    #     self.HFrame_price_std.index = coin_date_df.index  
        
    #     self.SFrame_vwap_up = ta.rma(high_poc, length=self.window_LFrame)
    #     self.SFrame_vwap_up_getin = ta.rma(high_poc + self.HFrame_price_std, length=self.window_LFrame)
    #     self.SFrame_vwap_up_getout = ta.rma(high_poc - self.HFrame_price_std, length=self.window_LFrame)

    #     self.SFrame_vwap_down = ta.rma(low_poc, length=self.window_LFrame)
    #     self.SFrame_vwap_down_getin = ta.rma(low_poc - self.HFrame_price_std, length=self.window_LFrame)
    #     self.SFrame_vwap_down_getout = ta.rma(low_poc + self.HFrame_price_std, length=self.window_LFrame)

    #     self.SFrame_vwap_down_sl = ta.rma(low_poc - 2*self.HFrame_price_std, length=self.window_LFrame)
    #     self.SFrame_vwap_up_sl = ta.rma(high_poc + 2*self.HFrame_price_std, length=self.window_LFrame)
    def calculate_SFrame_vpPOC_and_std(self, coin_date_df):
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
                self.window_LFrame, 40, 0.995, 0.99
            )
            future_H = executor.submit(
                vwap_calc.vpvr_center_vwap_log_decay,
                open_, close, vol,
                self.window_HFrame, 40, 0.995, 0.99
            )
            future_S = executor.submit(
                vwap_calc.vpvr_center_vwap_log_decay,
                open_, close, vol,
                self.window_SFrame, 40, 0.995, 0.99
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
                vwap_series=sframe_vp
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
                vwap_series=hframe_vp
            )

            # 取回结果
            self.LFrame_vpPOC_series = future_L.result()
            self.SFrame_vpPOC       = sframe_vp
            self.HFrame_vpPOC       = hframe_vp
            slow_poc, shigh_poc       = sfuture_band.result()
            hlow_poc, hhigh_poc       = hfuture_band.result()

        # 以下全为向量化运算
        self.LFrame_ohlc5_series = pd.Series(close.values, index=coin_date_df.index)

        # 对 SFrame_vpPOC 做 RMA 平滑
        self.SFrame_vpPOC = ta.rma(self.SFrame_vpPOC, length=self.window_LFrame)

        # 计算 HFrame 的最大摆幅
        delta_high = np.maximum(close - self.HFrame_vpPOC, 0)
        max_delta_high = delta_high.rolling(240).max().abs()
        delta_low  = np.minimum(close - self.HFrame_vpPOC, 0)
        min_delta_low = delta_low.rolling(240).min().abs()
        HFrame_max_swing = np.maximum(max_delta_high, min_delta_low)
        
        # 计算 SFrame 的最大摆幅
        delta_high = np.maximum(close - self.SFrame_vpPOC, 0)
        max_delta_high = delta_high.rolling(240).max().abs()
        delta_low  = np.minimum(close - self.SFrame_vpPOC, 0)
        min_delta_low = delta_low.rolling(240).min().abs()
        SFrame_max_swing = np.maximum(max_delta_high, min_delta_low)

        # 1. 计算 SFrame 的价格标准差
        # HFrame 的价格标准差
        self.SFrame_price_std = (
            close.rolling(self.window_HFrame).std() * 0.9
            + SFrame_max_swing * 0.1
        )
        self.SFrame_price_std.index = coin_date_df.index
        
        # 2. SFrame 上下边界及进出场线（用 ta.rma）
        self.SFrame_vwap_up        = ta.rma(shigh_poc, length=self.window_LFrame)
        self.SFrame_vwap_up_getin  = ta.rma(shigh_poc + self.SFrame_price_std, length=self.window_LFrame)
        self.SFrame_vwap_up_getout = ta.rma(shigh_poc - self.SFrame_price_std, length=self.window_LFrame)
        self.SFrame_vwap_up_sl     = ta.rma(shigh_poc + 2 * self.SFrame_price_std, length=self.window_LFrame)

        self.SFrame_vwap_down        = ta.rma(slow_poc, length=self.window_LFrame)
        self.SFrame_vwap_down_getin  = ta.rma(slow_poc - self.SFrame_price_std, length=self.window_LFrame)
        self.SFrame_vwap_down_getout = ta.rma(slow_poc + self.SFrame_price_std, length=self.window_LFrame)
        self.SFrame_vwap_down_sl     = ta.rma(slow_poc - 2 * self.SFrame_price_std, length=self.window_LFrame)

        # 3. 计算 HFrame 的价格标准差，并对齐索引
        h_std = close.rolling(self.window_HFrame).std() * 0.9 + HFrame_max_swing * 0.1
        self.HFrame_price_std = h_std.reindex(coin_date_df.index)

        # 4. HFrame 上下边界及进出场线（元素级取最大/最小）
        #    注意这里继续用 length=self.window_LFrame，若要用 HFrame 窗口可自行调整
        rma = lambda series: ta.rma(series, length=self.window_LFrame)

        self.HFrame_vwap_up        = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc))
        self.HFrame_vwap_up_getin  = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc + self.HFrame_price_std))
        self.HFrame_vwap_up_getout = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc - self.HFrame_price_std))
        self.HFrame_vwap_up_sl     = np.maximum(self.SFrame_vwap_up_getin,    rma(hhigh_poc + 2 * self.HFrame_price_std))

        self.HFrame_vwap_down        = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc))
        self.HFrame_vwap_down_getin  = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc - self.HFrame_price_std))
        self.HFrame_vwap_down_getout = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc + self.HFrame_price_std))
        self.HFrame_vwap_down_sl     = np.minimum(self.SFrame_vwap_down_getin,  rma(hlow_poc - 2 * self.HFrame_price_std))
        
import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt

def plot_all_multiftfpoc_vars(multFramevpPOC, symbol='', is_trading= False, save_to_file=True):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('black')

    # 颜色定义
    colors = {
        'LFrame_vpPOC_series': 'yellow',
        'LFrame_ohlc5_series': 'green',
        'SFrame_vpPOC': 'purple',
        'SFrame_vwap_up': 'red',
        'SFrame_vwap_up_getin': 'orange',
        'SFrame_vwap_up_getout': 'chocolate',
        'SFrame_vwap_down': 'blue',
        'SFrame_vwap_down_getin': 'deepskyblue',
        'SFrame_vwap_down_getout': 'cyan',

        'SFrame_vwap_up_sl': 'red',
        'SFrame_vwap_down_sl': 'green',
    }

    # 依次绘制所有线
    vars_to_plot = [
        'LFrame_ohlc5_series',
        'LFrame_vpPOC_series',
        'SFrame_vpPOC',
        'SFrame_vwap_up',
        'SFrame_vwap_up_getin',
        'SFrame_vwap_up_getout',
        'SFrame_vwap_down',
        'SFrame_vwap_down_getin',
        'SFrame_vwap_down_getout',
        'SFrame_vwap_up_sl',
        'SFrame_vwap_down_sl',
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

    if save_to_file:
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)

        from datetime import datetime
        # 年月日_时-分-秒 (更可读)
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S-%f")[:-3]  # 去掉后3位微秒
        # 输出: 20250531_09-04-50_629

        prefix = f"{symbol}_" if symbol else ""
        if is_trading:
            prefix = f"trade_{prefix}" 
        filename = os.path.join(save_dir, f"{prefix}multFramevpPOC_combined_plot_{timestamp}.png")
        fig.savefig(filename)
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

