import numpy as np  
import pandas as pd  
import os   

class MultiTFvpPOC:  
    def __init__(self,  
                 lambd=0.03,  
                 window_LFrame=15,  
                 window_HFrame=15*48,  
                 std_window_LFrame=15):  
        self.lambd = lambd  
        self.window_LFrame = window_LFrame  
        self.window_HFrame = window_HFrame  
        self.std_window_LFrame = std_window_LFrame  

        # 预定义所有结果属性为None  
        self.LFrame_vpPOC_series = None  
        self.LFrame_ohlc5_series = None  
        self.LFrame_rolling_std = None  
        self.LFrame_std_2_upper = None  
        self.LFrame_std_2_lower = None  
        self.LFrame_std_4_upper = None  
        self.LFrame_std_4_lower = None  

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

    @staticmethod  
    def calculate_ohlc5(coin_date_df: pd.DataFrame) -> pd.Series:  
        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  

        weights_l, weights_c, weights_h, weights_o = 1.5, 2.0, 1.5, 0.5  
        weight_sum = weights_l + weights_c + weights_h + weights_o  

        ohlc5 = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
        return ohlc5  

    def twpoc_calc_with_lambda_for_LFrame(self, coin_date_df ):  
        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  
        volume = coin_date_df.iloc[:, 5]  

        decay = np.exp(-self.lambd)  
        n = len(coin_date_df)  
        twpoc_values = np.full(n, np.nan)  

        for idx in range(n):  
            start_idx = max(0, idx - self.window_LFrame + 1)  
            sub_open = open_.iloc[start_idx:idx + 1]  
            sub_high = high.iloc[start_idx:idx + 1]  
            sub_low = low.iloc[start_idx:idx + 1]  
            sub_close = close.iloc[start_idx:idx + 1]  
            sub_volume = volume.iloc[start_idx:idx + 1]  

            origin_LFrame_vpPOC = np.average(sub_close, weights=sub_volume) if sub_volume.sum() > 0 else np.nan  

            if np.isnan(origin_LFrame_vpPOC):  
                twpoc_values[idx] = np.nan  
                continue  

            # 区分上涨和下跌通过最后一个收盘和vpPOC比较，直接区分weight结构  
            if sub_close.iloc[-1] >= origin_LFrame_vpPOC:  
                weights_l, weights_c, weights_h, weights_o = 1.25, 2.0, 1.75, 0.5  
            else:  
                weights_l, weights_c, weights_h, weights_o = 1.75, 2.0, 1.25, 0.5  
            weight_sum = weights_l + weights_c + weights_h + weights_o  

            ohlc5 = (sub_low * weights_l + sub_close * weights_c + sub_high * weights_h + sub_open * weights_o) / weight_sum  

            twpoc_num = 0.0  
            twpoc_den = 0.0  
            length = len(sub_close)  

            for i in range(length):  
                w = decay ** i  
                price = ohlc5.iloc[length - 1 - i]  
                vol = sub_volume.iloc[length - 1 - i]  
                twpoc_num += price * vol * w  
                twpoc_den += vol * w  

            twpoc_values[idx] = twpoc_num / twpoc_den if twpoc_den > 0 else np.nan  

        return pd.Series(twpoc_values, index=coin_date_df.index)   
    
    def twpoc_calc_with_lambda_for_HFrame(self, coin_date_df):  
        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  
        volume = coin_date_df.iloc[:, 5]  

        decay = np.exp(-self.lambd)  
        n = len(coin_date_df)  
        window = self.window_HFrame  

        twpoc_values = np.full(n, np.nan)  

        for idx in range(n):  
            start_idx = max(0, idx - window + 1)  
            sub_open = open_.iloc[start_idx:idx + 1]  
            sub_high = high.iloc[start_idx:idx + 1]  
            sub_low = low.iloc[start_idx:idx + 1]  
            sub_close = close.iloc[start_idx:idx + 1]  
            sub_volume = volume.iloc[start_idx:idx + 1]  

            origin_HFrame_vpPOC = np.average(sub_close, weights=sub_volume) if sub_volume.sum() > 0 else np.nan  

            if np.isnan(origin_HFrame_vpPOC):  
                twpoc_values[idx] = np.nan  
                continue  

            # 直接区分上涨下跌决定加权参数  
            if sub_close.iloc[-1] >= origin_HFrame_vpPOC:  
                weights_l, weights_c, weights_h, weights_o = 1.25, 2.0, 1.75, 0.5  
            else:  
                weights_l, weights_c, weights_h, weights_o = 1.75, 2.0, 1.25, 0.5  
            weight_sum = weights_l + weights_c + weights_h + weights_o  

            ohlc5 = (sub_low * weights_l + sub_close * weights_c + sub_high * weights_h + sub_open * weights_o) / weight_sum  

            twpoc_num = 0.0  
            twpoc_den = 0.0  
            length = len(sub_close)  

            for i in range(length):  
                w = decay ** i  
                price = ohlc5.iloc[length - 1 - i]  
                vol = sub_volume.iloc[length - 1 - i]  
                twpoc_num += price * vol * w  
                twpoc_den += vol * w  

            twpoc_values[idx] = twpoc_num / twpoc_den if twpoc_den > 0 else np.nan  

        return pd.Series(twpoc_values, index=coin_date_df.index)  
    
    def calculate_HFrame_vpPOC_and_std(self, coin_date_df):  
        self.LFrame_vpPOC_series = self.twpoc_calc_with_lambda_for_LFrame(coin_date_df)  

        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  

        weights_l, weights_c, weights_h, weights_o = 1.5, 2, 1.5, 0.5  
        weight_sum = weights_l + weights_c + weights_h + weights_o  
        ohlc5_values = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
        self.LFrame_ohlc5_series = pd.Series(ohlc5_values.values, index=coin_date_df.index)  

        self.LFrame_rolling_std = self.LFrame_vpPOC_series.rolling(window=self.std_window_LFrame, min_periods=1).std()  
        self.LFrame_rolling_std.index = coin_date_df.index  

        self.LFrame_std_2_upper = self.LFrame_vpPOC_series + 2 * self.LFrame_rolling_std  
        self.LFrame_std_2_lower = self.LFrame_vpPOC_series - 2 * self.LFrame_rolling_std  
        self.LFrame_std_4_upper = self.LFrame_vpPOC_series + 4 * self.LFrame_rolling_std  
        self.LFrame_std_4_lower = self.LFrame_vpPOC_series - 4 * self.LFrame_rolling_std  

        self.HFrame_vpPOC = self.twpoc_calc_with_lambda_for_HFrame(coin_date_df)  
        self.HFrame_ohlc5_series = self.LFrame_ohlc5_series  

        self.HFrame_price_std = self.HFrame_ohlc5_series.rolling(window=self.window_HFrame, min_periods=1).std()  
        self.HFrame_price_std.index = coin_date_df.index  

        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]  
        for m in multipliers:  
            upper = self.HFrame_vpPOC + m * self.HFrame_price_std  
            lower = self.HFrame_vpPOC - m * self.HFrame_price_std  
            setattr(self, f'HFrame_std_{str(m).replace(".", "_")}_up', upper)  
            setattr(self, f'HFrame_std_{str(m).replace(".", "_")}_down', lower)  

    def rsi_with_ema_smoothing(self, coin_date_df, length=13):  
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
        'LFrame_std_4_upper': 'lightblue',
        'LFrame_std_4_lower': 'lightblue',
        'HFrame_vpPOC': 'purple',
        'HFrame_ohlc5_series': 'orange',
    }

    # 绘制LFrame线
    for var in [
        'LFrame_ohlc5_series',
        'LFrame_std_2_upper', 'LFrame_std_2_lower',
        'LFrame_std_4_upper', 'LFrame_std_4_lower',
    ]:
        val = getattr(multFramevpPOC, var, None)
        if val is not None and hasattr(val, 'index') and hasattr(val, 'values'):
            ax.plot(val.index, val.values, label=var, color=colors.get(var, 'white'), linewidth=1)

    lframe_vp = getattr(multFramevpPOC, 'LFrame_vpPOC_series', None)
    if lframe_vp is not None and hasattr(lframe_vp, 'index') and hasattr(lframe_vp, 'values'):
        ax.plot(lframe_vp.index, lframe_vp.values, label='LFrame vpPOC', color=colors['LFrame_vpPOC_series'], linewidth=2)

    # 绘制HFrame vpPOC
    hframe_vp = getattr(multFramevpPOC, 'HFrame_vpPOC', None)
    if hframe_vp is not None and hasattr(hframe_vp, 'index') and hasattr(hframe_vp, 'values'):
        ax.plot(hframe_vp.index, hframe_vp.values, label='HFrame vpPOC', color=colors['HFrame_vpPOC'], linewidth=3)

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
        'LFrame_std_4_upper', 'LFrame_std_4_lower',
        'LFrame_vpPOC_series',
        'HFrame_vpPOC',
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
