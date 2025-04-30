# factors.py  
import pandas as pd  
import numpy as np  
import ta  
from typing import Dict, List, Optional, Union  
from dataclasses import dataclass  
from scipy.stats import linregress  
from sklearn.linear_model import LinearRegression  
from datetime import datetime
from .improved_psar import ImprovedPSARIndicator
from scipy.stats import shapiro, normaltest

@dataclass  
class FactorConfig:  
    """因子配置类"""  
    # 基础配置  
    window_size: int = 15        # 基础窗口大小  
    tau:float = 0.03
    abnormal_flags = None

    smoothing: int = 3          # 平滑参数  
    std_window: int = 20        # 标准化窗口  
    
    # MA系列配置  
    ma_windows: List[int] = (5, 10, 20, 60)  
    
    # MACD配置  
    macd_fast: int = 12  
    macd_slow: int = 26  
    macd_signal: int = 9  
    
    # Bollinger Bands配置  
    bb_window: int = 20  
    bb_std: float = 2.0  
    
    # SAR配置  
    sar_acceleration: float = 0.02  
    sar_maximum: float = 0.2  
    
    # KDJ配置  
    kdj_window: int = 9  
    kdj_smooth_window: int = 3  
    
    # 波动率配置  
    volatility_window: int = 10  
    
    def __post_init__(self):  
        if isinstance(self.ma_windows, tuple):  
            self.ma_windows = list(self.ma_windows)  

class BaseFactors:  
    """因子基类"""  
    def __init__(self, config: FactorConfig=None):  
        self.config = config  
    
    def normalize_factor(self, series: pd.Series) -> pd.Series:  
        """标准化因子"""  
        return (series - series.rolling(self.config.std_window).mean()) / \
               (series.rolling(self.config.std_window).std() + 1e-8)  
    
    def smooth_factor(self, series: pd.Series) -> pd.Series:  
        """平滑处理"""  
        return series.ewm(span=self.config.smoothing, adjust=False).mean()      
    
class TrendFactors(BaseFactors):  
    """趋势类因子"""  
    def macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """MACD指标"""  
        macd_ind = ta.trend.MACD(  
            df['close'],
            # df['twvwap'] , 
            window_slow=self.config.macd_slow,  
            window_fast=self.config.macd_fast,  
            window_sign=self.config.macd_signal  
        )  
        
        return {  
            'macd': macd_ind.macd(),  
            'macd_signal': macd_ind.macd_signal(),  
            'macd_diff': macd_ind.macd_diff(),  
        }  
    
    def adx(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算平均趋向指标 (Average Directional Index)"""  
        try:  
            # 确保有足够的数据来计算 ADX  
            if len(df) < self.config.window_size * 2:  
                # 如果数据不足，返回空的序列  
                empty_series = pd.Series(np.nan, index=df.index)  
                return {  
                    'adx': empty_series,  
                    '+di': empty_series,  
                    '-di': empty_series  
                }  
            
            window = self.config.window_size  # 默认窗口，例如 14
            # if self.config.abnormal_flags.any():
            #     # 异常时缩短窗口，增强近期敏感性
            #     window = max(int(window / 2), 3)  # 至少 3，避免过短

            adx_ind = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=window
            ) 
            
            return {  
                'adx': adx_ind.adx(),  
                '+di': adx_ind.adx_pos(),  
                '-di': adx_ind.adx_neg()  
            }  
        except Exception as e:  
            print(f"计算ADX失败: {e}")  
            empty_series = pd.Series(np.nan, index=df.index)  
            return {  
                'adx': empty_series,  
                '+di': empty_series,  
                '-di': empty_series  
            }
    
class MomentumFactors(BaseFactors):  
    """动量类因子"""  
    
    def stochastic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """随机指标"""  
        stoch = ta.momentum.StochasticOscillator(  
            df['high'],  
            df['low'],  
            df['close'],  
            window=self.config.window_size,  
            smooth_window=self.config.smoothing  
        )  
        
        return {  
            'stoch_k': stoch.stoch(),  
            'stoch_d': stoch.stoch_signal(),  
        }  
    
    def roc(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """变动率指标"""  
        roc = ta.momentum.ROCIndicator(  
            df['close'],  
            window=self.config.window_size  
        ).roc()  
        
        return {  
            'roc': roc,  
        }  
    
    def williams_r(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """威廉指标"""  
        wr = ta.momentum.WilliamsRIndicator(  
            df['high'],  
            df['low'],  
            df['close'],  
            lbp=self.config.window_size  # loop back period
        ).williams_r()  
        
        return {  
            'wr': wr,  
        }  
    
def weighted_atr(atr_hist:List):
    def compute_weighted_atr(atr_hist, N, lambda_decay):  
        """  
        计算大周期ATR（加权加衰减）  
        atr_hist: ATR数据列表，按时间顺序排列（最老在前，最新在后）  
        N: 大周期窗口长度  
        lambda_decay: 衰减因子，例如 2/(N+1)  
        """  
        # 考虑最近N个周期数据  
        relevant_atr = np.array(atr_hist[-N:])  
        # 新的数据权重更高，因此权重按时间升序  
        weights = np.exp(-lambda_decay * np.arange(N-1, -1, -1))  
        return np.sum(relevant_atr * weights) / np.sum(weights)  

    def compute_simple_atr(atr_hist, M):  
        """  
        计算小周期ATR（简单均值）  
        M: 小周期窗口长度  
        """  
        return np.mean(atr_hist[-M:])  

    def adaptive_alpha(atr_large, atr_small, threshold_low=0.1, threshold_high=0.3):  
        """  
        根据大周期和小周期ATR的相对差异计算自适应权重  
        """  
        relative_diff = abs(atr_small - atr_large) / atr_large  
        if relative_diff <= threshold_low:  
            return 0.4  
        elif relative_diff >= threshold_high:  
            return 0.8  
        else:  
            # 线性插值，0.4至0.8之间过渡  
            return 0.4 + ((relative_diff - threshold_low) / (threshold_high - threshold_low)) * 0.4  

    def compute_combined_atr(atr_hist, N=20, M=5):  
        # 计算λ，建议取2/(N+1)  
        lambda_decay = 2 / (N + 1)  
        atr_large = compute_weighted_atr(atr_hist, N, lambda_decay)  
        atr_small = compute_simple_atr(atr_hist, M)  
        a = adaptive_alpha(atr_large, atr_small)  
        combined_atr = a * atr_large + (1 - a) * atr_small  
        return combined_atr, atr_large, atr_small, a  
    
    return compute_combined_atr(atr_hist=atr_hist, N=60, M=14)

class VolatilityFactors(BaseFactors):  
    """波动率类因子"""  
    
    def atr(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """平均真实范围"""  
        atr = ta.volatility.AverageTrueRange(  
            df['high'],  
            df['low'],  
            df['close'],  
            window=self.config.window_size  
        )  
        
        atr_value = atr.average_true_range()  
        return {  
            'atr': atr_value,   
            'atr_pct': atr_value / df['close'] * 100  # ATR百分比  
        }  
    
    def keltner(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """肯特纳通道"""  
        kc = ta.volatility.KeltnerChannel(  
            df['high'],  
            df['low'],  
            df['close'],  
            window=self.config.window_size  
        )  
        
        return {  
            'kc_high': kc.keltner_channel_hband(),  
            'kc_mid': kc.keltner_channel_mband(),  
            'kc_low': kc.keltner_channel_lband(),  
            'kc_width': kc.keltner_channel_wband(), 
        }  
    
    def volatility(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """波动率指标"""  
        # 计算对数收益率  
        log_returns = np.log(df['close'] / df['close'].shift(1))  
        
        # 计算历史波动率  
        hist_vol = log_returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)  
        
        # 计算Parkinson波动率  
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) *   
                              (np.log(df['high'] / df['low']) ** 2).rolling(window=self.config.volatility_window).mean() *   
                              np.sqrt(252))  
        
        return {  
            'hist_vol': hist_vol,  
            'parkinson_vol': parkinson_vol,  
        }  
    
    def price_range(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """价格范围"""  
        daily_range = (df['high'] - df['low']) / df['close'] * 100  
        
        return {  
            'daily_range': daily_range,  
        }

class AdaptiveTVWAP:  
    def __init__(self, base_tau=0.03):  
        self.base_tau = base_tau  
        
    def calculate(self, df, abnormal_flags):  
        """  
        带异常量能加权的VWAP计算  
        输入：OHLCV数据框，异常量能标记  
        输出：自适应VWAP序列  
        """  
        weights = np.ones(len(df))  
        assert len(df) == len(abnormal_flags), f'abnormal_flags{len(abnormal_flags)}not equal to {len(df)}'
        decay_factor_normal = np.exp(-self.base_tau)
        decay_factor_abnormal = np.exp(-self.base_tau * 2)

        # Generate decay factors for each position based on abnormal_flags
        decay_factors = np.where(abnormal_flags, decay_factor_abnormal, decay_factor_normal)

        # Apply cumulative decay to weights array
        for i in range(1, len(df)):
            weights[:i] *= decay_factors[i-1]
                
        twvwap = (df['low'] * df['volume'] * weights).cumsum() / (df['volume'] * weights).cumsum()  
        return twvwap

class TimeSplitedTVWAP:
    def __init__(self, base_tau=0.03, session_start_time='00:00:00', timezone='UTC', num_std_devs=2):
        """
        带交易日重置、标准差带和异常量能加权的 TVWAP 计算 (矩阵优化版)

        参数:
            base_tau (float):  基础衰减系数.
            session_start_time (str): 交易日开始时间 (HH:MM:SS 格式). 默认 '00:00:00' (UTC).
            timezone (str): 交易日开始时间的时区.  默认 'UTC'.  可以是 'US/Eastern', 'America/New_York' 等.
            num_std_devs (int): 要计算的标准差带数量 (例如, 1 表示 +/- 1 个标准差).
        """
        self.base_tau = base_tau
        self.session_start_time = session_start_time
        self.timezone = timezone
        self.num_std_devs = num_std_devs
        self.reset_values()  # 初始化累积变量

    def reset_values(self):
        """重置 VWAP 和标准差计算的累积变量"""
        self.cumulative_pv = 0  # 累积 Price * Volume
        self.cumulative_volume = 0
        self.cumulative_price_deviation_volume_sq = 0
        self.cumulative_volume_sq = 0

    def _is_new_session(self, timestamp):
        return False
        """判断是否是新的交易日"""
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            timestamp = timestamp.tz_localize(self.timezone)
        else:
            timestamp = timestamp.tz_convert(self.timezone)

        session_start_dt = datetime.datetime.strptime(self.session_start_time, '%H:%M:%S').time()
        current_time = timestamp.time()
        return current_time >= session_start_dt and current_time < (
                datetime.datetime.combine(datetime.date.min, session_start_dt) + datetime.timedelta(
            seconds=1)).time()

    def detect_reversals(self, df, window=5, volume_multiplier=1.3):  
        """  
        检测局部极值作为反转候选信号，同时要求连续三根K线成交量显著放大。  
        
        条件说明：  
        1. 在窗口内，检测当前收盘价是否为局部最高或最低，可作为反转候选信号。  
        2. 同时要求当前位置及其前两根K线，成交量均大于各自前半个窗口内平均成交量 volume_multiplier 倍，  
        则认为该反转信号更为有效。  
        
        返回一个表示反转点索引的集合。  
        """  
        import numpy as np  

        reversal_indices = set()  
        close_values = df['close'].values  
        volume_values = df['volume'].values  
        n = len(close_values)  

        # 从 window+2 开始，确保有连续三根K线  
        for i in range(window + 2, n - window):  
            # 判断当前K线是否为局部极值  
            window_slice = close_values[i - window: i + window + 1]  
            if close_values[i] == max(window_slice) or close_values[i] == min(window_slice):  
                # 检查连续三根K线成交量是否放大  
                volume_condition_met = True  
                for j in range(i - 2, i + 1):  
                    # 使用前半个窗口的成交量均值作为基准  
                    volume_window = volume_values[j - window: j]  
                    if len(volume_window) == 0:  
                        volume_condition_met = False  
                        break  
                    avg_volume = np.mean(volume_window)  
                    if not (avg_volume > 0 and volume_values[j] >= volume_multiplier * avg_volume):  
                        volume_condition_met = False  
                        break  
                if volume_condition_met:  
                    reversal_indices.add(i)  
        return reversal_indices
    
    #检测到反转后等下根K线再重置计算起点。（因子没有信号滞后）
    def calculate(self, df, abnormal_flags, window_size=14, reversal_window=5):  
        """  
        带交易日重置、标准差带和异常量能加权的TVWAP计算（矩阵优化版）。  

        额外增加了反转检测逻辑：  
        - 当在df中检测到局部反转时，不立即回溯，而是在下一根K线（窗口的第二个位置）  
            将TWVWAP等指标重新初始化，从新锚点开始（首根K线标准差为0）。  

        参数:  
        df: 包含 'timestamp', 'close', 'volume', 'open', 'low', 'high' 列的DataFrame。  
            'timestamp'列必须是datetime类型，建议带时区信息。  
        abnormal_flags: 布尔型Series，标记异常成交量，需和df的index一致。  
        window_size: 滑动窗口大小，默认为14。  
        reversal_window: 用于反转检测的窗口大小，默认为5。  

        返回:  
        dict: 包含 'twvwap', 'twvwap_std_dev', 'twvwap_upper_band_1', 'twvwap_lower_band_1',  
                'twvwap_upper_band_2', 'twvwap_lower_band_2', 以及其他相关指标。  
        """  
        from collections import deque  
        import numpy as np  
        import pandas as pd  
        import scipy.stats as stats  

        df_len = len(df)  

        # 计算全局衰减权重（仅作参考，下文转换为相对权重）  
        weights = np.ones(df_len)  
        decay_factor_normal = np.exp(-self.base_tau)  
        decay_factor_abnormal = np.exp(-self.base_tau * 2)  
        decay_factors = np.where(abnormal_flags.values, decay_factor_abnormal, decay_factor_normal)  
        for i in range(1, df_len):  
            weights[:i] *= decay_factors[i-1]  

        # 预先检测反转点  
        reversal_indices = self.detect_reversals(df, window=reversal_window, volume_multiplier=1.5)  

        # 初始化结果数组  
        twvwap_values = np.full(df_len, np.nan)  
        std_dev_values = np.full(df_len, np.nan)  
        upper_band_10_values = np.full(df_len, np.nan)  
        lower_band_10_values = np.full(df_len, np.nan)  
        upper_band_20_values = np.full(df_len, np.nan)  
        lower_band_20_values = np.full(df_len, np.nan)  
        upper_band_05_values = np.full(df_len, np.nan)  
        lower_band_05_values = np.full(df_len, np.nan)  
        upper_band_15_values = np.full(df_len, np.nan)  
        lower_band_15_values = np.full(df_len, np.nan)  
        upper_band_40_values = np.full(df_len, np.nan)  
        lower_band_40_values = np.full(df_len, np.nan)  

        # 初始化滑动窗口队列  
        pv_queue = deque(maxlen=window_size)  
        volume_queue = deque(maxlen=window_size)  
        price_dev_sq_queue = deque(maxlen=window_size)  
        volume_sq_queue = deque(maxlen=window_size)  

        # cur_base用于在每个累计段中换算相对权重，确保新起点effective_weight为1  
        cur_base = None  

        # 获取numpy数组  
        close_values = df['close'].values  
        # open_values = df['open'].values  
        low_values = df['low'].values  
        # high_values = df['high'].values  
        volume_values = df['volume'].values  
        timestamp_values = df.index.values  # 假设索引为timestamp  

        # 定义各带宽因子  
        f05, f10, f15, f20, f40 = 0.5, 1, 1.5, 2, 4  

        # 辅助函数：更新队列后计算TWVWAP、标准差以及各轨带  
        def update_metrics(price, volume, effective_weight, close_price):  
            pv_queue.append(price * volume * effective_weight)  
            volume_queue.append(volume * effective_weight)  
            cumulative_pv = sum(pv_queue)  
            cumulative_volume = sum(volume_queue)  
            if cumulative_volume > 0:  
                twvwap = cumulative_pv / cumulative_volume  
                price_deviation = close_price - twvwap  
                price_dev_sq_queue.append((price_deviation * volume * effective_weight) ** 2)  
                volume_sq_queue.append((volume * effective_weight) ** 2)  
                cumulative_price_dev_sq = sum(price_dev_sq_queue)  
                cumulative_volume_sq = sum(volume_sq_queue)  
                std_dev = np.sqrt(cumulative_price_dev_sq / cumulative_volume_sq) if cumulative_volume_sq > 0 else 0  

                upper_band_05 = twvwap + f05 * std_dev   
                lower_band_05 = twvwap - f05 * std_dev   
                upper_band_10 = twvwap + f10 * std_dev  
                lower_band_10 = twvwap - f10 * std_dev  
                upper_band_15 = twvwap + f15 * std_dev  
                lower_band_15 = twvwap - f15 * std_dev  
                upper_band_20 = twvwap + f20 * std_dev  
                lower_band_20 = twvwap - f20 * std_dev  
                upper_band_40 = twvwap + f40 * std_dev  
                lower_band_40 = twvwap - f40 * std_dev  
            else:  
                twvwap = std_dev = np.nan  
                upper_band_05 = lower_band_05 = upper_band_10 = lower_band_10 = np.nan  
                upper_band_15 = lower_band_15 = upper_band_20 = lower_band_20 = np.nan  
                upper_band_40 = lower_band_40 = np.nan  
            return (twvwap, std_dev,  
                    upper_band_05, lower_band_05,  
                    upper_band_10, lower_band_10,  
                    upper_band_15, lower_band_15,  
                    upper_band_20, lower_band_20,  
                    upper_band_40, lower_band_40)  

        anchor_points = []
        # 反转待重置标志  
        pending_reset = False  

        # 主循环：遍历每一个数据点  
        for i in range(df_len):  
            timestamp = timestamp_values[i]  
            # 计算中间均价  
            price = low_values[i]
            close_price = close_values[i]  
            volume = volume_values[i]  

            # 判断是否新的交易日  
            if self._is_new_session(timestamp):  
                pv_queue.clear()  
                volume_queue.clear()  
                price_dev_sq_queue.clear()  
                volume_sq_queue.clear()  
                pending_reset = False  
                cur_base = weights[i]  # 新交易日，更新累计段基准  

            # 根据不同情况选择逻辑：  
            if pending_reset:  
                # 若待重置标志有效则清空累计队列，并以当前数据作为新起点  
                pv_queue.clear()  
                volume_queue.clear()  
                price_dev_sq_queue.clear()  
                volume_sq_queue.clear()  
                cur_base = weights[i]  
                # effective_weight始终为1  
                effective_weight = weights[i] / cur_base  
                metrics = update_metrics(price, volume, effective_weight, close_price)  
                pending_reset = False  
            elif i in reversal_indices:  
                # 如果当前点标记到反转，则先设置待重置，同时累积当前数据  
                pending_reset = True  
                if cur_base is None:  
                    cur_base = weights[i]  
                effective_weight = weights[i] / cur_base  
                metrics = update_metrics(price, volume, effective_weight, close_price)  
            else:  
                # 正常累积逻辑  
                if cur_base is None:  
                    cur_base = weights[i]  
                effective_weight = weights[i] / cur_base  
                metrics = update_metrics(price, volume, effective_weight, close_price)  
            
            anchor_points.append(pending_reset)
            # metrics中包含 (twvwap, std_dev, upper_band_05, lower_band_05,   
            # upper_band_10, lower_band_10, upper_band_15, lower_band_15,  
            # upper_band_20, lower_band_20, upper_band_40, lower_band_40)  
            (twvwap, std_dev,  
            upper_band_05, lower_band_05,  
            upper_band_10, lower_band_10,  
            upper_band_15, lower_band_15,  
            upper_band_20, lower_band_20,  
            upper_band_40, lower_band_40) = metrics  

            twvwap_values[i] = twvwap  
            std_dev_values[i] = std_dev  
            upper_band_05_values[i] = upper_band_05  
            lower_band_05_values[i] = lower_band_05  
            upper_band_10_values[i] = upper_band_10  
            lower_band_10_values[i] = lower_band_10  
            upper_band_15_values[i] = upper_band_15  
            lower_band_15_values[i] = lower_band_15  
            upper_band_20_values[i] = upper_band_20  
            lower_band_20_values[i] = lower_band_20  
            upper_band_40_values[i] = upper_band_40  
            lower_band_40_values[i] = lower_band_40  

        
        return {  
            'twvwap': pd.Series(twvwap_values, index=df.index),  
            'twvwap_std_dev': pd.Series(std_dev_values, index=df.index),  
            'twvwap_anchor_points':pd.Series(anchor_points, index=df.index),
            'twvwap_upper_band_1': pd.Series(upper_band_10_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_lower_band_1': pd.Series(lower_band_10_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_upper_band_2': pd.Series(upper_band_20_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_lower_band_2': pd.Series(lower_band_20_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_upper_band_05': pd.Series(upper_band_05_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_lower_band_05': pd.Series(lower_band_05_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_upper_band_15': pd.Series(upper_band_15_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_lower_band_15': pd.Series(lower_band_15_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_upper_band_40': pd.Series(upper_band_40_values, index=df.index).rolling(window=4).mean(),  
            'twvwap_lower_band_40': pd.Series(lower_band_40_values, index=df.index).rolling(window=4).mean(),  
        } 
    
class VolumeFactors(BaseFactors):  
    """成交量类因子"""  
    def obv(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """能量潮指标(On Balance Volume)"""  
        obv = ta.volume.OnBalanceVolumeIndicator(  
            df['close'],  
            df['volume']  
        ).on_balance_volume()  
        
        obv_ma = obv.rolling(window=self.config.window_size).mean()  
        obv_slope = pd.Series(index=df.index)  
        
        # 计算OBV斜率  
        for i in range(self.config.window_size, len(df)):  
            obv_slope.iloc[i] = linregress(  
                range(self.config.window_size),  
                obv[i-self.config.window_size+1:i+1]  
            )[0]  
        
        return {  
            'obv': obv,  
            'obv_ma': obv_ma,  
            'obv_slope': obv_slope,  
        }  

    def vwap(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """  
        计算传统 VWAP 以及基于价格反转与成交量异常调整的 AVWAP。  
        当连续检测到满足条件的数据点时，将视为价格反转的信号，从下一个点开始重置累计，  
        使得 AVWAP 能更敏感地捕捉突变的起止点。  

        参数:  
            df: 包含 'open', 'high', 'low', 'close', 'volume' 的 DataFrame  

        返回:  
            包含 'vwap' 和 'avwap' 的字典  
        """  
        epsilon = 1e-8  # 防止除零  
        
        # 计算典型价格  
        typical_price = df['low']
        
        # ---------- A. 传统 VWAP 计算（全量累计） ----------  
        cum_volume = df['volume'].cumsum()  
        vwap = (typical_price * df['volume']).cumsum() / (cum_volume + epsilon)  
        
        # ---------- B. AVWAP 计算：重置累计在价格反转信号处 ----------  
        # 根据时间间隔自动调整参数  
        time_interval = df.index[1] - df.index[0]  
        if time_interval.days >= 1:  
            window = 7  
            factor = 1.5  
        elif time_interval.total_seconds() / 3600 == 1:  # hourly  
            window = 24  
            factor = 1.5  
        elif time_interval.total_seconds() / 60 == 5:  # 5分钟  
            window = 12  
            factor = 2.0  
        else:  
            window = 14  
            factor = 1.5  

        # 滚动计算成交量均值  
        rolling_mean_volume = df['volume'].rolling(window=window, min_periods=1).mean()  
        # 计算成交量比率，衡量异常成交量程度  
        volume_ratio = df['volume'] / (rolling_mean_volume + epsilon)  
        # 当 volume_ratio 超过设定阈值 factor 时，超出部分即视为成交量异常信号  
        excess = np.maximum(volume_ratio - factor, 0)  
        
        # 利用滚动窗口同时计算均值和中位数，以更平滑地检测价格反转  
        rolling_price = typical_price.rolling(window=window, min_periods=1)  
        rolling_mean_price = rolling_price.mean()  
        rolling_median_price = rolling_price.median()  
        # 折中使用均值和中位数：能兼顾回归性和鲁棒性  
        blended_price = (rolling_mean_price + rolling_median_price) / 2.0  
        
        # 计算价格偏离（正值表示与反转方向相反的偏离程度）  
        reversal_strength = np.maximum(blended_price - typical_price, 0)  
        # 用偏离比例构建价格影响因子  
        reversal_factor = 1 + (reversal_strength / (blended_price + epsilon))  
        
        # 计算调整后的成交量：当同时存在较大成交量异常与价格反转时，放大成交量权重  
        adjusted_volume = df['volume'] * (1 + excess * (reversal_factor - 1))  
        
        # 连续性检测参数  
        required_consecutive = 5     # 连续达到条件的最小数据点数量  
        excess_threshold = 0.05      # 判断成交量异常的下限门槛  
        reversal_threshold = 0.01    # 判断价格反转的下限门槛  
        
        avwap_list = []  # 存储 AVWAP 的计算结果  
        cum_weighted_price = 0.0  
        cum_adjusted_volume = 0.0  
        consecutive_count = 0  # 连续满足异常条件的计数  

        # 逐点累积计算 AVWAP，同时检查是否需要重置累计  
        for idx in range(len(df)):  
            cur_excess = excess.iloc[idx]  
            cur_reversal = reversal_strength.iloc[idx]  
            # 判断是否同时满足异常成交量和价格反转条件  
            if cur_excess > excess_threshold and cur_reversal > reversal_threshold:  
                consecutive_count += 1  
            else:  
                consecutive_count = 0  

            # 计算当前点的贡献：典型价格 * 调整后的成交量  
            cur_price = typical_price.iloc[idx]  
            cur_adjusted_volume = adjusted_volume.iloc[idx]  
            cum_weighted_price += cur_price * cur_adjusted_volume  
            cum_adjusted_volume += cur_adjusted_volume  

            # 计算当前 AVWAP：避免除零  
            current_avwap = cum_weighted_price / (cum_adjusted_volume + epsilon)  
            avwap_list.append(current_avwap)  

            # 若连续异常达到预设阈值，认为当前点为价格反转点，  
            # 则在下个点重置累计，使 AVWAP 作为新起点计算。  
            if consecutive_count >= required_consecutive:  
                # 为减少频繁重置，可以选择重置后将累计值设为当前点数据的初始值  
                cum_weighted_price = cur_price * cur_adjusted_volume  
                cum_adjusted_volume = cur_adjusted_volume  
                consecutive_count = 0  # 重置计数器  

        avwap = pd.Series(avwap_list, index=df.index)  

        return {  
            'vwap': vwap,  
            'avwap': avwap,  
        }  
    
    #time_weighted_vwap
    def twvwap(self, df: pd.DataFrame) -> dict:  
        """  
        优化版时间加权VWAP计算：  
        1. 显式通过列名访问数据  
        2. 使用向量化计算提升性能  
        """  

        # 寻找最优参数组合  
        best_tau, best_window = self.config.tau, self.config.window_size  

        # 检测异常量能  
        abnormal_flags = self.config.abnormal_flags
 
        assert len(df) == len(self.config.abnormal_flags), f"twvwap() df length: {len(df)}, abnormal_flags length: {len(abnormal_flags)}"
        twvwap_dict = TimeSplitedTVWAP(best_tau).calculate(df, abnormal_flags, best_window*10, reversal_window=5)#reversal_window=best_window) 
        twvwap_series = twvwap_dict['twvwap']
        # twvwap_series = AdaptiveTVWAP(best_tau).calculate(df, abnormal_flags) 
        t = twvwap_series.rolling(window=best_window*10)
        twvwap_smooth = (t.mean() + t.median())/2
        # twvwap_price_deviation = (df['close'] - twvwap_series) / twvwap_series * 100  
        twvwap_deviation = (twvwap_series - twvwap_smooth) / twvwap_smooth * 100

        # 计算TVWAP斜率 M（这里简单用一阶差分，实际可根据需求换成合适算法）  
        twvwap_slope = twvwap_series.diff(best_window)  
        # df['twvwap'] = twvwap_series
        ret_dict = {  
            'abnormal_flags':abnormal_flags,
            'twvwap': twvwap_series,  
            # 'twvwap_price_deviation': twvwap_price_deviation,  
            'twvwap_deviation':twvwap_deviation,
            'twvwap_smooth':twvwap_smooth,
            'twvwap_slope': twvwap_slope, 
            # "twvwap_rolling_std":twvwap_series.rolling(window=best_window*3).std() 
        }  
        ret_dict.update(twvwap_dict)
        return ret_dict

    
    def volume_profile(self, df: pd.DataFrame, n_bins: int = 10) -> Dict[str, pd.Series]:  
        """成交量分布"""  
        price_range = df['high'].max() - df['low'].min()  
        bin_size = price_range / n_bins  
        
        volume_profile = pd.Series(0.0, index=range(n_bins))  
        price_points = np.arange(df['low'].min(), df['high'].max(), bin_size)  
        
        for i in range(len(df)):  
            bin_idx = int((df['close'].iloc[i] - df['low'].min()) / bin_size)  
            if bin_idx >= n_bins:  
                bin_idx = n_bins - 1  
            volume_profile[bin_idx] += df['volume'].iloc[i]  
        
        # 计算POC (Point of Control)  
        poc_idx = volume_profile.argmax()  
        poc_price = df['low'].min() + poc_idx * bin_size  
        
        return {  
            'volume_profile': volume_profile,  
            'poc_price': poc_price,  
            'volume_distribution': volume_profile / volume_profile.sum()  
        }  
    
    def calculate_vpvr(self, df: pd.DataFrame, bins: int = 100) -> dict:  
        """  
        计算 VPVR 因子并返回字典，分开买入和卖出成交量  
        :param df: 数据框，必须包含 'open'、'close' 和 'volume' 列  
        :param bins: 价格区间数量  
        :return: VPVR 数据字典，包含买入和卖出成交量  
        """  
        price_col = 'close'  
        volume_col = 'volume'  
        open_price_col = 'open'  # 假设有开盘价列  

        price_min = df[price_col].min()  
        price_max = df[price_col].max()  
        
        # 创建价格区间  
        price_bins = np.linspace(price_min, price_max, bins)  
        buy_vpvr = np.zeros(bins - 1)  
        sell_vpvr = np.zeros(bins - 1)  
        
        # 计算 VPVR  
        for i in range(len(df)):  
            price = df[price_col].iloc[i]  
            volume = df[volume_col].iloc[i]  
            open_price = df[open_price_col].iloc[i]  
            
            # 根据开盘价和收盘价的关系判断买入或卖出  
            if price > open_price:  
                side = 'buy'  
            elif price < open_price:  
                side = 'sell'  
            else:  
                continue  # 收盘价与开盘价相等时跳过  
            
            # 计算所处区间  
            bin_index = np.digitize(price, price_bins) - 1  
            if 0 <= bin_index < len(buy_vpvr):  
                if side == 'buy':  
                    buy_vpvr[bin_index] += volume  
                elif side == 'sell':  
                    sell_vpvr[bin_index] += volume  
        
        # 计算总成交量  
        total_volume = buy_vpvr + sell_vpvr  
        
        # 判断成交量是否符合正态分布  
        stat, p_value = normaltest(total_volume)  
        is_normal_distribution = p_value > 0.06  # p 值大于 0.05 表示不拒绝正态分布假设  

        # 返回 VPVR 数据字典  
        vpvr_dict = {  
            'Price': price_bins[:-1],  
            'Buy_Volume': buy_vpvr,  
            'Sell_Volume': sell_vpvr,  
            "is_normal_distribution":is_normal_distribution,
            'Total_Volume': buy_vpvr + sell_vpvr  # 总成交量  
        }  
        return vpvr_dict  

    def generate_fixed_vp(self, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:  
        """  
        计算成交量分布，并生成固定 VP 数据，用于绘图。  
        返回一个 DataFrame，其中包含 price_low, price_high, volume 三列。  
        """  
        price_min = df['low'].min()  
        price_max = df['high'].max()  
        price_range = price_max - price_min  
        bin_size = price_range / n_bins  

        # 初始化成交量分布  
        volume_profile = pd.Series(0.0, index=range(n_bins))  
        bins = []  
        for i in range(n_bins):  
            bins.append((price_min + i * bin_size, price_min + (i + 1) * bin_size))  

        # 累加每根K线的成交量到对应价格bin中  
        for i in range(len(df)):  
            close_price = df['close'].iloc[i]  
            bin_idx = int((close_price - price_min) / bin_size)  
            if bin_idx >= n_bins:  
                bin_idx = n_bins - 1  
            volume_profile.iloc[bin_idx] += df['volume'].iloc[i]  

        vp_df = pd.DataFrame(bins, columns=['price_low', 'price_high'])  
        vp_df['volume'] = volume_profile.values  

        return vp_df  

    def cmf(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """蜡烛图资金流指标(Chaikin Money Flow)"""  
        cmf = ta.volume.ChaikinMoneyFlowIndicator(  
            df['high'],  
            df['low'],  
            df['close'],  
            df['volume'],  
            window=self.config.window_size  
        ).chaikin_money_flow()  
        
        return {  
            'cmf': cmf,  
        }  
    
    def force_index(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """强力指数"""  
        fi = ta.volume.ForceIndexIndicator(  
            df['close'],  
            df['volume'],  
            window=self.config.window_size  
        ).force_index()  
        
        return {  
            'force_index': fi,  
        }  
    def pvt(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算价量趋势指标 (Price Volume Trend)"""  
        try:  
            # 计算价格变化百分比  
            price_change_pct = df['close'].pct_change()  
            
            # 计算 PVT  
            pvt = (price_change_pct * df['volume']).cumsum()  
            
            return {'pvt': pvt}  
        except Exception as e:  
            print(f"计算PVT失败: {e}")  
            return {'pvt': pd.Series(np.nan, index=df.index)}  

class PriceStructureFactors(BaseFactors):  
    """价格结构类因子"""  
    def pivot_points(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """轴心点位"""  
        # 计算基准轴心点  
        pivot = (df['high'] + df['low'] + df['close']) / 3  
        
        # 计算支撑和阻力位  
        r1 = 2 * pivot - df['low']  
        r2 = pivot + (df['high'] - df['low'])  
        r3 = r1 + (df['high'] - df['low'])  
        
        s1 = 2 * pivot - df['high']  
        s2 = pivot - (df['high'] - df['low'])  
        s3 = s1 - (df['high'] - df['low'])  
        
        return {  
            'pivot': pivot,  
            'r1': r1, 'r2': r2, 'r3': r3,  
            's1': s1, 's2': s2, 's3': s3,  
        }  
    
    def fractal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """分形指标"""  
        high_fractal = pd.Series(0, index=df.index)  
        low_fractal = pd.Series(0, index=df.index)  
        
        # 识别高点分形  
        for i in range(2, len(df)-2):  
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and   
                df['high'].iloc[i] > df['high'].iloc[i-2] and  
                df['high'].iloc[i] > df['high'].iloc[i+1] and  
                df['high'].iloc[i] > df['high'].iloc[i+2]):  
                high_fractal.iloc[i] = 1  
        
        # 识别低点分形  
        for i in range(2, len(df)-2):  
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and   
                df['low'].iloc[i] < df['low'].iloc[i-2] and  
                df['low'].iloc[i] < df['low'].iloc[i+1] and  
                df['low'].iloc[i] < df['low'].iloc[i+2]):  
                low_fractal.iloc[i] = 1  
        
        return {  
            'high_fractal': high_fractal,  
            'low_fractal': low_fractal  
        }  
    
    def zigzag(self, df: pd.DataFrame, deviation: float = 0.05) -> Dict[str, pd.Series]:  
        """之字形转折点"""  
        zigzag = pd.Series(0.0, index=df.index)  
        high_point = df['high'].iloc[0]  
        low_point = df['low'].iloc[0]  
        trend = 1  # 1: 上升, -1: 下降  
        
        for i in range(1, len(df)):  
            if trend == 1:  
                if df['high'].iloc[i] > high_point:  
                    high_point = df['high'].iloc[i]  
                elif df['low'].iloc[i] < (high_point * (1 - deviation)):  
                    zigzag.iloc[i] = high_point  
                    trend = -1  
                    low_point = df['low'].iloc[i]  
            else:  
                if df['low'].iloc[i] < low_point:  
                    low_point = df['low'].iloc[i]  
                elif df['high'].iloc[i] > (low_point * (1 + deviation)):  
                    zigzag.iloc[i] = low_point  
                    trend = 1  
                    high_point = df['high'].iloc[i]  
        
        return {  
            'zigzag': zigzag,  
        }  
    
    def support_resistance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """支撑阻力位"""  
        # 使用价格分布直方图识别支撑阻力位  
        hist, bins = np.histogram(df['close'], bins=50)  
        
        # 找出局部最大值作为潜在支撑阻力位  
        peaks = []  
        for i in range(1, len(hist)-1):  
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:  
                peaks.append((bins[i], hist[i]))  
        
        # 按成交量排序  
        peaks.sort(key=lambda x: x[1], reverse=True)  
        
        # 取前N个最重要的支撑阻力位  
        n_levels = min(5, len(peaks))  
        levels = [peak[0] for peak in peaks[:n_levels]]  
        
        return {  
            'support_resistance_levels': pd.Series(levels),  
            'price_distribution': pd.Series(hist, index=bins[:-1])  
        }
    

class MicroStructureFactors(BaseFactors):  
    """市场微观结构因子"""  
    def _has_level2_data(self, df: pd.DataFrame) -> bool:  
        """检查是否包含Level 2数据"""  
        required_columns = ['bid', 'ask', 'bid_size', 'ask_size']  
        return all(col in df.columns for col in required_columns)  

    def _has_trade_data(self, df: pd.DataFrame) -> bool:  
        """检查是否包含逐笔成交数据"""  
        required_columns = ['trade_direction', 'trade_size']  
        return all(col in df.columns for col in required_columns)  

    def bid_ask_spread(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """买卖价差"""  
        if not self._has_level2_data(df):  
            return {}  

        try:  
            spread = df['ask'] - df['bid']  
            spread_pct = spread / df['ask'] * 100  
            
            return {  
                'spread': spread,  
                'spread_pct': spread_pct,  
            }  
        except Exception as e:  
            print(f"计算买卖价差时出错: {e}")  
            return {}  

    def order_flow_imbalance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """订单流失衡"""  
        if not self._has_trade_data(df):  
            return {}  

        try:  
            buy_volume = df[df['trade_direction'] == 1]['volume'].rolling(  
                window=self.config.window_size).sum()  
            sell_volume = df[df['trade_direction'] == -1]['volume'].rolling(  
                window=self.config.window_size).sum()  
            
            total_volume = buy_volume + sell_volume  
            imbalance = (buy_volume - sell_volume) / total_volume.where(total_volume != 0)  
            
            return {  
                'order_imbalance': imbalance,  
            }  
        except Exception as e:  
            print(f"计算订单流失衡时出错: {e}")  
            return {}  

    def market_depth(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """市场深度"""  
        if not self._has_level2_data(df):  
            return {}  

        try:  
            depth = df['bid_size'] + df['ask_size']  
            depth_imbalance = (df['bid_size'] - df['ask_size']) / depth.where(depth != 0)  
            
            return {  
                'market_depth': depth,  
                'depth_imbalance': depth_imbalance,  
            }  
        except Exception as e:  
            print(f"计算市场深度时出错: {e}")  
            return {}  

    def trade_size_analysis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """成交规模分析"""  
        if not self._has_trade_data(df):  
            return {}  

        try:  
            avg_size = df['trade_size'].rolling(window=self.config.window_size).mean()  
            size_std = df['trade_size'].rolling(window=self.config.window_size).std()  
            
            large_trade_threshold = avg_size + 2 * size_std  
            large_trades = (df['trade_size'] > large_trade_threshold).astype(int)  
            
            return {  
                'avg_trade_size': avg_size,  
                'trade_size_std': size_std,  
                'large_trades': large_trades,  
            }  
        except Exception as e:  
            print(f"计算成交规模分析时出错: {e}")  
            return {}
    

class OrderBookFactors(BaseFactors):  
    """
    这个盘口因子类提供了以下主要功能：

    盘口失衡因子：

    基础失衡指标
    加权失衡指标（考虑价格优先级）
    价格冲击因子：

    不同数量级别的买入/卖出价格冲击
    标准化的冲击系数
    盘口斜率：

    买卖盘斜率
    斜率比率
    流动性指标：

    买卖盘深度
    相对流动性
    压力指标：

    买卖压力
    净压力和压力比
    """
        
    def __init__(self, config: FactorConfig):  
        super().__init__(config)  
        self.depth_levels = 10  # 设置需要分析的盘口深度  

    def _has_order_book_data(self, order_book: pd.DataFrame) -> bool:  
        """检查是否包含必要的盘口数据"""  
        required_prefixes = ['bid_price_', 'bid_size_', 'ask_price_', 'ask_size_']  
        for prefix in required_prefixes:  
            if not any(col.startswith(prefix) for col in order_book.columns):  
                return False  
        return True  

    def order_book_imbalance(self, order_book: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算盘口失衡因子"""  
        if not self._has_order_book_data(order_book):  
            return {}  

        results = {}  
        try:  
            # 计算各层的量价积累  
            bid_value = pd.Series(0.0, index=order_book.index)  
            ask_value = pd.Series(0.0, index=order_book.index)  

            for i in range(self.depth_levels):  
                if f'bid_size_{i}' in order_book.columns and f'bid_price_{i}' in order_book.columns:  
                    bid_value += order_book[f'bid_size_{i}'] * order_book[f'bid_price_{i}']  
                if f'ask_size_{i}' in order_book.columns and f'ask_price_{i}' in order_book.columns:  
                    ask_value += order_book[f'ask_size_{i}'] * order_book[f'ask_price_{i}']  

            # 计算基础失衡指标  
            total_value = bid_value + ask_value  
            basic_imbalance = (bid_value - ask_value) / total_value.where(total_value != 0)  

            # 计算加权失衡指标  
            weighted_bid = pd.Series(0.0, index=order_book.index)  
            weighted_ask = pd.Series(0.0, index=order_book.index)  

            weights = [1 / (i + 1) for i in range(self.depth_levels)]  
            weight_sum = sum(weights)  
            weights = [w / weight_sum for w in weights]  

            for i, weight in enumerate(weights):  
                if f'bid_size_{i}' in order_book.columns and f'bid_price_{i}' in order_book.columns:  
                    weighted_bid += weight * order_book[f'bid_size_{i}'] * order_book[f'bid_price_{i}']  
                if f'ask_size_{i}' in order_book.columns and f'ask_price_{i}' in order_book.columns:  
                    weighted_ask += weight * order_book[f'ask_size_{i}'] * order_book[f'ask_price_{i}']  

            total_weighted = weighted_bid + weighted_ask  
            weighted_imbalance = (weighted_bid - weighted_ask) / total_weighted.where(total_weighted != 0)  

            results.update({  
                'basic_imbalance': basic_imbalance,  
                'weighted_imbalance': weighted_imbalance,  
            })  

        except Exception as e:  
            print(f"计算盘口失衡因子时出错: {e}")  

        return results  

    def price_impact(self, order_book: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算价格冲击因子"""  
        if not self._has_order_book_data(order_book):  
            return {}  

        results = {}  
        try:  
            if 'bid_price_0' in order_book.columns and 'ask_price_0' in order_book.columns:  
                mid_price = (order_book['bid_price_0'] + order_book['ask_price_0']) / 2  

                # 计算不同数量级别的价格冲击  
                volumes = [100, 500, 1000, 5000]  # 可配置的数量级别  

                for vol in volumes:  
                    # 买入价格冲击  
                    buy_impact = pd.Series(0.0, index=order_book.index)  
                    remaining_vol = pd.Series(vol, index=order_book.index)  

                    for i in range(self.depth_levels):  
                        if f'ask_size_{i}' in order_book.columns and f'ask_price_{i}' in order_book.columns:  
                            executable = np.minimum(remaining_vol, order_book[f'ask_size_{i}'])  
                            buy_impact += executable * order_book[f'ask_price_{i}']  
                            remaining_vol -= executable  

                    buy_impact = (buy_impact / vol - mid_price) / mid_price  

                    # 卖出价格冲击  
                    sell_impact = pd.Series(0.0, index=order_book.index)  
                    remaining_vol = pd.Series(vol, index=order_book.index)  

                    for i in range(self.depth_levels):  
                        if f'bid_size_{i}' in order_book.columns and f'bid_price_{i}' in order_book.columns:  
                            executable = np.minimum(remaining_vol, order_book[f'bid_size_{i}'])  
                            sell_impact += executable * order_book[f'bid_price_{i}']  
                            remaining_vol -= executable  

                    sell_impact = (mid_price - sell_impact / vol) / mid_price  

                    results.update({  
                        f'buy_impact_{vol}': buy_impact,  
                        f'sell_impact_{vol}': sell_impact,  
                    })  

        except Exception as e:  
            print(f"计算价格冲击因子时出错: {e}")  

        return results  

    def order_book_slope(self, order_book: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算盘口斜率"""  
        if not self._has_order_book_data(order_book):  
            return {}  

        results = {}  
        try:  
            # 计算买卖盘的斜率  
            bid_slope = pd.Series(0.0, index=order_book.index)  
            ask_slope = pd.Series(0.0, index=order_book.index)  

            for i in range(self.depth_levels - 1):  
                # 买盘斜率（价格差除以累积量差）  
                if all(f'bid_{x}_{i}' in order_book.columns and f'bid_{x}_{i+1}' in order_book.columns   
                      for x in ['price', 'size']):  
                    bid_slope += (order_book[f'bid_price_{i}'] - order_book[f'bid_price_{i+1}']) / \
                                (order_book[f'bid_size_{i}'].cumsum() - order_book[f'bid_size_{i+1}'].cumsum())  

                # 卖盘斜率  
                if all(f'ask_{x}_{i}' in order_book.columns and f'ask_{x}_{i+1}' in order_book.columns   
                      for x in ['price', 'size']):  
                    ask_slope += (order_book[f'ask_price_{i+1}'] - order_book[f'ask_price_{i}']) / \
                                (order_book[f'ask_size_{i}'].cumsum() - order_book[f'ask_size_{i+1}'].cumsum())  

            # 计算斜率比  
            slope_ratio = bid_slope / ask_slope.where(ask_slope != 0)  

            results.update({  
                'bid_slope': bid_slope,  
                'ask_slope': ask_slope,  
                'slope_ratio': slope_ratio,   
            })  

        except Exception as e:  
            print(f"计算盘口斜率时出错: {e}")  

        return results  

    def liquidity_indicators(self, order_book: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算流动性指标"""  
        if not self._has_order_book_data(order_book):  
            return {}  

        results = {}  
        try:  
            # 计算买卖盘的累积深度  
            bid_depth = pd.Series(0.0, index=order_book.index)  
            ask_depth = pd.Series(0.0, index=order_book.index)  

            for i in range(self.depth_levels):  
                if f'bid_size_{i}' in order_book.columns:  
                    bid_depth += order_book[f'bid_size_{i}']  
                if f'ask_size_{i}' in order_book.columns:  
                    ask_depth += order_book[f'ask_size_{i}']  

            # 计算买卖盘的价格范围  
            if all(col in order_book.columns for col in ['bid_price_0', f'bid_price_{self.depth_levels-1}',  
                                                       'ask_price_0', f'ask_price_{self.depth_levels-1}']):  
                bid_range = order_book['bid_price_0'] - order_book[f'bid_price_{self.depth_levels-1}']  
                ask_range = order_book[f'ask_price_{self.depth_levels-1}'] - order_book['ask_price_0']  

                # 计算流动性指标  
                bid_liquidity = bid_depth / bid_range.where(bid_range != 0)  
                ask_liquidity = ask_depth / ask_range.where(ask_range != 0)  

                # 计算相对流动性  
                relative_liquidity = bid_liquidity / ask_liquidity.where(ask_liquidity != 0)  

                results.update({  
                    'bid_liquidity': bid_liquidity,  
                    'ask_liquidity': ask_liquidity,  
                    'relative_liquidity': relative_liquidity,  
                })  

        except Exception as e:  
            print(f"计算流动性指标时出错: {e}")  

        return results  

    def pressure_indicators(self, order_book: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算压力指标"""  
        if not self._has_order_book_data(order_book):  
            return {}  

        results = {}  
        try:  
            if 'bid_price_0' in order_book.columns and 'ask_price_0' in order_book.columns:  
                mid_price = (order_book['bid_price_0'] + order_book['ask_price_0']) / 2  

                # 计算买卖压力  
                buy_pressure = pd.Series(0.0, index=order_book.index)  
                sell_pressure = pd.Series(0.0, index=order_book.index)  

                for i in range(self.depth_levels):  
                    if all(f'{side}_{x}_{i}' in order_book.columns   
                          for side in ['bid', 'ask'] for x in ['price', 'size']):  
                        # 距离中间价的百分比  
                        bid_distance = (mid_price - order_book[f'bid_price_{i}']) / mid_price  
                        ask_distance = (order_book[f'ask_price_{i}'] - mid_price) / mid_price  

                        # 加权累加（距离越近权重越大）  
                        weight = 1 / (i + 1)  
                        buy_pressure += weight * order_book[f'bid_size_{i}'] / (1 + bid_distance)  
                        sell_pressure += weight * order_book[f'ask_size_{i}'] / (1 + ask_distance)  

                # 计算净压力和压力比  
                net_pressure = buy_pressure - sell_pressure  
                pressure_ratio = buy_pressure / sell_pressure.where(sell_pressure != 0)  

                results.update({  
                    'buy_pressure': buy_pressure,  
                    'sell_pressure': sell_pressure,  
                    'net_pressure': net_pressure,  
                    'pressure_ratio': pressure_ratio,  
                })  

        except Exception as e:  
            print(f"计算压力指标时出错: {e}")  

        return results

import pandas as pd  
from typing import Dict  


from concurrent.futures import ThreadPoolExecutor  
from typing import Dict, Tuple  

class FactorManager:  
    """因子管理器"""  
    def __init__(self, config: FactorConfig = None):  
        self.config = config or FactorConfig()  
        self.volume = VolumeFactors(self.config)

        self.trend = TrendFactors(self.config)  
        self.momentum = MomentumFactors(self.config)  
        self.volatility = VolatilityFactors(self.config)  
        self.price_structure = PriceStructureFactors(self.config)  
        self.micro_structure = MicroStructureFactors(self.config)  # 添加微观结构因子  
        self.order_book = OrderBookFactors(self.config)  

    def _reset_config_for_all(self):
        """
        基于df数据的夏普率重置参数。  当然需要考虑历史延续的，本来是动态计算的，这里一次性计算其实相当于是静态的。
        """
        self.volume = VolumeFactors(self.config)
        self.trend = TrendFactors(self.config)  
        self.momentum = MomentumFactors(self.config)  
        self.volatility = VolatilityFactors(self.config)  
        self.price_structure = PriceStructureFactors(self.config)  
        self.micro_structure = MicroStructureFactors(self.config)  # 添加微观结构因子  
        self.order_book = OrderBookFactors(self.config)  

    def _detect_abnormal_volume(self, df, window=20, threshold=3):  
        """  
        基于标准差的三级异常量能检测  
        输入：OHLCV数据框，窗口长度，标准差倍数  
        输出：异常K线布尔序列  
        """  
        v_roll =  df['volume'].rolling(window)
        p_roll = df['close'].rolling(window)
        df['volume_ma'] = v_roll.mean() + v_roll.median() / 2
        df['close_ma'] = p_roll.mean()
        df['volume_std'] = v_roll.std()
        df['close_std'] = p_roll.std()

        # abnormal_flags = df['volume'] > (df['volume_ma'] + threshold*df['volume_std'])
        # 在 Pandas 中，当你要对两个或多个条件进行元素级的逻辑运算时，不能直接使用 Python 的 and 运算符，
        # 而是应该使用位运算符 &，并且每个条件都需要用圆括号括起来。
        abnormal_flags = ((df['volume'] > (df['volume_ma'] + threshold * df['volume_std'])) &  
                  (df['close'] > (df['close_ma'] + threshold * df['close_std'])))  
        abnormal_flags.fillna(False)
        # for i in range(len(abnormal_flags)):
        #     if abnormal_flags[i]:
        #         print(f'annormal[{i}] is {df.iloc[i]}')
        #         break

        return abnormal_flags
    
    def _optimize_parameters(self, df, tau_candidates=np.linspace(0.03,0.3, 5),
                            window_candidates=range(14,29,5)):
        """
        网格搜索最优参数组合
        - 目标：优化 tau 和 window_size 参数，基于 Shapiro-Wilk 统计量最大化VWAP的分布正态性
          (Note: The rationale for optimizing normality should be clarified.
           Consider using a performance-related metric if the goal is trading strategy optimization)
        输入：OHLCV数据框，tau候选值，窗口候选值
        输出：最佳(tau, window)组合
        """
        best_score = -np.inf

        abnormal_flags = None
        best_tau = tau_candidates[0]
        best_window_size = window_candidates[0]

        abnormal_flags = self._detect_abnormal_volume(df, window=best_window_size, threshold=2.5)  #动态调整
        best_tau, best_window_size = 0.03, 14
        for tau in tau_candidates:
            for window in window_candidates:
                abnormal_flags = self._detect_abnormal_volume(df, window=best_window_size, threshold=2.5)  #动态调整
                vwap = AdaptiveTVWAP(tau).calculate(df, abnormal_flags)
                # 计算分布正态性得分 (Shapiro-Wilk statistic)
                score = shapiro(vwap).statistic
                if score > best_score:
                    best_score = score

                    best_tau = tau
                    best_window_size = window

        # print(f'best tau={best_tau} window_size={best_window_size} ')
        return best_tau, best_window_size, abnormal_flags
    
    def calculate_factors(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:  
        """并行计算所有技术因子"""  
        self.config.tau, self.config.window_size, self.config.abnormal_flags  = self._optimize_parameters(df)
        self.config.smoothing = int(self.config.window_size * 3.5)
        self._reset_config_for_all()

        factors = {}  

        # 确保数据框有正确的时间索引  
        df = df.copy()  
        if not isinstance(df.index, pd.DatetimeIndex):  
            df.index = pd.to_datetime(df.index)  

        # 创建一个线程池  
        with ThreadPoolExecutor() as executor:  
            # 提交任务到线程池  
            futures = {  
                executor.submit(self._calculate_volume_factors, df): "volume",  
                #executor.submit(self._calculate_volume_leverage_ratio, df): "leverage",
                executor.submit(self._calculate_momentum_factors, df): "momentum",  
                executor.submit(self._calculate_trend_factors, df): "trend",  
                executor.submit(self._calculate_volatility_factors, df): "volatility",  
                #executor.submit(self._calculate_price_structure_factors, df): "price_structure",  
                #executor.submit(self._calculate_micro_structure_factors, df): "micro_structure",  
                #executor.submit(self._calculate_order_book_factors, df): "order_book",  
            }  

            # 收集结果  
            for future in futures:  
                factor_type = futures[future]  
                try:  
                    result = future.result()  
                    factors.update(result)  
                except Exception as e:  
                    print(f"Error calculating {factor_type} factors: {e}")  

        # 找出所有需要考虑的指标的第一个有效数据点  
        valid_starts = []  
        for key, value in factors.items():  
            if isinstance(value, (pd.Series, pd.DataFrame)):  
                
                if not isinstance(value.index, pd.DatetimeIndex):  
                    value.index = pd.to_datetime(value.index)  
                first_valid = value.first_valid_index()  
                
                if first_valid is not None:  
                    valid_starts.append(first_valid)  
                    # if first_valid != df.index[0]:
                    #     print(f'{key} debug len {first_valid} df-first = {df.index[0]}')
        # 使用最后一个有效的起始位置作为截断点  
        if valid_starts:  
            valid_data_start = max(valid_starts)  
            
            # 截断原始数据框  
            df = df.loc[valid_data_start:]  

            # 截断所有因子  
            new_factors = {}  
            for key, value in factors.items():  
                if isinstance(value, (pd.Series, pd.DataFrame)):  
                    new_factors[key] = value.loc[valid_data_start:]  
                else:  
                    new_factors[key] = value  
            factors = new_factors  

        return factors, df  
    
    def update_twvwap(self, df, old_factors, not_calc_abnormals = False):
        if not_calc_abnormals:
            assert (df.index[0] == old_factors['vwap'].index[0] and  df.index[-1] == old_factors['vwap'].index[-1]), f'factors.py update_twvwap df and factors are not same range'
            self.config.tau, self.config.window_size, self.config.abnormal_flags  = self._optimize_parameters(df)
            self.config.smoothing = int(self.config.window_size * 3.5)
            self.volume = VolumeFactors(self.config)
            old_factors.update(self.volume.volume_profile(df))
            old_factors.update(self.volume.calculate_vpvr(df))
            old_factors.update(self.volume.vwap(df))
        else:
            # 获取时间区间  
            start_time = df.index.min().to_pydatetime()  
            end_time = df.index.max().to_pydatetime() 
            self.volume.config.abnormal_flags = self.config.abnormal_flags.loc[start_time:end_time].copy()
        old_factors.update(self.volume.twvwap(df))
        return old_factors
    
    def _calculate_volume_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算成交量相关因子"""  
        factors = {}  
        # factors.update(self.volume.obv(df))    
        factors.update(self.volume.vwap(df))
        #factors.update(self.volume.cmf(df))  
        #factors.update(self.volume.force_index(df))  
        #factors.update(self.volume.pvt(df))  
        return factors  

    def _calculate_momentum_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算动量因子"""  
        factors = {}  
        # 添加所有动量类因子  
        # factors.update(self.momentum.rsi(df))  
        factors.update(self.momentum.stochastic(df))  
        # factors.update(self.momentum.kdj(df))  
        # factors.update(self.momentum.cci(df))  
        # factors.update(self.momentum.roc(df))  
        # factors.update(self.momentum.williams_r(df))  
        return factors

    def _calculate_volatility_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算波动率因子"""  
        factors = {}  
        factors.update(self.volatility.atr(df))  
        # factors.update(self.volatility.keltner(df))  
        # factors.update(self.volatility.volatility(df))  
        # factors.update(self.volatility.price_range(df))  
        return factors  
    
    def _calculate_volume_leverage_ratio(self, df: pd.DataFrame) -> Dict[str, pd.Series]: 
        factors = {}
        factors.update(self.volume.leverage_ratio(df))
        return factors
    
    def _calculate_trend_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算趋势因子"""  
        factors = {}  
        factors.update(self.trend.macd(df))  
        factors.update(self.trend.adx(df))    
        return factors  

    def _calculate_price_structure_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算价格结构因子"""  
        factors = {}  
        factors.update(self.price_structure.pivot_points(df))  
        factors.update(self.price_structure.fractal(df))  
        factors.update(self.price_structure.zigzag(df))  
        factors.update(self.price_structure.support_resistance(df))  
        return factors  

    def _calculate_micro_structure_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算市场微观结构因子"""  
        factors = {}  
        factors.update(self.micro_structure.bid_ask_spread(df))  
        factors.update(self.micro_structure.order_flow_imbalance(df))  
        factors.update(self.micro_structure.market_depth(df))  
        factors.update(self.micro_structure.trade_size_analysis(df))  
        return factors  

    def _calculate_order_book_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
        """计算盘口因子"""  
        factors = {}  
        factors.update(self.order_book.order_book_imbalance(df))  
        factors.update(self.order_book.price_impact(df))  
        factors.update(self.order_book.order_book_slope(df))  
        factors.update(self.order_book.liquidity_indicators(df))  
        factors.update(self.order_book.pressure_indicators(df))  
        return factors
    


class IncrementalFactorManager:  
    """支持增量计算的因子管理器"""  
    def __init__(self, config: FactorConfig = None):  
        self.config = config or FactorConfig()  
        self._initialize_factor_classes()  
        
        # 状态追踪变量  
        self._previous_factors = {}  # 存储先前计算的因子结果  
        self._last_timestamp = None  # 最后处理的时间戳  
        self._is_initialized = False  # 是否已完成初始化  
        self._abnormal_flags = None  # 异常标记缓存  
        
        # 参数优化状态 - 定期更新而非每次计算  
        self._parameter_update_counter = 0  
        self._parameter_update_frequency = 1000  # 每1000个tick重新评估参数  
    
    def _initialize_factor_classes(self):  
        """初始化各因子类"""  
        self.volume = VolumeFactors(self.config)  
        self.trend = TrendFactors(self.config)  
        self.momentum = MomentumFactors(self.config)  
        self.volatility = VolatilityFactors(self.config)  
        self.price_structure = PriceStructureFactors(self.config)  
        self.micro_structure = MicroStructureFactors(self.config)  
        self.order_book = OrderBookFactors(self.config)  

    def _update_config_if_needed(self, df):  
        """定期更新配置参数，而非每次增量计算都更新"""  
        self._parameter_update_counter += 1  
        
        # 只在初始化或达到预定频率时更新参数  
        if not self._is_initialized or self._parameter_update_counter >= self._parameter_update_frequency:  
            # 参数优化  
            self.config.tau, self.config.window_size, self.config.abnormal_flags = self._optimize_parameters(df)  
            self.config.smoothing = int(self.config.window_size * 3.5)  
            
            # 使用新参数重置因子类  
            self._initialize_factor_classes()  
            
            # 重置计数器  
            self._parameter_update_counter = 0  
            
            # 标记为已初始化  
            self._is_initialized = True  
            
            return True  
        
        return False  
    
    def _validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:  
        """  
        验证数据有效性 - 检测NaN和零值  
        
        返回:  
            - 清洗后的DataFrame  
            - 指示数据是否被修改的布尔值  
        """  
        # 复制数据以避免修改原始数据  
        df_clean = df.copy()  
        data_modified = False  
        
        # 检查关键列中的NaN值  
        key_columns = ['open', 'high', 'low', 'close', 'volume']  
        available_cols = [col for col in key_columns if col in df_clean.columns]  
        
        # 1. 检查NaN值  
        nan_mask = df_clean[available_cols].isna().any(axis=1)  
        if nan_mask.any():  
            # 记录有多少行包含NaN  
            nan_count = nan_mask.sum()  
            
            # 决定如何处理NaN行 - 这里选择前向填充，适合时间序列  
            df_clean.loc[nan_mask, available_cols] = df_clean.loc[nan_mask, available_cols].fillna(method='ffill')  
            
            # 如果前向填充后仍有NaN（可能是开始几行），则用后向填充  
            if df_clean[available_cols].isna().any().any():  
                df_clean[available_cols] = df_clean[available_cols].fillna(method='bfill')  
                
            data_modified = True  
            print(f"警告: 检测到{nan_count}行数据包含NaN值，已使用填充方法修复")  
        
        # 2. 检查价格列中的零值  
        price_cols = ['open', 'high', 'low', 'close']  
        price_cols = [col for col in price_cols if col in df_clean.columns]  
        
        if len(price_cols) > 0:  
            zero_price_mask = (df_clean[price_cols] == 0).any(axis=1)  
            if zero_price_mask.any():  
                zero_count = zero_price_mask.sum()  
                
                # 对于价格为0的行，使用前值填充  
                for col in price_cols:  
                    df_clean.loc[zero_price_mask, col] = df_clean.loc[zero_price_mask, col].replace(0, method='ffill')  
                
                data_modified = True  
                print(f"警告: 检测到{zero_count}行价格数据包含0值，已使用前值填充修复")  
        
        # 3. 检查成交量为零的情况 (成交量为0是允许的，但可能需要特殊处理)  
        if 'volume' in df_clean.columns:  
            zero_volume_mask = (df_clean['volume'] == 0)  
            zero_volume_count = zero_volume_mask.sum()  
            
            if zero_volume_count > 0 and zero_volume_count / len(df_clean) > 0.5:  
                # 如果超过50%的数据成交量为0，发出警告  
                print(f"警告: 数据中有较高比例({zero_volume_count/len(df_clean):.1%})的成交量为0")  
        
        return df_clean, data_modified  
    
    def calculate_factors(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:  
        """计算因子 - 智能判断是执行全量还是增量计算"""  
        # 确保数据框有正确的时间索引  
        
        # df = df.copy()  

        clean, df = self._validate_data(df)
        if not isinstance(df.index, pd.DatetimeIndex):  
            df.index = pd.to_datetime(df.index)  
        
        # 确定是全量计算还是增量计算  
        if not self._is_initialized or self._last_timestamp is None:  
            # 首次计算 - 执行全量计算并初始化  
            return self._calculate_initial_factors(df)  
        else:  
            # 获取新增数据  
            new_data = df[df.index > self._last_timestamp]  
            if len(new_data) == 0:  
                # 没有新数据，返回上次的结果  
                return self._previous_factors, df  
            
            # 有新数据 - 执行增量计算  
            return self._calculate_incremental_factors(df, new_data)  
    
    def _calculate_initial_factors(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:  
        """首次运行的全量因子计算"""  
        # 更新配置  
        self._update_config_if_needed(df)  
        
        # 并行计算所有因子  
        factors = {}  
        
        # 使用线程池并行计算  
        with ThreadPoolExecutor() as executor:  
            futures = {  
                executor.submit(self._calculate_volume_factors, df, None): "volume",  
                executor.submit(self._calculate_momentum_factors, df, None): "momentum",  
                executor.submit(self._calculate_trend_factors, df, None): "trend",  
                executor.submit(self._calculate_volatility_factors, df, None): "volatility",  
                # 其他暂时注释掉的因子计算类似  
            }  
            
            # 收集结果  
            for future in futures:  
                factor_type = futures[future]  
                try:  
                    result = future.result()  
                    factors.update(result)  
                except Exception as e:  
                    print(f"Error calculating {factor_type} factors: {e}")  
        
        # 数据对齐处理  
        valid_starts = []  
        for key, value in factors.items():  
            if isinstance(value, (pd.Series, pd.DataFrame)):  
                if not isinstance(value.index, pd.DatetimeIndex):  
                    value.index = pd.to_datetime(value.index)  
                first_valid = value.first_valid_index()  
                if first_valid is not None:  
                    valid_starts.append(first_valid)  
        
        # 使用最后一个有效的起始位置作为截断点  
        if valid_starts:  
            valid_data_start = max(valid_starts)  
            
            # 截断原始数据框和因子  
            df = df.loc[valid_data_start:]  
            factors = {k: (v.loc[valid_data_start:] if isinstance(v, (pd.Series, pd.DataFrame)) else v)   
                      for k, v in factors.items()}  
        
        # 更新状态  
        self._previous_factors = factors  
        self._last_timestamp = df.index[-1]  
        
        return factors, df  
    
    def _calculate_incremental_factors(self, full_df: pd.DataFrame, new_data: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:  
        """增量因子计算"""  
        # 定期更新参数  
        params_updated = self._update_config_if_needed(full_df)  
        
        # 准备结果容器  
        updated_factors = {}  
        
        # 如果参数已更新，需要重新计算所有因子  
        if params_updated:  
            return self._calculate_initial_factors(full_df)  
        
        # 并行增量计算  
        with ThreadPoolExecutor() as executor:  
            futures = {  
                executor.submit(self._calculate_volume_factors, new_data,   
                               {k: v for k, v in self._previous_factors.items() if k in self.volume.get_factor_names()}): "volume",  
                
                executor.submit(self._calculate_momentum_factors, new_data,  
                               {k: v for k, v in self._previous_factors.items() if k in self.momentum.get_factor_names()}): "momentum",  
                
                executor.submit(self._calculate_trend_factors, new_data,  
                               {k: v for k, v in self._previous_factors.items() if k in self.trend.get_factor_names()}): "trend",  
                
                executor.submit(self._calculate_volatility_factors, new_data,  
                               {k: v for k, v in self._previous_factors.items() if k in self.volatility.get_factor_names()}): "volatility",  
                
                # 其他因子类似  
            }  
            
            # 收集结果  
            for future in futures:  
                factor_type = futures[future]  
                try:  
                    result = future.result()  
                    updated_factors.update(result)  
                except Exception as e:  
                    print(f"Error calculating incremental {factor_type} factors: {e}")  
        
        # 更新状态  
        self._previous_factors = updated_factors  
        self._last_timestamp = full_df.index[-1]  
        
        return updated_factors, full_df  
    
    def _calculate_volume_factors(self, df: pd.DataFrame, previous_results: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:  
        """计算成交量相关因子"""  
        factors = {}  
        
        # 增量计算各个因子  
        factors.update(self.volume.twvwap(df, previous_results.get('twvwap', None) if previous_results else None))  
        factors.update(self.volume.vwap(df, previous_results.get('vwap', None) if previous_results else None))  
        
        return factors  
    
    def _calculate_momentum_factors(self, df: pd.DataFrame, previous_results: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:  
        """计算动量因子"""  
        factors = {}  
        
        # 增量计算随机指标  
        stoch_previous = {k: previous_results.get(k, None) for k in ['stoch_k', 'stoch_d']} if previous_results else None  
        factors.update(self.momentum.stochastic(df, stoch_previous))  
        
        return factors  
    
    def _calculate_trend_factors(self, df: pd.DataFrame, previous_results: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:  
        """计算趋势因子"""  
        factors = {}  
        
        # 增量计算MACD  
        macd_previous = {k: previous_results.get(k, None) for k in ['macd', 'macd_signal', 'macd_diff']} if previous_results else None  
        factors.update(self.trend.macd(df, macd_previous))  
        
        # 增量计算ADX  
        adx_previous = {k: previous_results.get(k, None) for k in ['adx', '+di', '-di']} if previous_results else None  
        factors.update(self.trend.adx(df, adx_previous))  
        
        return factors  
    
    def _calculate_volatility_factors(self, df: pd.DataFrame, previous_results: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:  
        """计算波动率因子"""  
        factors = {}  
        
        # 增量计算ATR  
        atr_previous = {k: previous_results.get(k, None) for k in ['atr', 'atr_pct']} if previous_results else None  
        factors.update(self.volatility.atr(df, atr_previous))  
        
        return factors  
    
    # 其他辅助方法保持不变  
    _detect_abnormal_volume = FactorManager._detect_abnormal_volume  
    _optimize_parameters = FactorManager._optimize_parameters  