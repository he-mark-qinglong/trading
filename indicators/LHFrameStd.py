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
        #34:21:9
        self.febonaqis  = [i+1 for i in [0, 0.236, 0.382, 0.5, 0.618, 0.768, 1] ]
        #参数解释，按照日线的趋势均线规则，参数取, 5,14,30. 但是30作为短线回归几乎不太有效（后续资金量如果比较大可以考虑)，所以30周期的先不用。
        #全部基于1h的级别基础之上来乘以这个系数.
        self.window_tau_s = int(4*24*10*self.febonaqis[0])  #1d = 1h x 24, 14x24 == 336
        self.window_tau_h = int(120*self.febonaqis[0])     #8h = 1h x 12, 5x24 == 120
        self.window_tau_l = int(24*self.febonaqis[0])   #1h = 2.5m x 24

class MultiTFVWAP:
    def __init__(self,
                 lambd=0.03,
                 window_LFrame=12,
                 window_HFrame=12*10,
                 window_SFrame=12*10*4,
                 std_window_LFrame=5):
        self.lambd               = lambd
        self.window_LFrame       = window_LFrame
        self.window_HFrame       = window_HFrame
        self.window_SFrame       = window_SFrame
        self.std_window_LFrame   = std_window_LFrame
        self.rma_smooth_window   = int(std_window_LFrame)
        self.rma_smooth_window_s   = int(std_window_LFrame)
        
        self.febonaqis  = [i+1 for i in [0, 0.236, 0.382, 0.5, 0.618, 0.768, 1] ]
        

        # 原始 OHLCV 全量存储
        self.df = None

        # 各种结果的全量 Series（索引与 self.df 保持一致）
        self.SFrame_center = pd.Series(dtype=float)
        self.SFrame_vwap_poc       = pd.Series(dtype=float)
        self.slow_poc3            = pd.Series(dtype=float)
        self.shigh_poc3           = pd.Series(dtype=float)
        self.slow_poc2            = pd.Series(dtype=float)
        self.shigh_poc2           = pd.Series(dtype=float)
        # 向量化后的一些 Series
        self.LFrame_ohlc5_series = pd.Series(dtype=float)

        # 上下轨
        for attr in [
            'SFrame_vwap_up_poc','SFrame_vwap_up_getin','SFrame_vwap_up_getout',
            'SFrame_vwap_up_sl','SFrame_vwap_up_sl2','SFrame_vwap_down_poc',
            'SFrame_vwap_down_getin','SFrame_vwap_down_getout',
            'SFrame_vwap_down_sl','SFrame_vwap_down_sl2',
        ]:
            setattr(self, attr, pd.Series(dtype=float))


        self.vol_df = None

    def _run_heavy(self, df_block, debug=False):
        """在一段 df_block 上并行计算 vp_poc 和 band，返回带 index 的 Series"""
        o, c, v = df_block['open'], df_block['close'], df_block['vol']
        with ThreadPoolExecutor(max_workers=3) as ex:
            sigma3 = 99.73/100
            sigma2 = 95.45/100

            fS = ex.submit(vwap_calc.vpvr_center_vwap_log_decay,
                           o, c, v,
                           self.window_SFrame, 60, sigma3, 0.99,
                           debug=debug)
            svp = fS.result()

            
            fbS3 = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_SFrame, bins=60,
                            pct=(1-sigma3)/2, decay=0.995, vwap_series=svp,
                            debug=debug)
            
            fbS2 = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_SFrame, bins=60,
                            pct=(1-sigma2)/2, decay=0.995, vwap_series=svp,
                            debug=debug)
            
        idx = df_block.index
        slow_arr3, shigh_arr3 = fbS3.result()
        slow_arr2, shigh_arr2 = fbS2.result()
        assert len(svp) == len(idx) and  len(slow_arr3) == len(idx) and  len(shigh_arr3) == len(idx), 'heavy result length not equal to df'
        return {
            'S':     pd.Series(svp,           index=idx),
            'slow3':  pd.Series(slow_arr3,      index=idx),
            'shigh3': pd.Series(shigh_arr3,     index=idx),
            's_low2':  pd.Series(slow_arr2,      index=idx),
            's_high2': pd.Series(shigh_arr2,     index=idx),
        }

    def calc_atr(self, period=14, high_col="high", low_col="low", close_col="close", std_multiplier=2,):
        """
        计算ATR并返回Series，可自动识别DataFrame列名
        Params:
            df      -- 带有high/low/close列的DataFrame
            period  -- ATR窗口期（默认14）
            high_col, low_col, close_col -- 列名，如自定义表头可改
        Returns:
            Series: ATR序列。如需直接加列可用 df['ATR'] = ...
        """
        df = self.df
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
        # 计算ATR的标准差
        atr_std = atr.rolling(window=period).std()

        # 计算2倍标准差边界
        upper_bound = atr + (std_multiplier * atr_std)
        lower_bound = atr - (std_multiplier * atr_std)
        
        return {
            'ATR': atr,
            'ATR_std': atr_std,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
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
        overlap = max(self.window_LFrame, self.window_SFrame)
        if old_df is None:
            block = new_df
        else:
            head  = old_df.iloc[-overlap:] if len(old_df) >= overlap else old_df
            block = pd.concat([head, new_df], axis=0)

        # 4) 在 block 上跑最耗时部分
        out = self._run_heavy(block, debug=debug)

        # 5) 只取 > old_last_ts 的那段新结果，dropna 再 concat
        # def take_new(old: pd.Series, new: pd.Series) -> pd.Series:
        #     if old_last_ts is not None:
        #         new = new[new.index > old_last_ts]
        #     new = new.dropna()
        #     if new.empty:
        #         return old
        #     return pd.concat([old, new])
        

        def round_series_to_hundred_floor(s: pd.Series) -> pd.Series:
            """
            将 Series 中的数值向下取整到 100 的倍数，并转换为整数类型（无小数位）。
            处理 NaN：保留 NaN（如果需要可以改为填充）。
            """
            # 仅对非空值进行处理
            mask = s.notna()
            # 向下取整到 100 的倍数：先除以 100，向下取整，再乘回 100
            values = s[mask].astype(float)
            rounded = (np.floor(values / 20) * 20).astype('int64')
            result = s.copy()
            result.loc[mask] = rounded
            return result

        def take_new(old: pd.Series, new: pd.Series, old_last_ts=None) -> pd.Series:
            """
            合并 old 与 new（仅取 new 中时间晚于 old_last_ts 的部分），
            并把结果中的数值都向下取整到 100 的倍数（去掉小数位）。
            """
            if old_last_ts is not None:
                new = new[new.index > old_last_ts]
            new = new.dropna()
            if new.empty:
                # 对返回的 old 也进行相同的取整处理，确保一致性（可选）
                return round_series_to_hundred_floor(old)
            combined = pd.concat([old, new])
            # 对合并后的 series 做取整处理并返回
            return round_series_to_hundred_floor(combined)


        
        self.SFrame_vwap_poc       = take_new(self.SFrame_vwap_poc,       out['S'])
        self.slow_poc3            = take_new(self.slow_poc,            out['slow3'])
        self.shigh_poc3           = take_new(self.shigh_poc,           out['shigh3'])
        self.slow_poc2            = take_new(self.slow_poc2,            out['s_low2'])
        self.shigh_poc2           = take_new(self.shigh_poc2,           out['s_high2'])
        self.SFrame_center = (self.slow_poc2 + self.shigh_poc2)/2

        # 6) 向量化后续计算
        self._run_vectorized()

        return self.df

    def _run_vectorized(self):
        df    = self.df
        close = df['close']

        # 1) OHLC5
        self.LFrame_ohlc5_series = pd.Series(close.values, index=close.index)

        # 2) RMA 平滑
        self.SFrame_vwap_poc = ta.rma(self.SFrame_vwap_poc,
                                    length=self.rma_smooth_window_s)

        # 5) 计算上下轨（RMA + max/min 合并逻辑）
        # rma_s = lambda s: ta.ema(s, length=self.rma_smooth_window_s)
        rma_s = lambda s : s
        slow3, shigh3 = self.slow_poc3, self.shigh_poc3
        s_low2, s_high2 = self.slow_poc2, self.shigh_poc2

        # SFrame
        self.SFrame_vwap_up_poc    = rma_s(shigh3)
        self.SFrame_vwap_up_getin  = rma_s(s_high2)  #shigh > s_high2
        self.SFrame_vwap_up_sl     = rma_s(shigh3 + (shigh3 - s_high2))
        self.SFrame_vwap_up_sl2    = rma_s(shigh3 + 2*(shigh3 - s_high2))

        self.SFrame_vwap_down_poc    = rma_s(slow3)
        self.SFrame_vwap_down_getin  = rma_s(s_low2)  #s_low2 > slow
        self.SFrame_vwap_down_sl     = rma_s(slow3 - (s_low2 - slow3))
        self.SFrame_vwap_down_sl2    = rma_s(slow3 - 2* (s_low2 - slow3))

        self.SFrame_vwap_poc.reindex(self.df.index, method='ffill')
        self.SFrame_vwap_down_sl2.reindex(self.df.index, method='ffill')
        self.SFrame_vwap_up_sl2.reindex(self.df.index, method='ffill')

    def calculate_SFrame_vwap_poc_and_std(self, coin_date_df, debug=False):
        """
        旧接口兼容：清空后全量跑一次。
        """
        self.df = None
        for attr in [
            'SFrame_center','SFrame_vwap_poc',
            'slow_poc','shigh_poc', 'slow_poc2','shigh_poc2',
        ]:
            setattr(self, attr, pd.Series(dtype=float))
        self.append_df(coin_date_df, debug=debug)

        self.vol_df = self.compute_volume_channels(self.df)
        self.momentum_df = self.anchored_momentum()

        self.atr_dic = self.calc_atr()
        self.atr_dic['ATR'].reindex(self.df.index,  method='ffill')

    # 1) volume核心指标计算
    def compute_volume_channels(self, df: pd.DataFrame, 
                                volume_map_period: int = 240, 
                                volume_scale: float = 1
                                ) -> pd.DataFrame:
        """
        在原 df 上计算：
        - sma_vol  : 成交量简单移动平均
        - std_vol  : 成交量标准差
        - upper    : sma_vol + 3 * std_vol
        - lower    : sma_vol + 2 * std_vol
        - vol_scaled, sma_scaled, upper_scaled, lower_scaled: 均乘以 volume_scale
        - color & alpha: 根据涨跌和区间设定 RGBA
        """
        df = df.copy()
        # 均线和标准差
        df['sma_vol'] = df['vol'].rolling(volume_map_period).mean()
        df['std_vol'] = df['vol'].rolling(volume_map_period).std()

        # 通道上下轨
        df['upper'] = df['sma_vol'] + 3 * df['std_vol']
        df['lower'] = df['sma_vol'] + 2 * df['std_vol']

        
        # 缩放后数值
        df['vol_scaled']  = df['vol'] * volume_scale
        df['sma_scaled']  = df['sma_vol'] * volume_scale
        df['upper_scaled'] = df['upper'] * volume_scale
        df['lower_scaled'] = df['lower'] * volume_scale

        return df

    def anchored_momentum(
        self,
        sm: bool = True,
        showHistogram: bool = True,
    ) -> pd.DataFrame:
        """
        在原 df 上计算：
        - amom (Fast Momentum)
        - amoms (Signal)
        - hl (Histogram)
        - hlc (Bar-color)
        返回新的 DataFrame（同 df.index）。
        """
        # 周期参数
        momentumPeriod = int(self.window_SFrame / 2)
        signalPeriod   = int(self.window_HFrame / 2)
        smp            = int(self.window_LFrame / 2)

        # 拷一份避免修改原始
        df = self.df.copy()

        def ema(s: pd.Series, per: int) -> pd.Series:
            alpha = 1 - 2 ** (-1.0 / per)
            return s.ewm( alpha=alpha, adjust=False).mean()

        def sma(s: pd.Series, per: int) -> pd.Series:
            return s.rolling(per).mean()

        # --- 1) 计算 src ---
        src = self.SFrame_center.reindex(df.index)
        if sm:
            src = ema(src, smp)

        # --- 2) 计算 fast/slow momentum ---
        p = 2 * momentumPeriod + 1
        base_sma = sma(self.SFrame_center, p).reindex(df.index)
        df['amom']  = 100 * (src / base_sma - 1)
        df['amoms'] = sma(df['amom'], signalPeriod)

        # --- 3) 计算 histogram hl ---
        df['hl'] = np.nan
        if showHistogram:
            pos = (df['amom'] > df['amoms']) & (df['amom'] > 0) & (df['amoms'] > 0)
            neg = (df['amom'] < df['amoms']) & (df['amom'] < 0) & (df['amoms'] < 0)
            df.loc[pos, 'hl'] = np.minimum(df.loc[pos,'amom'],  df.loc[pos,'amoms'])
            df.loc[neg, 'hl'] = np.maximum(df.loc[neg,'amom'],  df.loc[neg,'amoms'])
        # 这样不会触发 inplace 警告
        df['hl'] = df['hl'].fillna(0)

        # --- 4) 计算 bar-color (hlc) ---
        cond_fast = df['amom'] > df['amoms']
        cond_pos  = df['amom'] >= 0

        # 矢量化赋值，避免循环和 df.at
        df['hlc'] = np.where(
            cond_fast &  cond_pos,  'green',
            np.where(
                cond_fast & ~cond_pos, 'orange',
                np.where(
                    ~cond_fast & cond_pos, 'orange',
                    'red'
                )
            )
        )
        
        return df #df[['datetime', 'amom','amoms','hl','hlc']].reindex(self.df.index)

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

