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
        self.window_tau_s = 300
        self.window_tau_h = int(self.window_tau_s / 3)
        self.window_tau_l = int(self.window_tau_h/3)

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
        self.rma_smooth_window   = 4
        self.febonaqis  = [i+1 for i in [0, 0.236, 0.382, 0.5, 0.618, 0.768, 1] ]
        

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


        self.vol_df = None

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
                           self.window_HFrame, 40, 1-0.0027, 0.99,
                           debug=debug)
            fS = ex.submit(vwap_calc.vpvr_center_vwap_log_decay,
                           o, c, v,
                           self.window_SFrame, 40, 1-0.0027, 0.99,
                           debug=debug)
            hvp = fH.result()
            svp = fS.result()
            fbH = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_HFrame, bins=40,
                            pct=0.0027, decay=0.995, vwap_series=hvp,
                            debug=debug)
            fbS = ex.submit(vwap_calc.vpvr_pct_band_vwap_log_decay,
                            open_prices=o, close_prices=c, vol=v,
                            length=self.window_SFrame, bins=40,
                            pct=0.0027, decay=0.995, vwap_series=svp,
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
        ) 
        h_std = (
            close.rolling(self.window_HFrame).std() * 0.9
            + H_swing * 0.1
        ) 
        self.HFrame_price_std = pd.Series(h_std.values, index=close.index)

        # 5) 计算上下轨（RMA + max/min 合并逻辑）
        rma = lambda s: ta.rma(s, length=self.rma_smooth_window)
        slow, shigh = self.slow_poc, self.shigh_poc
        hlow, hhigh = self.hlow_poc, self.hhigh_poc

        # SFrame
        self.SFrame_vwap_up_poc    = rma(shigh)
        self.SFrame_vwap_up_getin  = rma(shigh + self.SFrame_price_std * self.febonaqis[1])
        self.SFrame_vwap_up_sl     = rma(shigh + self.SFrame_price_std * self.febonaqis[2])
        self.SFrame_vwap_up_sl2    = rma(shigh + self.SFrame_price_std * self.febonaqis[3])

        self.SFrame_vwap_down_poc    = rma(slow)
        self.SFrame_vwap_down_getin  = rma(slow - self.SFrame_price_std * self.febonaqis[1])
        self.SFrame_vwap_down_sl     = rma(slow - self.SFrame_price_std * self.febonaqis[2])
        self.SFrame_vwap_down_sl2    = rma(slow - self.SFrame_price_std * self.febonaqis[3])

        # HFrame（取更保守的 max/min）
        self.HFrame_vwap_up_poc    = rma(hhigh) #np.maximum(self.SFrame_vwap_up_getin, rma(hhigh))
        self.HFrame_vwap_up_getin  = np.maximum(self.SFrame_vwap_up_getin,
                                                rma(hhigh + self.HFrame_price_std))
        self.HFrame_vwap_up_sl     = np.maximum(self.SFrame_vwap_up_sl,
                                                rma(hhigh + 2*self.HFrame_price_std))
        self.HFrame_vwap_up_sl2    = np.maximum(self.SFrame_vwap_up_sl2,
                                                rma(hhigh + 4*self.HFrame_price_std))

        self.HFrame_vwap_down_poc    = rma(hlow) #np.minimum(self.SFrame_vwap_down_getin, rma(hlow))
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

        self.vol_df = self.compute_volume_channels(self.df)
        self.momentum_df = self.anchored_momentum()

        

    # 1) volume核心指标计算
    def compute_volume_channels(self, df: pd.DataFrame, 
                                volume_map_period: int = 240, 
                                volume_scale: float = 0.02
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
            return s.ewm(span=per, adjust=False).mean()

        def sma(s: pd.Series, per: int) -> pd.Series:
            return s.rolling(per).mean()

        # --- 1) 计算 src ---
        src = self.LFrame_vp_poc_series.reindex(df.index)
        if sm:
            src = ema(src, smp)

        # --- 2) 计算 fast/slow momentum ---
        p = 2 * momentumPeriod + 1
        base_sma = sma(self.LFrame_vp_poc_series, p).reindex(df.index)
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

import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_all_multiftfpoc_vars(multFramevp_poc,
                              symbol='',
                              is_trading=False,
                              save_to_file=True):
    # —— 开头：确保所有 index 都是 DatetimeIndex —— 
    df = getattr(multFramevp_poc, 'df', None)
    if isinstance(df, pd.DataFrame):
        # 如果没有 datetime 列，就把原 ts 索引（int 秒）转换
        if not isinstance(df.index, pd.DatetimeIndex):
            # 假设原 index 是 unix timestamp (s)
            df = df.copy()
            df.index = pd.to_datetime(df.index.astype(int), unit='s')
        # 如果你还有单独的 datetime 列，也可以更新一下
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        multFramevp_poc.df = df

    # 其它 series 也同理
    vars_to_plot = [
      'LFrame_vp_poc_series','SFrame_vp_poc', 'SFrame_vwap_up_poc',
      'SFrame_vwap_up_sl','SFrame_vwap_down_poc','SFrame_vwap_down_sl',
      'HFrame_vwap_up_getin','HFrame_vwap_up_sl','HFrame_vwap_up_sl2',
      'HFrame_vwap_down_getin','HFrame_vwap_down_sl','HFrame_vwap_down_sl2',
    ]
    for var in vars_to_plot:
        s = getattr(multFramevp_poc, var, None)
        if isinstance(s, pd.Series) and not isinstance(s.index, pd.DatetimeIndex):
            setattr(multFramevp_poc, var, 
                    pd.Series(s.values,
                              index=pd.to_datetime(s.index.astype(int), unit='s'),
                              name=s.name))

    # ———— 开始画图 ————
    fig, (ax_k, ax_vol) = plt.subplots(
        2,1, figsize=(15,8), sharex=True,
        gridspec_kw={'height_ratios':[7,3]},
        constrained_layout=True
    )
    fig.patch.set_facecolor('black')
    ax_k.set_facecolor('black')
    ax_vol.set_facecolor('black')

    # 1) K 线 / close
    if isinstance(df, pd.DataFrame) and {'open','high','low','close'}.issubset(df.columns):
        times = mdates.date2num(df.index.to_pydatetime())
        o,h,l,c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        quotes = np.column_stack([times, o, h, l, c])
        try:
            from mplfinance.original_flavor import candlestick_ohlc
            candlestick_ohlc(
                ax_k, quotes,
                width=(df.index[1]-df.index[0]).seconds/86400*0.8,
                colorup='lime', colordown='red', alpha=0.7
            )
        except ImportError:
            ax_k.plot(df.index, c, color='white', label='Close', linewidth=1)
    else:
        # fallback
        for var in ['LFrame_ohlc5_series','close']:
            s = getattr(multFramevp_poc, var, None)
            if isinstance(s, pd.Series):
                ax_k.plot(s.index, s.values,
                          color='lightgray', linewidth=1, label=var)
                break

    # 2) 其它 series
    colors = { # … 同上略 …
        'LFrame_vp_poc_series':'yellow','SFrame_vp_poc':'purple',
        # …
    }
    for var in vars_to_plot:
        s = getattr(multFramevp_poc, var, None)
        if isinstance(s, pd.Series) and not s.isna().all():
            ax_k.plot(
                s.index, s.values,
                label=var, color=colors.get(var,'white'),
                linewidth=2 if 'HFrame' not in var else 1,
                linestyle='-' if '_poc' in var else 'dotted'
            )

    # 3) 色带
    def _fill(a,b,idx,color):
        ax_k.fill_between(
            idx, a.values, b.values,
            color=color, alpha=0.2
        )
    up1 = getattr(multFramevp_poc,'HFrame_vwap_up_getin',None)
    up2 = getattr(multFramevp_poc,'HFrame_vwap_up_sl',None)
    if isinstance(up1,pd.Series) and isinstance(up2,pd.Series):
        _fill(up1, up2, up1.index, "hotpink")
    dn1 = getattr(multFramevp_poc,'HFrame_vwap_down_sl',None)
    dn2 = getattr(multFramevp_poc,'HFrame_vwap_down_getin',None)
    if isinstance(dn1,pd.Series) and isinstance(dn2,pd.Series):
        _fill(dn1, dn2, dn1.index, "deepskyblue")

    # 4) 成交量
    if isinstance(df, pd.DataFrame) and 'vol' in df.columns:
        ax_vol.bar(df.index, df['vol'].values,
                   width=(df.index[1]-df.index[0]).seconds/86400*0.8,
                   color='dodgerblue', alpha=0.6)

    # 5) 格式化 x 轴
    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # fig.autofmt_xdate(rotation=10)
    for ax in (ax_k, ax_vol):
        ax.xaxis.set_tick_params(rotation=10)

    # 6) 其余美化略……
    ax_k.set_title(f"{symbol} vp_poc/VWAP - {datetime.now()}", color='w')
    #ax_k.legend(loc='upper left', facecolor='black', labelcolor='white')
    ax_k.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,  # 可自行调节，一行排4~5个
        fontsize="small",
        facecolor="black",
        labelcolor="white"
    )
    ax_k.grid(True, alpha=0.2)
    ax_vol.grid(True, alpha=0.2)

    if save_to_file:
        os.makedirs("plots", exist_ok=True)
        fn = f"plots/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(fn, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Saved to", fn)
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

