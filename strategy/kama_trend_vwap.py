from indicators import LHFrameStd   

from .signals import OrderSignal
import pandas as pd
import numpy as np

def get_trend_signal(trend_df, multiVwap, window=200):
    kama_delta = trend_df['kama1'] - trend_df['kama2']
    sum_delta = kama_delta.iloc[-window:].sum()
    kama1_delta = trend_df['kama1'].iloc[-1] - trend_df['kama1'].iloc[-2] 
    
    if trend_df.iloc[-1]['kama1'] > trend_df.iloc[-1]['kama2'] and sum_delta > 0 and kama1_delta > 0:
        return 'long'
    elif trend_df.iloc[-1]['kama1'] < trend_df.iloc[-1]['kama2'] and sum_delta < 0 and kama1_delta < 0:
        return 'short'
    else:
        return 'neutral'

def trend_strength_3cls(series: pd.Series,
                        window: int = 50,
                        t_th: float = 2.0,
                        r2_min: float = 0.25,
                        use_log: bool = True,
                        labels: bool = False) -> pd.Series:
    """
    返回每根bar的趋势强度三分类：
      1 = 上升趋势, 0 = 震荡/不显著, -1 = 下降趋势
    判定依据：窗口内线性回归斜率的 t 统计量与 R²。
    参数:
      series  : 等间隔价格序列（如收盘价）
      window  : 滚动窗口长度（>=3）
      t_th    : t统计量阈值，默认|t|>2视为显著
      r2_min  : 最小R²，过滤拟合差的窗口
      use_log : 是否对数化价格（推荐True）
      labels  : True则返回 'up'/'sideways'/'down'
    """
    y = pd.Series(series, dtype="float64")
    if use_log:
        y = np.log(y)
    n, w = len(y), int(window)
    if w < 3: raise ValueError("window must be >= 3")

    x = np.arange(w, dtype="float64")
    Sx, Sxx = x.sum(), (x**2).sum()
    denom = w * Sxx - Sx**2
    x_mean = x.mean()
    Sxxc = ((x - x_mean)**2).sum()

    slope = np.full(n, np.nan)
    tstat = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    yv = y.values
    for i in range(w-1, n):
        wy = yv[i-w+1:i+1]
        Sy, Sxy = wy.sum(), (wy * x).sum()
        b = (w*Sxy - Sx*Sy) / denom
        a = wy.mean() - b * x_mean
        yhat = a + b*x
        resid = wy - yhat
        sse = (resid**2).sum()
        tss = ((wy - wy.mean())**2).sum()

        slope[i] = b
        r2[i] = 1 - sse/tss if tss > 0 else np.nan
        if sse > 0:
            sigma2 = sse / (w - 2)
            tstat[i] = b / np.sqrt(sigma2 / Sxxc)

    up = (tstat >  t_th) & (r2 >= r2_min) & (slope > 0)
    dn = (tstat < -t_th) & (r2 >= r2_min) & (slope < 0)

    if labels:
        out = np.where(up, "up", np.where(dn, "down", "sideways"))
    else:
        out = np.where(up, 1, np.where(dn, -1, 0))
    return pd.Series(out, index=y.index, name="trend_3cls")

class Strategy:
    def __init__(self, multiVwap:LHFrameStd.MultiTFVWAP, can_open_total):
        self.multiVwap = multiVwap
        self.can_open_total = can_open_total
        self.open_orders = {'long': None, 'short': None}
        self.eval_history = []
        self.strategy_log_interval = 0

    def cancel_order(self, side):
        self.open_orders[side] = None

    def clear_order(self, side):
        self.open_orders[side] = None

    def should_cancel(self, side):
        # 简单撤单：比如订单挂单超过固定时间可撤(示例里恒False)
        return False

    def evaluate(self, side, cur_close, close, multiFrameVwap, open2equity_pct):
        # 这里您的震荡策略逻辑入口，需要你补充细化
        # 返回 OrderSignal 或 None
        return None

    
    def generate_order_signals(self, trend_df, df, cur_index, closes, position_amount_dict_sub, min_amount = 2):
        cur_close = df.loc[cur_index, 'close']
        sig_long, sig_short = None, None
        all_values = sum(position_amount_dict_sub.values())
        open2equity_pct = all_values / self.can_open_total
        trend = get_trend_signal(trend_df, self.multiVwap)
        

        kama2_trend = trend_strength_3cls(trend_df['kama2'].iloc[-120:], 100)

        if True and trend == 'long': 
            self.cancel_order('long')
            self.clear_order('long')
            if open2equity_pct < 0.05 and df.loc[cur_index]['close'] <= self.multiVwap.SFrame_vwap_down_sl2.loc[cur_index]:
                price_prepare = trend_df['kama2' if kama2_trend.iloc[-1] == 1 else 'kama1'].iloc[-1] 
                price_prepare = min(price_prepare, min(df['low'].iloc[-90:]))
                price_prepare = min(price_prepare, self.multiVwap.SFrame_vwap_down_sl2.loc[cur_index]) - self.multiVwap.atr_dic['ATR'].loc[cur_index]/2
                assert self.multiVwap.df.loc[cur_index]['close'] == closes.iloc[-1], f"self.multiVwap.df[cur_index]['close'] == closes[-1] is false {self.multiVwap.df[cur_index]['close']} {closes.iloc[-1]}"

                sig_long = OrderSignal('long', True, price_prepare, min_amount, order_type='limit', tier_explain="trend long kama1 entry")
            sig_short = None

        elif True and trend == 'short':
            self.cancel_order('short')
            self.clear_order('short')
            if open2equity_pct < 0.05 and df.loc[cur_index]['close'] >= self.multiVwap.SFrame_vwap_up_sl2.loc[cur_index]:
                price_prepare = trend_df['kama2' if kama2_trend.iloc[-1] == -1 else 'kama1'].iloc[-1]
                price_prepare = max(price_prepare, max(df['high'].iloc[-90:]))
                price_prepare = max(price_prepare, self.multiVwap.SFrame_vwap_up_sl2.loc[cur_index]) + self.multiVwap.atr_dic['ATR'].loc[cur_index]/2
                sig_short = OrderSignal('short', True, price_prepare, min_amount, order_type='limit', tier_explain="trend short kama1 entry")
            sig_long = None

        else:
            # print('moderating, no trend')
            # 震荡区间
            for side in ('long', 'short'):
                if self.should_cancel(side):
                    self.cancel_order(side)
                    self.clear_order(side)
            sig_short = self.evaluate('short', cur_close, closes, self.multiVwap, open2equity_pct)
            if sig_short:
                sig_short.price = max(sig_short.price, max(df['high'].iloc[-30:]))
            sig_long  = self.evaluate('long',  cur_close, closes, self.multiVwap, open2equity_pct)
            if sig_long:
                sig_long.price = min(sig_long.price, min(df['low'].iloc[-30:]))
        # print(f'generated {sig_long}, \t {sig_short}')
        return sig_long, sig_short
