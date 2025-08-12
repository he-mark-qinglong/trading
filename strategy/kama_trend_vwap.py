from indicators import LHFrameStd   

from .signals import OrderSignal
import pandas as pd
import numpy as np

def get_trend_signal(trend_df, multiVwap, window=200):
    kama_delta = trend_df['kama1'] - trend_df['kama2']
    sum_delta = kama_delta.iloc[-window:].sum()
    if trend_df.iloc[-1]['kama1'] > trend_df.iloc[-1]['kama2'] and sum_delta > 0:
        return 'long'
    elif trend_df.iloc[-1]['kama1'] < trend_df.iloc[-1]['kama2'] and sum_delta < 0:
        return 'short'
    else:
        return 'neutral'

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
        
        #两种touch是回溯的，所以谁先是True代表谁有效--也就是break之后的那个就是最近触及的那个sl。
        touched_down = False
        touched_up = False
        for i in range(len(df)-1, len(df)-300, -1):
            index = df.index[i]
            down_sl2 = self.multiVwap.SFrame_vwap_down_sl2.loc[index]
            up_sl2 = self.multiVwap.SFrame_vwap_up_sl2.loc[index]

            if df.loc[index]['close'] <= down_sl2:
                touched_down = True
                break
            if df.loc[index]['close'] >= up_sl2:
                touched_up = True
                break

        if True and trend == 'long':
            if touched_down:
                min_amount *= 2
        
            self.cancel_order('long')
            self.clear_order('long')
            if open2equity_pct < 0.05:# and df.loc[cur_index]['low'] <= self.multiVwap.SFrame_vwap_down_sl2.loc[cur_index]:
                price_prepare = trend_df['kama1'].iloc[-1]
                price_prepare = min(price_prepare, min(df['low'].iloc[-90:]))
                price_prepare = min(price_prepare, self.multiVwap.SFrame_vwap_down_sl2.loc[cur_index]) - self.multiVwap.atr_dic['ATR'].loc[cur_index]/2
                assert self.multiVwap.df.loc[cur_index]['close'] == closes.iloc[-1], f"self.multiVwap.df[cur_index]['close'] == closes[-1] is false {self.multiVwap.df[cur_index]['close']} {closes.iloc[-1]}"

                sig_long = OrderSignal('long', True, price_prepare, min_amount, order_type='limit', tier_explain="trend long kama1 entry")
            sig_short = None

        elif True and trend == 'short':
            if touched_up:
                min_amount *= 2
            self.cancel_order('short')
            self.clear_order('short')
            if open2equity_pct < 0.05:# and df.loc[cur_index]['high'] >= self.multiVwap.SFrame_vwap_up_sl2.loc[cur_index]:
                price_prepare = trend_df['kama1'].iloc[-1]
                # if self.multiVwap.SFrame_vwap_poc.iloc[-1] < trend_df.loc[cur_index, 'kama2']:
                #     price_prepare = max(price_prepare, self.multiVwap.SFrame_vwap_poc.iloc[-1])
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
