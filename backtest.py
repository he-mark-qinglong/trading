import pandas as pd
import numpy as np
import time

# 假设你已有的模块和函数路径正确
from dynamic_kama import compute_dynamic_kama, anchored_momentum_via_kama
# multiVwap 计算相关，请确保可调用
# data 相关，read_and_sort_df, resample_to 可使用

class OrderSignal:
    def __init__(self, side, action, price, amount, order_type="limit", order_time=None, tier_explain=""):
        self.side = side
        self.action = action
        self.price = price
        self.amount = amount
        self.order_type = order_type
        self.order_time = order_time
        self.tier_explain = tier_explain
class Portfolio:
    def __init__(self, init_cash, margin_rate=0.05):
        self.cash = init_cash      # 可用现金
        self.margin = 0.0          # 冻结保证金
        self.position_long = 0
        self.position_short = 0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0
        self.margin_rate = margin_rate  # 保证金比例，默认0.5表示2倍杠杆
        self.history = []
        self.trade_log = []

    def update(self, price, long_change=0, short_change=0, cur_time=None, action=None):
        pnl = 0

        if long_change > 0:
            cost = price * long_change
            if self.cash >= cost:
                self.position_long += long_change
                self.avg_price_long = (self.avg_price_long * (self.position_long - long_change) + price * long_change) / self.position_long
                self.cash -= cost
                self.margin += cost  # 多头仓位保证金等于持仓成本
            else:
                print('cash not enough for long')
                return self.get_total_value(price)

        elif long_change < 0:
            close_amount = -long_change
            if close_amount > self.position_long:
                close_amount = self.position_long
            pnl = (price - self.avg_price_long) * close_amount
            self.position_long -= close_amount
            self.cash += price * close_amount + pnl
            self.margin -= self.avg_price_long * close_amount
            if self.position_long == 0:
                self.avg_price_long = 0.0

        elif short_change > 0:
            cost = price * short_change * self.margin_rate
            if self.cash >= cost:
                self.position_short += short_change
                self.avg_price_short = (self.avg_price_short * (self.position_short - short_change) + price * short_change) / self.position_short
                self.cash -= cost
                self.margin += cost  # 空头仓位保证金按比例计算
            else:
                print('cash not enough for long')
                return self.get_total_value(price)

        elif short_change < 0:
            close_amount = -short_change
            if close_amount > self.position_short:
                close_amount = self.position_short
            pnl = (self.avg_price_short - price) * close_amount
            self.position_short -= close_amount
            margin_return = self.avg_price_short * close_amount * self.margin_rate
            self.cash += margin_return + pnl
            self.margin -= margin_return
            if self.position_short == 0:
                self.avg_price_short = 0.0

        unrealized_pnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        unrealized_pnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0

        total_asset = self.cash + self.margin + unrealized_pnl_long + unrealized_pnl_short

        self.history.append(total_asset)

        if action is not None and (long_change != 0 or short_change != 0):
            from datetime import datetime
            record = {
                'datetime': datetime.fromtimestamp(cur_time) if isinstance(cur_time, (int, float)) else cur_time,
                'action': action,
                'price': price,
                'amount': long_change if long_change != 0 else short_change,
                'position_long': self.position_long,
                'position_short': self.position_short,
                'cash': self.cash,
                'margin': self.margin,
                'total_asset': total_asset,
                'pnl': pnl
            }
            self.trade_log.append(record)

        return total_asset

    def get_total_value(self, price):
        unrealized_pnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        unrealized_pnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0
        return self.cash + self.margin + unrealized_pnl_long + unrealized_pnl_short

    
def get_trend_signal(trend_df, multiVwap, window=200):
    kama_delta = trend_df['kama1'] - trend_df['kama2']
    sum_delta = kama_delta.iloc[-window:].sum()
    if trend_df['kama1'].iloc[-1] > trend_df['kama2'].iloc[-1] and sum_delta > 0:
        return 'long'
    elif trend_df['kama1'].iloc[-1] < trend_df['kama2'].iloc[-1] and sum_delta < 0:
        return 'short'
    else:
        return 'neutral'

class Strategy:
    def __init__(self, multiVwap, usdt_total):
        self.multiVwap = multiVwap
        self.usdt_total = usdt_total
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

    def generate_order_signals(self, trend_df, df, cur_close, close, position_amount_dict_sub):
        sig_long, sig_short = None, None
        all_values = sum(position_amount_dict_sub.values())
        open2equity_pct = all_values / self.usdt_total
        trend = get_trend_signal(trend_df, self.multiVwap)

        if trend == 'long':
            self.cancel_order('long')
            self.clear_order('long')
            if open2equity_pct < 0.04:
                price_prepare = trend_df['kama1'].iloc[-1]
                if self.multiVwap.SFrame_vwap_poc.iloc[-1] > trend_df['kama2'].iloc[-1]:
                    price_prepare = min(price_prepare, self.multiVwap.SFrame_vwap_poc.iloc[-1])
                price_prepare = min(price_prepare, min(df['high'].iloc[-30:]))
                sig_long = OrderSignal('long', True, price_prepare, 6, order_type='limit', tier_explain="trend long kama1 entry")
            sig_short = None

        elif trend == 'short':
            self.cancel_order('short')
            self.clear_order('short')
            if open2equity_pct < 0.04:
                price_prepare = trend_df['kama1'].iloc[-1]
                if self.multiVwap.SFrame_vwap_poc.iloc[-1] < trend_df['kama2'].iloc[-1]:
                    price_prepare = max(price_prepare, self.multiVwap.SFrame_vwap_poc.iloc[-1])
                price_prepare = max(price_prepare, max(df['high'].iloc[-30:]))
                sig_short = OrderSignal('short', True, price_prepare, 6, order_type='limit', tier_explain="trend short kama1 entry")
            sig_long = None

        else:
            # 震荡区间
            for side in ('long', 'short'):
                if self.should_cancel(side):
                    self.cancel_order(side)
                    self.clear_order(side)
            sig_short = self.evaluate('short', cur_close, close, self.multiVwap, open2equity_pct)
            if sig_short:
                sig_short.price = max(sig_short.price, max(df['high'].iloc[-30:]))
            sig_long  = self.evaluate('long',  cur_close, close, self.multiVwap, open2equity_pct)
            if sig_long:
                sig_long.price = min(sig_short.price, min(df['high'].iloc[-30:]))
        return sig_long, sig_short


import LHFrameStd
# from yyyyy2_okx_5m import trade_coin

from db_client import SQLiteWALClient
# from db_read import read_and_sort_df
from history_kline import read_and_sort_df
# from history_kline import read_and_sort_df
from db_read import resample_to
symbol = "ETH-USDT-SWAP"

DEBUG = False
# DEBUG = True

DB_PATH = f'{symbol}.db'
client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")
trade_client = None

windowConfig = LHFrameStd.WindowConfig()

LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 310)
LIMIT_K_N = 1700 + LIMIT_K_N_APPEND 
TREND_LENGTH = 116000
# TREND_LENGTH = 2000
LIMIT_K_N += TREND_LENGTH



def backtest(client, usdt_init=10000, resample_period='5min'):
    df_raw = read_and_sort_df(client, LIMIT_K_N)
   
    df = resample_to(df_raw, resample_period)

    df['datetime'] = pd.to_datetime(df.index, unit='s')
    print('data time range', df['datetime'].iloc[0], df['datetime'].iloc[-1] )

    # 初始化VWAP多周期计算器，参数你可根据需求调整
    windowConfig = LHFrameStd.WindowConfig()
    multiVwap = LHFrameStd.MultiTFvp_poc(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)

    # 计算动态KAMA
    kama_params = dict(
        src_col="close",
        len_er=200,
        fast=15,
        slow2fast_times=2.0,
        slow=1800,
        intervalP=0.01,
        minLen=10,
        maxLen=60,
        volLen=30
    )
    df_raw__ = read_and_sort_df(client, LIMIT_K_N)
    df_5m = resample_to(df_raw__.copy(), resample_period)
    df_15m = resample_to(df_raw__.copy(), '15min')
   

    df_kama = compute_dynamic_kama(df_15m, **kama_params).reindex(df_5m.reset_index(drop=True).index, method='ffill')

    # df_kama_5m = compute_dynamic_kama(df, **kama_params)
    # df_kama = compute_dynamic_kama(df_15m, **kama_params).reindex(df_kama_5m.index, method='ffill')
    
    print(f'len of df={len(df)} and kama={len(df_kama)}')
    portfolio = Portfolio(usdt_init)
    strategy = Strategy(multiVwap, usdt_init)

    position_amount_dict_sub = {'long':0, 'short':0}

    for i in range(len(df)):
        if i < 200:
            continue
        cur_index = df.index[i]
        trend_df = df_kama.iloc[i-200:i].copy()

        cur_close = df.loc[cur_index, 'close']
        close = df.loc[cur_index, 'close']

        sig_long, sig_short = strategy.generate_order_signals(trend_df, df[:cur_index], cur_close, close, position_amount_dict_sub)

        # ---- 平仓逻辑（以开仓信号反向，且持仓存在时一次性全平） ----
        # 多头平仓
        if sig_short and sig_short.action and position_amount_dict_sub['long'] > 0:
            sell_price = sig_short.price
            portfolio.update(sell_price, long_change=-position_amount_dict_sub['long'], cur_time=cur_index, action='close_long')
            position_amount_dict_sub['long'] = 0

        # 空头平仓
        if sig_long and sig_long.action and position_amount_dict_sub['short'] > 0:
            buy_price = sig_long.price
            portfolio.update(buy_price, short_change=-position_amount_dict_sub['short'], cur_time=cur_index, action='close_short')
            position_amount_dict_sub['short'] = 0

        # 多头开仓
        if sig_long and sig_long.action and position_amount_dict_sub['long'] == 0:
            portfolio.update(sig_long.price, long_change=sig_long.amount, cur_time=cur_index, action='open_long')
            position_amount_dict_sub['long'] += sig_long.amount

        # 空头开仓
        if sig_short and sig_short.action and position_amount_dict_sub['short'] == 0:
            portfolio.update(sig_short.price, short_change=sig_short.amount, cur_time=cur_index, action='open_short')
            position_amount_dict_sub['short'] += sig_short.amount

        total_val = portfolio.get_total_value(cur_close)
        portfolio.history.append(total_val)
    
    return portfolio

# 运行示例
if __name__ == "__main__":
    portfolio = backtest(client, usdt_init=10000, resample_period='5min')


    # 假设你返回了portfolio对象，可以输出交易日志
    for trade in portfolio.trade_log:
        print(trade)
    import pandas as pd

    trade_df = pd.DataFrame(portfolio.trade_log)
    trade_df.to_csv('trade_log.csv', index=False)
    
    import matplotlib.pyplot as plt
    plt.plot(portfolio.history)
    plt.title('Backtest Account Equity')
    plt.show()
