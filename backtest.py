import pandas as pd
import numpy as np
import time

# 假设你已有的模块和函数路径正确
from dynamic_kama import compute_dynamic_kama, anchored_momentum_via_kama
# multiVwap 计算相关，请确保可调用
# data 相关，read_and_sort_df, resample_to 可使用

from datetime import datetime



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
            cost = price * long_change * self.margin_rate
            if self.cash >= cost:
                self.position_long += long_change
                self.avg_price_long = (self.avg_price_long * (self.position_long - long_change) + price * long_change) / self.position_long
                self.cash -= cost
                self.margin += cost
            else:
                print(f'cash not enough for {action}')
                return self.get_total_value(price)

        elif long_change < 0:
            close_amount = -long_change
            if close_amount > self.position_long:
                close_amount = self.position_long
            pnl = (price - self.avg_price_long) * close_amount
            self.position_long -= close_amount
            margin_return = self.avg_price_long * close_amount * self.margin_rate
            self.cash += margin_return + pnl
            self.margin -= margin_return
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
                print(f'cash not enough for {action}')
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

        if cur_time is not None:
            cur_time_dt = datetime.fromtimestamp(cur_time)
        else:
            cur_time_dt = datetime.now()
        self.history.append((cur_time_dt, total_asset))

        if action is not None and (long_change != 0 or short_change != 0):
            
            record = {
                'datetime': datetime.fromtimestamp(cur_time) if isinstance(cur_time, (int, float)) else cur_time,
                'action': action,
                'price': round(price,3),
                'amount': long_change if long_change != 0 else short_change,
                'position_long': self.position_long,
                'position_short': self.position_short,
                'cash': round(self.cash, 3),
                'margin': round(self.margin, 3),
                'total_asset': round(total_asset, 3),
                'pnl': pnl
            }
            self.trade_log.append(record)
            print(record)

        return total_asset

    def get_total_value(self, price):
        unrealized_pnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        unrealized_pnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0
        return self.cash + self.margin + unrealized_pnl_long + unrealized_pnl_short

    
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

    def generate_order_signals(self, trend_df, df, cur_close, close, position_amount_dict_sub, min_amount = 2):
        sig_long, sig_short = None, None
        all_values = sum(position_amount_dict_sub.values())
        open2equity_pct = all_values / self.usdt_total
        trend = get_trend_signal(trend_df, self.multiVwap)

        if trend == 'long':
            self.cancel_order('long')
            self.clear_order('long')
            if open2equity_pct < 0.04:
                price_prepare = trend_df.iloc[-1]['kama1']
                if self.multiVwap.SFrame_vwap_poc.iloc[-1] > trend_df.iloc[-1]['kama2']:
                    price_prepare = min(price_prepare, self.multiVwap.SFrame_vwap_poc.iloc[-1])
                price_prepare = min(price_prepare, min(df['high'].iloc[-30:]))
                sig_long = OrderSignal('long', True, price_prepare, min_amount, order_type='limit', tier_explain="trend long kama1 entry")
            sig_short = None

        elif trend == 'short':
            self.cancel_order('short')
            self.clear_order('short')
            if open2equity_pct < 0.04:
                price_prepare = trend_df.iloc[-1]['kama1']
                if self.multiVwap.SFrame_vwap_poc.iloc[-1] < trend_df.iloc[-1]['kama2']:
                    price_prepare = max(price_prepare, self.multiVwap.SFrame_vwap_poc.iloc[-1])
                price_prepare = max(price_prepare, max(df['high'].iloc[-30:]))
                sig_short = OrderSignal('short', True, price_prepare, min_amount, order_type='limit', tier_explain="trend short kama1 entry")
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
        # print(f'generated {sig_long}, \t {sig_short}')
        return sig_long, sig_short





def backtest(client, usdt_init=10000, resample_period='5min'):
    df_raw = read_and_sort_df(client, LIMIT_K_N)
   
    df = resample_to(df_raw, resample_period)

    df['datetime'] = pd.to_datetime(df.index, unit='s')
    print('data time range', df['datetime'].iloc[0], df['datetime'].iloc[-1] )

    

    # 计算动态KAMA
    kama_params = dict(
        src_col="close",
        len_er=200,
        fast=15,
        second2first_times=2.0,
        slow=1800,
        intervalP=0.01,
        minLen=10,
        maxLen=60,
        volLen=30
    )

    df_raw__ = read_and_sort_df(client, LIMIT_K_N)
    df_5m = resample_to(df_raw__.copy(), resample_period)
    df_15m = resample_to(df_raw__.copy(), '15min')
   

    df_kama = compute_dynamic_kama(df_15m, **kama_params)
    assert len(df_kama) == len(df_15m), '15m kama not equal to its df'
    #df_kama = df_kama.reindex(df_5m.index, method='ffill')
    print(df.index[0], df_kama.index[0], '\t', df.index[-1], df_kama.index[-1])
    print(f'df.values[0]={df.values[0]}')
    print(f'df_5m.values[0]={df_5m.values[0]}')
    print(f'df_kama.values[0]={df_kama.values[0]}')
    print(f'df_15m.values[0]={df_15m.values[0]}')
    
    print(f"df.index[0]{  datetime.fromtimestamp(df.index[0])} == df_kama.index[2]{ datetime.fromtimestamp(df_kama.index[0])} is {df.index[2] == df_kama.index[1]} df.index[-1]{ datetime.fromtimestamp(df.index[-1])} == df_kama.index[-1]{ datetime.fromtimestamp(df_kama.index[-1])} is {df.index[-1] == df_kama.index[-1]}")
    

    for i in range(len(df_kama)):
        a = df_kama.iloc[i]['kama1']
        # 这里添加对a的检查，比如判断是否为NaN、无穷大或其他异常
        if pd.isna(a):
            print(f"Index {i} has NaN in kama1")
        elif not np.isfinite(a):
            print(f"Index {i} has non-finite value in kama1: {a}")
        # 你也可以检查是否超出合理范围，比如负值或极端值
    # df_kama = compute_dynamic_kama(df, **kama_params) #.reindex(df.index, method='ffill')

    
    print(f'len of df={len(df)} and kama={len(df_kama)}')
    portfolio = Portfolio(usdt_init)

    # 初始化VWAP多周期计算器，参数你可根据需求调整
    windowConfig = LHFrameStd.WindowConfig()
    multiVwap = LHFrameStd.MultiTFvp_poc(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)
    strategy = Strategy(multiVwap, usdt_init)

    position_amount_dict_sub = {'long':0, 'short':0}


    no_signal_count_long = 0
    no_signal_count_short = 0
    peak_value = -float('inf')  # 初始化峰值（回测开始时）
    
    for i in range(len(df)):
        if i < 400:
            continue
        
        cur_index = df.index[i]
        if i % 1000 == 0:
            print(i, cur_index)
            print('5m df time=', df['datetime'].iloc[i], '\t 15m kama time=', df_kama['datetime'].iloc[int(i/3)])
            print('5m df close=', df['close'].iloc[i], '\t 15m kama close=', df_kama['close'].iloc[int(i/3)])

        cur_close = df.loc[cur_index, 'close']
        close = df.loc[cur_index, 'close']

        sig_long, sig_short = strategy.generate_order_signals(df_kama.iloc[:int(i/3)], df.iloc[:i], cur_close, close, position_amount_dict_sub)

        # 判断是否有开仓信号
        has_long_signal = sig_long and sig_long.action
        has_short_signal = sig_short and sig_short.action
        
        # 更新无信号计数器
        if has_long_signal:
            no_signal_count_long = 0
        else:
            no_signal_count_long += 1

        if has_short_signal:
            no_signal_count_short = 0
        else:
            no_signal_count_short += 1
        
        total_val = portfolio.get_total_value(cur_close)

        # 更新峰值
        if total_val > peak_value:
            peak_value = total_val

        # 计算回撤
        drawdown = (peak_value - total_val) / peak_value if peak_value > 0 else 0
        # 如果回撤超过6%，执行全平多头和空头
        if drawdown > 0.06:
            if position_amount_dict_sub['long'] > 0:
                portfolio.update(cur_close, long_change=-position_amount_dict_sub['long'], cur_time=cur_index, action='stop_loss_long')
                position_amount_dict_sub['long'] = 0
            if position_amount_dict_sub['short'] > 0:
                portfolio.update(cur_close, short_change=-position_amount_dict_sub['short'], cur_time=cur_index, action='stop_loss_short')
                position_amount_dict_sub['short'] = 0
            # 重置峰值，避免重复触发，可以根据策略需求决定是否重置
            peak_value = total_val  
        else:
            # ---- 平仓逻辑（以开仓信号反向，且持仓存在时一次性全平） ----
            # 先执行信号驱动的平仓
            if sig_short and sig_short.action and position_amount_dict_sub['long'] > 0:
                sell_price = sig_short.price
                portfolio.update(sell_price, long_change=-position_amount_dict_sub['long'], cur_time=cur_index, action='close_long')
                position_amount_dict_sub['long'] = 0
                no_signal_count_long = 0  # 有信号，重置计数器

            if sig_long and sig_long.action and position_amount_dict_sub['short'] > 0:
                buy_price = sig_long.price
                portfolio.update(buy_price, short_change=-position_amount_dict_sub['short'], cur_time=cur_index, action='close_short')
                position_amount_dict_sub['short'] = 0
                no_signal_count_short = 0

            # 然后执行无信号衰减逻辑
            if no_signal_count_long >= 1000 and position_amount_dict_sub['long'] > 0:
                reduce_amount = position_amount_dict_sub['long'] // 2
                if reduce_amount > 0:
                    portfolio.update(cur_close, long_change=-reduce_amount, cur_time=cur_index, action='decay_long')
                    position_amount_dict_sub['long'] -= reduce_amount
                no_signal_count_long = 0

            if no_signal_count_short >= 1000 and position_amount_dict_sub['short'] > 0:
                reduce_amount = position_amount_dict_sub['short'] // 2
                if reduce_amount > 0:
                    portfolio.update(cur_close, short_change=-reduce_amount, cur_time=cur_index, action='decay_short')
                    position_amount_dict_sub['short'] -= reduce_amount
                no_signal_count_short = 0

            # 多头开仓
            maxh_short_condition = position_amount_dict_sub['long']*cur_close*portfolio.margin_rate < portfolio.get_total_value(cur_close)/10
            # maxh_short_condition = position_amount_dict_sub['long'] < 40

            if sig_long and sig_long.action and maxh_short_condition:
                portfolio.update(sig_long.price, long_change=sig_long.amount, cur_time=cur_index, action='open_long')
                position_amount_dict_sub['long'] += sig_long.amount

            # 空头开仓
            max_long_condition = position_amount_dict_sub['short']*cur_close*portfolio.margin_rate < portfolio.get_total_value(cur_close)/10
            # max_long_condition = position_amount_dict_sub['short'] < 40

            if sig_short and sig_short.action and max_long_condition:
                portfolio.update(sig_short.price, short_change=sig_short.amount, cur_time=cur_index, action='open_short')
                position_amount_dict_sub['short'] += sig_short.amount

        total_val = portfolio.get_total_value(cur_close)
        if cur_index is not None:
            cur_time_dt = datetime.fromtimestamp(cur_index)
        else:
            cur_time_dt = datetime.now()

        portfolio.history.append((cur_time_dt, total_val))
    return portfolio

# 运行示例
if __name__ == "__main__":
    portfolio = backtest(client, usdt_init=10000, resample_period='5min')


    # 假设你返回了portfolio对象，可以输出交易日志
    # for trade in portfolio.trade_log:
    #     print(trade)
    import pandas as pd

    trade_df = pd.DataFrame(portfolio.trade_log)
    trade_df.to_csv('trade_log.csv', index=False)
    
    import matplotlib.pyplot as plt
    # 转换成 DataFrame，设置时间索引
    df_history = pd.DataFrame(portfolio.history, columns=['datetime', 'total_asset'])
    df_history.set_index('datetime', inplace=True)

    # 绘制曲线，matplotlib 和 pandas 配合很方便
    df_history['total_asset'].plot()

    plt.title('Backtest Account Equity')
    plt.xlabel('Time')
    plt.ylabel('Total Asset')
    plt.grid(True)
    plt.show()
