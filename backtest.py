import pandas as pd
import numpy as np
import time

# 假设你已有的模块和函数路径正确
from indicators import compute_dynamic_kama, anchored_momentum_via_kama
# multiVwap 计算相关，请确保可调用
# data 相关，read_and_sort_df, resample_to 可使用
from datetime import datetime, timedelta

import indicators as LHFrameStd
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
        self.fee_rate = 15/10000
        self.cash = init_cash      # 可用现金
        self.margin = 0.0          # 冻结保证金
        self.position_long = 0
        self.position_short = 0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0
        self.margin_rate = margin_rate  # 保证金比例，默认0.5表示2倍杠杆
        self.history = []
        self.trade_log = []
        
        self.total_asset = init_cash

    def upnl(self, price):
        upnl_long  = (price - self.avg_price_long)  * self.position_long if self.position_long  > 0 else 0.0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0.0
        return upnl_long + upnl_short

    def free_margin(self, price):
        # 全仓：可用保证金 = 现金 + 未实现盈亏（初始保证金已从现金锁定到 self.margin）
        return self.cash + self.upnl(price)

    def update(self, price, long_change=0, short_change=0, cur_time=None, action=None):
        rpnl = 0  # 已实现盈亏（平仓时生效）

        if long_change > 0:
            fee_rate = getattr(self, "fee_rate", 0.0)  # 若无手续费，可不设置或设为0
            cost = price * long_change * self.margin_rate   # 初始保证金（固定锁定）
            fee  = price * long_change * fee_rate           # 手续费（可选）
            free = self.free_margin(price)
            need = cost + fee

            if free >= need:
                prev = self.position_long
                self.position_long += long_change
                # 加权平均开仓价（用于后续UPnL；保证金基数也随均价/仓位变化）
                self.avg_price_long = ((self.avg_price_long * prev) + price * long_change) / self.position_long

                # 记账：锁定保证金到 margin（固定），现金扣除 保证金+手续费
                self.cash   -= need
                self.margin += cost
            else:
                print(f'Free margin not enough for {action} (long): need {need:.3f}, free={free:.3f}')
                return self.get_total_value(price)

        elif long_change < 0:  # 平多头仓
            close_amount = -long_change
            if close_amount > self.position_long:
                close_amount = self.position_long
            rpnl = (price - self.avg_price_long) * close_amount
            self.position_long -= close_amount
            margin_return = self.avg_price_long * close_amount * self.margin_rate
            self.cash += margin_return + rpnl
            self.margin -= margin_return

            if self.position_long == 0:
                self.avg_price_long = 0.0

        if short_change > 0:
            # 可选：手续费
            fee_rate = getattr(self, "fee_rate", 0.0)
            cost = price * short_change * self.margin_rate         # 初始保证金（固定锁定）
            fee  = price * short_change * fee_rate                 # 手续费（若有）

            free = self.free_margin(price)
            need = cost + fee
            if free >= need:
                prev = self.position_short
                self.position_short += short_change
                # 加权平均开仓价（用于计算UPnL；注意：均价改变 => 未来“固定保证金”的基数也随之改变）
                self.avg_price_short = ((self.avg_price_short * prev) + price * short_change) / self.position_short

                # 记账：锁定保证金到 margin（固定），现金扣除保证金+手续费
                self.cash   -= need
                self.margin += cost
            else:
                print(f'Free margin not enough for {action} (short): need {need:.3f}, free={free:.3f}')
                return self.get_total_value(price)

        elif short_change < 0:  # 平空头仓
            close_amount = -short_change
            if close_amount > self.position_short:
                close_amount = self.position_short
            rpnl = (self.avg_price_short - price) * close_amount
            self.position_short -= close_amount
            margin_return = self.avg_price_short * close_amount * self.margin_rate
            self.cash += margin_return + rpnl
            self.margin -= margin_return
            if self.position_short == 0:
                self.avg_price_short = 0.0

        # 未实现盈亏计算
        upnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0

        # 总资产=现金+冻结保证金+未实现盈亏
        self.total_asset = self.cash + self.margin + upnl_long + upnl_short

        # 时间处理
        if cur_time is not None:
            cur_time_dt = datetime.utcfromtimestamp(cur_time) if isinstance(cur_time, (int, float)) else cur_time
        else:
            cur_time_dt = datetime.now()
        self.history.append((cur_time_dt, self.total_asset))

        # 记录交易日志（只在有动作时记录）
        if action is not None and (long_change != 0 or short_change != 0):
            record = {
                'datetime': cur_time_dt,
                'action': action,
                'price': round(price, 3),
                'amount': round(long_change if long_change != 0 else short_change, 2),
                'position_long': round(self.position_long, 2),
                'position_short': round(self.position_short, 2),
                'cash': round(self.cash, 3),
                'margin': round(self.margin, 3),
                'asset': round(self.total_asset, 3),
                'rpnl': round(rpnl, 3),
                'upnl_long': round(upnl_long, 3),
                'upnl_short': round(upnl_short, 3),
            }
            self.trade_log.append(record)
            # if rpnl != 0:
            print(record)

        return self.total_asset

    def get_total_value(self, price):
        upnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0
        return self.cash + self.margin + upnl_long + upnl_short

    
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


def backtest(client, usdt_init=10000, resample_period='5min'):
    df_raw = read_and_sort_df(client, LIMIT_K_N)
   
    df = resample_to(df_raw, resample_period)

    # df['datetime'] = pd.to_datetime(df.index, unit='s')
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
    df_5m = resample_to(df_raw__.copy(deep=True), resample_period)
    df_15m = resample_to(df_raw__.copy(deep=True), '15min')
    print('df_15m.index[0]', df_15m.index[0])
    print()

    df_kama = compute_dynamic_kama(df_15m, **kama_params)
    assert len(df_kama) == len(df_15m), '15m kama not equal to its df'
    df_kama = df_kama.reindex(df_15m.index, method='ffill')
    print("df.index[0], df_kama.index[0],", df.index[0], df_kama.index[0], '\t', "df.index[-1], df_kama.index[-1]", df.index[-1], df_kama.index[-1])
    print()

    print("len of df(5m) and df_kama", len(df), len(df_kama))
    print()

    print(f'df.values[0]={df.values[0]}')
    print(f'df_5m.values[0]={df_5m.values[0]}')
    print('kama head3:', df_kama.head(3))

    print('df(5m) head3:', df.head(3))


    print()
    print(f'df_kama.values[3]={df_kama.values[3]}')
    print(f'df_kama.values[-1]={df_kama.values[-1]}')
    print(f'df_15m.values[0]={df_15m.values[0]}')
    
    print(f"df.index[0]{  datetime.utcfromtimestamp(df.index[0])} == df_kama.index[2]{ datetime.utcfromtimestamp(df_kama.index[0])} is {df.index[2] == df_kama.index[1]} df.index[-1]{ datetime.utcfromtimestamp(df.index[-1])} == df_kama.index[-1]{ datetime.utcfromtimestamp(df_kama.index[-1])} is {df.index[-1] == df_kama.index[-1]}")
    

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
    multiVwap = LHFrameStd.MultiTFVWAP(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)

    multiVwap_15m = LHFrameStd.MultiTFVWAP(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap_15m.calculate_SFrame_vwap_poc_and_std(df_15m, DEBUG)
    

    position_amount_dict_sub = {'long':0, 'short':0}
    no_signal_count_short,no_signal_count_long = 0, 0
    peak_value = -float('inf')  # 初始化峰值（回测开始时）
    kama_begin_require = 5000
    assert kama_begin_require < len(df), f'df length not enough to kama_begin_require, which is {kama_begin_require}'

    reached_to_sl2 = False
    from_open_kline_counter = 0
    for i in range(kama_begin_require, len(df), 1):        
        cur_index = df.index[i]
        kama_slice = df_kama.iloc[:int((i+1)/3)]
        df_slice = df.iloc[:i+1]

        # if i % 10000 == 0:
        #     print('Dialog df not future function:', df_slice, kama_slice , 'df 5m is latter:', df.index[i+1] >= df_kama.index[int((i+1)/3)])

        cur_close = df.loc[cur_index, 'close']
        assert(cur_close == df['close'].iloc[i]), 'loc value not equal to iloc value'
        closes = df['close'].iloc[:i+1]
        
        total_value = portfolio.get_total_value(cur_close)
        base_value = 10_000
        base_amount = 0.2

        # 计算倍数，根据总价值按base_value递增
        multiplier = total_value // base_value  # 取整倍数

        # 最小开仓量，防止为0
        min_amount = min(200, max(base_amount * int(multiplier), base_amount))

        
        strategy = Strategy(multiVwap, portfolio.cash/cur_close/portfolio.margin_rate)
        sig_long, sig_short = strategy.generate_order_signals(kama_slice, df_slice, cur_index, closes, position_amount_dict_sub, min_amount=min_amount)

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
        df_15m_pos = df_15m.index.searchsorted(cur_index, side="left")
        if df_15m_pos == len(df_15m) :
            print('df_15m_index == None')
            reached_to_sl2 = position_amount_dict_sub['long'] > 0 or position_amount_dict_sub['short'] > 0
        else:
            df_15m_index = df_15m.index[df_15m_pos]
            if ((position_amount_dict_sub['long'] > 0 and cur_close >= multiVwap_15m.SFrame_vwap_poc.loc[df_15m_index])  or\
                (position_amount_dict_sub['short'] > 0 and cur_close <= multiVwap_15m.SFrame_vwap_poc.loc[df_15m_index])):
                reached_to_sl2 = True
            else:
                reached_to_sl2 = False
        reached_to_sl2 = True
        # 更新峰值
        if total_val > peak_value:
            peak_value = total_val
            # print(f'{ datetime.utcfromtimestamp(cur_index) } peak value caused by price {cur_close}')

        # 计算回撤
        drawdown = (peak_value - total_val) / peak_value if peak_value > 0 else 0
        # 如果回撤超过6%，执行全平多头和空头
        dynamic_drawdown_thres = 0.01

        if (drawdown > dynamic_drawdown_thres and reached_to_sl2) or drawdown > 0.06:
            if position_amount_dict_sub['long'] > 0:
                stop_loss_price = df.iloc[i]['close']
                if stop_loss_price <= max(df.iloc[i:min(i+160, len(df))]['high']):
                    reduce_amount = position_amount_dict_sub['long'] 
                    portfolio.update(stop_loss_price, long_change=-reduce_amount, cur_time=cur_index, action='stop_loss_long')
                    position_amount_dict_sub['long'] -= reduce_amount
                    reached_to_sl2 = False
                else:
                    print('**' * 8, 'unable to stop loss long because future max high is lower than required close')
            if position_amount_dict_sub['short'] > 0:
                stop_loss_price = df.iloc[i]['close']
                if stop_loss_price >= min(df.iloc[i:min(i+160, len(df))]['low']):
                    reduce_amount = position_amount_dict_sub['short']
                    portfolio.update(stop_loss_price, short_change=-reduce_amount, cur_time=cur_index, action='stop_loss_short')
                    position_amount_dict_sub['short'] -= reduce_amount
                    reached_to_sl2 = False
                else:
                    print('**' * 8, 'unable to stop loss short because future min low is higher than required close')
            # 重置峰值，避免重复触发，可以根据策略需求决定是否重置
            total_val = portfolio.get_total_value(cur_close)
            peak_value = total_val  

            #清空无信号计数器
            no_signal_count_short, no_signal_count_long = 0, 0
        # elif holding_profit_more_than_2pct():
        #     leave_profit_keep_going()
        else:
            # ---- 平仓逻辑（以开仓信号反向，且持仓存在时一次性全平） ----
            # 先执行信号驱动的平仓
            if sig_short and sig_short.action and position_amount_dict_sub['long'] > 0:
                future_highs = df.iloc[i+1:min(i+160, len(df))]['high']
                if sig_short.price > max(future_highs):
                    sell_price = sig_short.price
                    portfolio.update(sell_price, long_change=-position_amount_dict_sub['long'], cur_time=cur_index, action='close_long')
                    position_amount_dict_sub['long'] = 0
                    no_signal_count_long = 0  # 有信号，重置计数器
                    dynamic_drawdown_thres = 0.01
                    reached_to_sl2 = False

            if sig_long and sig_long.action and position_amount_dict_sub['short'] > 0:
                future_lows = df.iloc[i+1:min(i+160, len(df))]['low']
                if sig_long.price > min(future_lows):
                    buy_price = sig_long.price

                    portfolio.update(buy_price, short_change=-position_amount_dict_sub['short'], cur_time=cur_index, action='close_short')
                    position_amount_dict_sub['short'] = 0
                    no_signal_count_short = 0
                    dynamic_drawdown_thres = 0.01
                    reached_to_sl2 = False

            # 然后执行无信号衰减逻辑
            auto_decay = True
            auto_decay = False
            if auto_decay:
                if no_signal_count_long >= (60/5) * 24 * 0.5 and position_amount_dict_sub['long'] > 0:
                    #衰减仓位时用position_amount_dict_sub['long'] // 2，减半处理合理，但如果仓位为1时会变0，导致永远无法衰减到0。
                    reduce_amount = position_amount_dict_sub['long'] if position_amount_dict_sub['long'] < 3 else position_amount_dict_sub['long'] // 3
                    if reduce_amount > 0:
                        portfolio.update(cur_close, long_change=-reduce_amount, cur_time=cur_index, action='decay_long')
                        position_amount_dict_sub['long'] -= reduce_amount
                    no_signal_count_long = 0
                    dynamic_drawdown_thres = 0.01

                if no_signal_count_short >= (60/5) * 24 * 0.5  and position_amount_dict_sub['short'] > 0:
                    #衰减仓位时用position_amount_dict_sub['long'] // 2，减半处理合理，但如果仓位为1时会变0，导致永远无法衰减到0。
                    reduce_amount = position_amount_dict_sub['short'] if position_amount_dict_sub['short'] < 3 else position_amount_dict_sub['short'] // 3
                    if reduce_amount > 0:
                        portfolio.update(cur_close, short_change=-reduce_amount, cur_time=cur_index, action='decay_short')
                        position_amount_dict_sub['short'] -= reduce_amount
                    no_signal_count_short = 0
                    dynamic_drawdown_thres = 0.01

            equity = portfolio.get_total_value(cur_close)

            # 多头开仓
            max_long_condition = position_amount_dict_sub['long']*cur_close*portfolio.margin_rate < equity/10
            max_long_condition = True
            # max_long_condition = position_amount_dict_sub['long'] < 40
            
            if sig_long and sig_long.action and max_long_condition:
                future_df = df.iloc[i+1:min(i+160, len(df))]
                if len(future_df) == 0 or sig_long.price > min(future_df['low']):  #low比开仓价格还低，才可能进得去多头
                    portfolio.update(sig_long.price, long_change=sig_long.amount, cur_time=cur_index, action='open_long')
                    position_amount_dict_sub['long'] += sig_long.amount
                    
                    if from_open_kline_counter > 60:
                        dynamic_drawdown_thres = 0.005
                    else:
                        dynamic_drawdown_thres = 0.01
                    if 0:
                        print('sig_long', df.loc[cur_index], 'sig_long',sig_long, 'sl2:', multiVwap.SFrame_vwap_down_sl2.loc[cur_index])
                        print(multiVwap.SFrame_vwap_down_sl2.loc[df.index[i-1]],
                            multiVwap.SFrame_vwap_down_sl2.loc[df.index[i+1]], 
                            multiVwap.SFrame_vwap_down_sl2.loc[df.index[i+2]],
                            multiVwap.SFrame_vwap_down_sl2.loc[df.index[i+3]],
                            multiVwap.SFrame_vwap_down_sl2.loc[df.index[i+4]],)
                        print('beggining', df.iloc[0])
                        break
                else:
                    pass
                    #print('++'*10, datetime.utcfromtimestamp(cur_index), f'long limit={round(sig_long.price, 2) } not touched')
                from_open_kline_counter += 1

            # 空头开仓
            maxh_short_condition = position_amount_dict_sub['short']*cur_close*portfolio.margin_rate < equity/10
            maxh_short_condition = True
            # maxh_short_condition = position_amount_dict_sub['short'] < 40

            if sig_short and sig_short.action and maxh_short_condition:
                future_df = df.iloc[i+1:min(i+160, len(df))]
                if len(future_df) == 0 or sig_short.price < max(future_df['high']):#high比开仓价格还高，才可能进得去空头
                    portfolio.update(sig_short.price, short_change=sig_short.amount, cur_time=cur_index, action='open_short')
                    position_amount_dict_sub['short'] += sig_short.amount

                    if from_open_kline_counter > 60:
                        dynamic_drawdown_thres = 0.005
                    else:
                        dynamic_drawdown_thres = 0.01

                else:
                    pass
                    #print('--'*10, datetime.utcfromtimestamp(cur_index), f'short limit={round(sig_short.price, 2) } not touched')
                    
                from_open_kline_counter += 1

        total_val = portfolio.get_total_value(cur_close)
        if cur_index is not None:
            cur_time_dt = datetime.utcfromtimestamp(cur_index)
        else:
            cur_time_dt = datetime.now()

        portfolio.history.append((cur_time_dt, total_val))

    return portfolio, peak_value


import matplotlib.pyplot as plt

# # 运行示例
# if __name__ == "__main__":
#     portfolio,peak_value = backtest(client, usdt_init=10000, resample_period='5min')

#     print('peak_value', peak_value)
#     # 假设你返回了portfolio对象，可以输出交易日志
#     # for trade in portfolio.trade_log:
#     #     print(trade)
#     import pandas as pd

#     trade_df = pd.DataFrame(portfolio.trade_log)
#     trade_df.to_csv('trade_log.csv', index=False)
    
#     
#     # 转换成 DataFrame，设置时间索引
#     df_history = pd.DataFrame(portfolio.history, columns=['datetime', 'total_asset'])
#     df_history.set_index('datetime', inplace=True)

#     # 绘制曲线，matplotlib 和 pandas 配合很方便
#     df_history['total_asset'].plot()

#     plt.title('Backtest Account Equity')
#     plt.xlabel('Time')
#     plt.ylabel('Total Asset')
#     plt.grid(True)
#     plt.show()

def render_trades_with_price(df_price, manager):
    trade_df = manager.load_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe='5min')

    # 确保索引是datetime
    if not pd.api.types.is_datetime64_any_dtype(df_price.index):
        df_price.index = pd.to_datetime(df_price.index)
    if not pd.api.types.is_datetime64_any_dtype(trade_df.index):
        trade_df.index = pd.to_datetime(trade_df.index)

    plt.figure(figsize=(14*3, 7))
    plt.plot(df_price.index, df_price['close'], label='Close Price', lw=1)

    # 直接用交易日志时间点，找对应价格绘制信号
    def get_price_at_trade_times(trade_times):
        # 对应价格用行情数据重采样或最近时间点价格
        return df_price['close'].reindex(trade_times, method='nearest')

    # 多头开仓
    open_long_mask = trade_df['action'].str.contains('open_long', na=False)
    open_long_times = trade_df.index[open_long_mask]
    plt.scatter(open_long_times, get_price_at_trade_times(open_long_times),
                marker='^', color='g', label='Open Long', s=30)

    # 多头平仓
    close_long_mask = trade_df['action'].str.contains('close_long|stop_loss_long|decay_long', na=False)
    close_long_times = trade_df.index[close_long_mask]
    print('count of close long:', len(close_long_times))
    plt.scatter(close_long_times, get_price_at_trade_times(close_long_times),
                marker='v', color='r', label='Close Long', s=30)

    # 空头开仓
    open_short_mask = trade_df['action'].str.contains('open_short', na=False)
    open_short_times = trade_df.index[open_short_mask]
    plt.scatter(open_short_times, get_price_at_trade_times(open_short_times),
                marker='^', color='blue', label='Open Short', s=30)

    # 空头平仓
    close_short_mask = trade_df['action'].str.contains('close_short|stop_loss_short|decay_short', na=False)
    close_short_times = trade_df.index[close_short_mask]
    print('count of close short:', len(close_short_times))
    plt.scatter(close_short_times, get_price_at_trade_times(close_short_times),
                marker='v', color='orange', label='Close Short', s=30)

    plt.title('Price and Trade Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()



import pandas as pd
from trade_log_manager import TradeLogManager

# 使用示例：
if __name__ == '__main__':
    # 假设 backtest 已经跑完，portfolio 和 df 是你的回测结果和行情数据
    # trade_log 存在 portfolio.trade_log 里面
    portfolio, peak_value = backtest(client, usdt_init=10000, resample_period='5min')

    df_raw = read_and_sort_df(client, LIMIT_K_N)
    df = resample_to(df_raw, '5min')
    df.index = pd.to_datetime(df.index, unit='s')

    # 假设 portfolio.trade_log 是你的交易记录列表（字典列表）
    manager = TradeLogManager(base_path='./my_trade_data')

    # 保存交易日志
    manager.save_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe='5min', trade_log=portfolio.trade_log)

    # render_trades_with_price(df, manager)

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