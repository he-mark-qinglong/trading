import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 假设你已有的模块和函数路径正确
from indicators import compute_dynamic_kama, anchored_momentum_via_kama
from strategy import Strategy, OrderSignal, Portfolio
import indicators as LHFrameStd
from db_read import resample_to



from db_client import SQLiteWALClient
# from db_read import read_and_sort_df
from history_kline import read_and_sort_df


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
        sig_long: OrderSignal
        sig_short: OrderSignal
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
                if sig_short.price <= max(future_highs):
                    sell_price = sig_short.price
                    portfolio.update(sell_price, long_change=-position_amount_dict_sub['long'], cur_time=cur_index, action='close_long')
                    position_amount_dict_sub['long'] = 0
                    no_signal_count_long = 0  # 有信号，重置计数器
                    dynamic_drawdown_thres = 0.01
                    reached_to_sl2 = False

            if sig_long and sig_long.action and position_amount_dict_sub['short'] > 0:
                future_lows = df.iloc[i+1:min(i+160, len(df))]['low']
                if sig_long.price >= min(future_lows):
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

    return df, portfolio, peak_value


from strategy_viz import render_equity_change, render_trades_with_price
from strategy_logs import TradeLogManager

# 使用示例：
if __name__ == '__main__':
    resample_period='5min'
    manager = TradeLogManager(base_path='./my_trade_data')

    use_local_log = False

    if use_local_log == False:
        df, portfolio, peak_value = backtest(client, usdt_init=10000, resample_period=resample_period)
        df.index = pd.to_datetime(df.index, unit='s')

        # 保存交易日志
        manager.save_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe=resample_period, trade_log=portfolio.trade_log)
    else:
        df_raw__ = read_and_sort_df(client, LIMIT_K_N)
        df = resample_to(df_raw__.copy(deep=True), resample_period)
    trade_df = manager.load_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe=resample_period)
    
    render_trades_with_price(df, trade_df)
    # render_equity_change(portfolio.history)
    
    