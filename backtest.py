import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC

# 假设你已有的模块和函数路径正确
from indicators import compute_dynamic_kama, anchored_momentum_via_kama
from strategy import Strategy, OrderSignal, Portfolio
import indicators as LHFrameStd
from db_read import resample_to
from strategy import OrderManager


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
        slow=365,
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
    
    print(f"df.index[0]{  datetime.fromtimestamp(df.index[0], tz=UTC)} == df_kama.index[2]{ datetime.fromtimestamp(df_kama.index[0],tz=UTC)} is {df.index[2] == df_kama.index[1]} df.index[-1]{ datetime.fromtimestamp(df.index[-1], tz=UTC)} == df_kama.index[-1]{ datetime.fromtimestamp(df_kama.index[-1], tz=UTC)} is {df.index[-1] == df_kama.index[-1]}")
    

    for i in range(len(df_kama)):
        a = df_kama.iloc[i]['kama1']
        # 这里添加对a的检查，比如判断是否为NaN、无穷大或其他异常
        if pd.isna(a):
            print(f"Index {i} has NaN in kama1")
        elif not np.isfinite(a):
            print(f"Index {i} has non-finite value in kama1: {a}")
        # 你也可以检查是否超出合理范围，比如负值或极端值
    
    print(f'len of df={len(df)} and kama={len(df_kama)}')
    portfolio = Portfolio(usdt_init)
    
    # 初始化订单管理器，挂单超过100根自动过期
    order_mgr = OrderManager(ttl_bars=100)

    # 初始化VWAP多周期计算器，参数你可根据需求调整
    windowConfig = LHFrameStd.WindowConfig()
    multiVwap = LHFrameStd.MultiTFVWAP(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap.calculate_SFrame_vwap_poc_and_std(df, DEBUG)

    multiVwap_15m = LHFrameStd.MultiTFVWAP(windowConfig.window_tau_l, windowConfig.window_tau_h, windowConfig.window_tau_s)
    multiVwap_15m.calculate_SFrame_vwap_poc_and_std(df_15m, DEBUG)
    
    position_amount_dict_sub = {'long':0, 'short':0}
    no_signal_count_short, no_signal_count_long = 0, 0
    peak_value = -float('inf')  # 初始化峰值（回测开始时）
    kama_begin_require = 5000
    assert kama_begin_require < len(df), f'df length not enough to kama_begin_require, which is {kama_begin_require}'

    reached_to_sl2 = False
    from_open_kline_counter = 0
    open_value = 0
    
    for i in range(kama_begin_require, len(df), 1):        
        cur_index = df.index[i]
        bar_row = df.iloc[i]
        
        # ===== 1. 先处理之前挂着的单 =====
        filled_now = order_mgr.on_bar(i, cur_index, bar_row, position_amount_dict_sub['long'], position_amount_dict_sub['short'])
        for od in filled_now:
            if od.action == "open_long":
                portfolio.update(od.filled_price, long_change=od.qty, cur_time=od.filled_ts, action='open_long')
                position_amount_dict_sub['long'] += od.qty
                if from_open_kline_counter > 60:
                    dynamic_drawdown_thres = 0.005
                else:
                    dynamic_drawdown_thres = 0.01
                from_open_kline_counter += 1
            elif od.action == "open_short":
                portfolio.update(od.filled_price, short_change=od.qty, cur_time=od.filled_ts, action='open_short')
                position_amount_dict_sub['short'] += od.qty
                if from_open_kline_counter > 60:
                    dynamic_drawdown_thres = 0.005
                else:
                    dynamic_drawdown_thres = 0.01
                from_open_kline_counter += 1
            elif od.action == "close_long":
                portfolio.update(od.filled_price, long_change=-od.qty, cur_time=od.filled_ts, action='close_long')
                position_amount_dict_sub['long'] -= od.qty
                no_signal_count_long = 0
                dynamic_drawdown_thres = 0.01
                reached_to_sl2 = False
            elif od.action == "close_short":
                portfolio.update(od.filled_price, short_change=-od.qty, cur_time=od.filled_ts, action='close_short')
                position_amount_dict_sub['short'] -= od.qty
                no_signal_count_short = 0
                dynamic_drawdown_thres = 0.01
                reached_to_sl2 = False
            elif od.action == "stop_loss_long":
                portfolio.update(od.filled_price, long_change=-od.qty, cur_time=od.filled_ts, action='stop_loss_long')
                position_amount_dict_sub['long'] -= od.qty
                reached_to_sl2 = False
                print('stop loss long closed')
            elif od.action == "stop_loss_short":
                portfolio.update(od.filled_price, short_change=-od.qty, cur_time=od.filled_ts, action='stop_loss_short')
                position_amount_dict_sub['short'] -= od.qty
                reached_to_sl2 = False
            if (od.action == 'stop_loss_short' or od.action == 'stop_loss_long' or od.action == 'close_long' or od.action == "close_short") and (position_amount_dict_sub['long'] == 0 or position_amount_dict_sub['short'] == 0):
                open_value = 0
                print(f'open_value recover to 0， because od.action=={od.action}')
            elif od.action == 'open_long' or od.action == 'open_short':
                if open_value == 0:
                    open_value = portfolio.get_total_value(cur_close)
                    print(f'open_value = {open_value}')

        profit = 0 if open_value == 0 else  total_val/open_value - 1

        # ===== 2. 正常策略逻辑 =====
        kama_slice = df_kama.iloc[:int((i+1)/3)]
        df_slice = df.iloc[:i+1]

        cur_close = df.loc[cur_index, 'close']
        assert(cur_close == df['close'].iloc[i]), 'loc value not equal to iloc value'
        closes = df['close'].iloc[:i+1]
        
        total_value = portfolio.get_total_value(cur_close)
        base_value = 10_000
        base_amount = 0.05

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
        if df_15m_pos == len(df_15m) :  #not found cur_index of df_5m in df_15m
            print('df_15m_index == None')
            reached_to_sl2 = position_amount_dict_sub['long'] > 0 or position_amount_dict_sub['short'] > 0
        else:
            df_15m_index = df_15m.index[df_15m_pos]
            if ((position_amount_dict_sub['long'] > 0 and cur_close >= multiVwap_15m.SFrame_vwap_up_sl2.loc[df_15m_index])  or\
                (position_amount_dict_sub['short'] > 0 and cur_close <= multiVwap_15m.SFrame_vwap_down_sl2.loc[df_15m_index])):
                reached_to_sl2 = True
                print('reached to sl2', f"current profit={profit * 100}/%")

        # 更新峰值
        if total_val > peak_value:
            peak_value = total_val
            # print(f'{ datetime.utcfromtimestamp(cur_index) } peak value caused by price {cur_close}')

        # 计算回撤
        drawdown = (peak_value - total_val) / peak_value if peak_value > 0 else 0
        # 如果回撤超过6%，执行全平多头和空头
        dynamic_drawdown_thres = 0.03
        
        # ===== 3. 风控止损/平仓逻辑 =====
        cur_kama1 = kama_slice['kama1'].iloc[-1] 
        cur_kama2 = kama_slice['kama2'].iloc[-1] 
        trend_reverse = (cur_kama1 > cur_kama2 and position_amount_dict_sub['short'] > 0) or (cur_kama2  and cur_kama2 and position_amount_dict_sub['long'] > 0) 
        if (False and drawdown > dynamic_drawdown_thres and reached_to_sl2) or profit > 0.4 or (reached_to_sl2 and profit > 0.2) or trend_reverse:
            
            if position_amount_dict_sub['long'] > 0:
                stop_loss_price = df.iloc[i]['close']
                # 不看未来，直接挂止损单（当前价位）
                order_mgr.submit_limit(
                    action="stop_loss_long", #f"stop_loss_long_{reached_to_sl2}", 
                    side="long", 
                    price=stop_loss_price, 
                    qty=position_amount_dict_sub['long'],
                    created_pos=i, 
                    created_ts=cur_index,
                    note="drawdown_stop_loss"
                )
                print(f'condition meet, stop_loss_long putted, {profit}')
            if position_amount_dict_sub['short'] > 0:
                stop_loss_price = df.iloc[i]['close']
                order_mgr.submit_limit(
                    action="stop_loss_short", #f"stop_loss_short_{reached_to_sl2}", 
                    side="short", 
                    price=stop_loss_price, 
                    qty=position_amount_dict_sub['short'],
                    created_pos=i, 
                    created_ts=cur_index,
                    note="drawdown_stop_loss"
                )
                print(f'condition meet, stop_loss_short putted {profit}')
            
            # 重置峰值，避免重复触发，可以根据策略需求决定是否重置
            total_val = portfolio.get_total_value(cur_close)
            peak_value = total_val  

            #清空无信号计数器
            no_signal_count_short, no_signal_count_long = 0, 0
        
        else:
            # ===== 4. 正常交易逻辑 =====
            
            # 先执行信号驱动的平仓
            if sig_short and sig_short.action and position_amount_dict_sub['long'] > 0:
                # 有空头信号且有多头持仓，挂平多单
                order_mgr.submit_limit(
                    action="close_long", 
                    side="long", 
                    price=sig_short.price, 
                    qty=min(position_amount_dict_sub['long'], min_amount*2),
                    created_pos=i, 
                    created_ts=cur_index,
                    note="signal_close_long"
                )

            if sig_long and sig_long.action and position_amount_dict_sub['short'] > 0:
                # 有多头信号且有空头持仓，挂平空单
                order_mgr.submit_limit(
                    action="close_short", 
                    side="short", 
                    price=sig_long.price, 
                    qty=min(min_amount*2,position_amount_dict_sub['short']),
                    created_pos=i, 
                    created_ts=cur_index,
                    note="signal_close_short"
                )

            # 然后执行无信号衰减逻辑
            auto_decay = False  # 关闭衰减逻辑
            if auto_decay:
                if no_signal_count_long >= (60/5) * 24 * 0.5 and position_amount_dict_sub['long'] > 0:
                    reduce_amount = position_amount_dict_sub['long'] if position_amount_dict_sub['long'] < 3 else position_amount_dict_sub['long'] // 3
                    if reduce_amount > 0:
                        # 挂平多单（部分）
                        order_mgr.submit_limit(
                            action="close_long", 
                            side="long", 
                            price=cur_close, 
                            qty=reduce_amount, 
                            created_pos=i, 
                            created_ts=cur_index,
                            note="decay_long"
                        )
                    no_signal_count_long = 0
                    dynamic_drawdown_thres = 0.01

                if no_signal_count_short >= (60/5) * 24 * 0.5 and position_amount_dict_sub['short'] > 0:
                    reduce_amount = position_amount_dict_sub['short'] if position_amount_dict_sub['short'] < 3 else position_amount_dict_sub['short'] // 3
                    if reduce_amount > 0:
                        # 挂平空单（部分）
                        order_mgr.submit_limit(
                            action="close_short", 
                            side="short", 
                            price=cur_close, 
                            qty=reduce_amount, 
                            created_pos=i, 
                            created_ts=cur_index,
                            note="decay_short"
                        )
                    no_signal_count_short = 0
                    dynamic_drawdown_thres = 0.01

            equity = portfolio.get_total_value(cur_close)

            # 多头开仓
            max_long_condition = position_amount_dict_sub['long']*cur_close*portfolio.margin_rate < equity/10
            max_long_condition = True  # 覆盖为True
            
            if sig_long and sig_long.action and max_long_condition:
                # 直接挂多头限价单
                order_mgr.submit_limit(
                    action="open_long", 
                    side="long", 
                    price=sig_long.price, 
                    qty=sig_long.amount, 
                    created_pos=i, 
                    created_ts=cur_index,
                    note="signal_open_long"
                )

            # 空头开仓
            maxh_short_condition = position_amount_dict_sub['short']*cur_close*portfolio.margin_rate < equity/10
            maxh_short_condition = True  # 覆盖为True

            if sig_short and sig_short.action and maxh_short_condition:
                # 直接挂空头限价单
                order_mgr.submit_limit(
                    action="open_short", 
                    side="short", 
                    price=sig_short.price, 
                    qty=sig_short.amount, 
                    created_pos=i, 
                    created_ts=cur_index,
                    note="signal_open_short"
                )

        # 记录每个时间点的市值
        if cur_index is not None:
            cur_time_dt = datetime.fromtimestamp(cur_index, tz=UTC)  #datetime.utcfromtimestamp(cur_index) 
        else:
            cur_time_dt = datetime.now()

        portfolio.history.append((cur_time_dt, total_val))

    return df, portfolio, peak_value, multiVwap_15m, df_kama


from strategy_viz import render_equity_change, render_trades_with_price
from strategy_logs import TradeLogManager

# 使用示例：
if __name__ == '__main__':
    resample_period='5min'
    manager = TradeLogManager(base_path='./my_trade_data')

    use_local_log = False

    if not use_local_log:
        df, portfolio, peak_value, multiVwap, df_kama = backtest(client, usdt_init=10000, resample_period=resample_period)
        df.index = pd.to_datetime(df.index, unit='s')

        # 保存交易日志
        manager.save_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe=resample_period, trade_log=portfolio.trade_log)
    # else:
    #     df_raw__ = read_and_sort_df(client, LIMIT_K_N)
    #     df = resample_to(df_raw__.copy(deep=True), resample_period)
    trade_df = manager.load_trade_log(exchange_id='okx', symbol='ETH/USDT', timeframe=resample_period)
    
    
    render_equity_change(portfolio.history)
    render_trades_with_price(df, trade_df, multiVwap, df_kama)
    
    