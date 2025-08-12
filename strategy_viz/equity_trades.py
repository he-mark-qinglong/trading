import pandas as pd
import matplotlib.pyplot as plt

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

def render_equity_change(history):
    # 转换成 DataFrame，设置时间索引
    df_history = pd.DataFrame(history, columns=['datetime', 'total_asset'])
    df_history.set_index('datetime', inplace=True)
    # 绘制曲线，matplotlib 和 pandas 配合很方便
    df_history['total_asset'].plot()

    plt.title('Backtest Account Equity')
    plt.xlabel('Time')
    plt.ylabel('Total Asset')
    plt.grid(True)
    plt.show()
