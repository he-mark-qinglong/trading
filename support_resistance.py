# 连续5根K线下跌
def consecutive_decline(close_series, n=8):
    return (close_series.iloc[-n:].diff().dropna() < 0).all()

# 连续5根K线上涨  
def consecutive_rise(close_series, n=8):
    return (close_series.iloc[-n:].diff().dropna() > 0).all()

# 连续5根K线都在下轨以下（你的原始逻辑）
def consecutive_below_support(close_series, support_series, n=8):
    _2N = n * 2
    return sum(close_series.iloc[-_2N:] - support_series.iloc[-_2N:] < 0) > n
    # count = 0

    # # 从倒数第 n 个到最后一项，逐一对比
    # for i in range(1, n+1):
    #     close_val = close_series.iloc[-i]
    #     support_val = support_series.iloc[-i]
    #     diff = close_val - support_val
    #     below = diff < 0

    #     # 打印中间结果，便于定位哪一条数据出问题
    #     print(f"第{-i} 项 -> close: {close_val}, support: {support_val}, "
    #         f"diff: {diff}, below_zero: {below}")

    #     if below:
    #         count += 1

    # return count == n

# 连续5根K线都在上轨以上
def consecutive_above_resistance(close_series, resistance_series, n=8):
    _2N = n * 2
    return sum(close_series.iloc[-_2N:] - resistance_series.iloc[-_2N:] > 0) > n
    # count = 0

    # # 从倒数第 n 个到最后一项，逐一对比
    # for i in range(1, n+1):
    #     close_val = close_series.iloc[-i]
    #     resistance_val = resistance_series.iloc[-i]
    #     diff = close_val - resistance_val
    #     uppon = diff > 0

    #     # 打印中间结果，便于定位哪一条数据出问题
    #     print(f"第{-i} 项 -> close: {close_val}, resistance: {resistance_val}, "
    #         f"diff: {diff}, uppon_zero: {uppon}")

    #     if uppon:
    #         count += 1
    
    # return count == n