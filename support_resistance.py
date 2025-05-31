# 连续5根K线下跌
def consecutive_decline(close_series, n=5):
    return (close_series.iloc[-n:].diff().dropna() < 0).all()

# 连续5根K线上涨  
def consecutive_rise(close_series, n=5):
    return (close_series.iloc[-n:].diff().dropna() > 0).all()

# 连续5根K线都在下轨以下（你的原始逻辑）
def consecutive_below_support(close_series, support_series, n=5):
    return sum(close_series.iloc[-n:] - support_series.iloc[-n:] < 0) == n

# 连续5根K线都在上轨以上
def consecutive_above_resistance(close_series, resistance_series, n=5):
    return sum(close_series.iloc[-n:] - resistance_series.iloc[-n:] > 0) == n