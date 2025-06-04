# 连续5根K线下跌
def consecutive_decline(close_series, n=8):
    return (close_series.iloc[-n:].diff().dropna() < 0).all()

# 连续5根K线上涨  
def consecutive_rise(close_series, n=8):
    return (close_series.iloc[-n:].diff().dropna() > 0).all()

# 连续5根K线都在下轨以下（你的原始逻辑）
def consecutive_below_support(close_series, support_series, n=8):
    compare = close_series.iloc[-n:]
    base = support_series.iloc[-n:]
    percent = sum(compare -  base < 0)/n
    return percent >= 0.6

# 连续5根K线都在上轨以上
def consecutive_above_resistance(close_series, resistance_series, n=8):
    compare = close_series.iloc[-n:]
    base = resistance_series.iloc[-n:]
    percent = sum(compare - base > 0)/n
    return percent > 0.6