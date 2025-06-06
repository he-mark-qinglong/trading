import numpy as np, pandas as pd, timeit
import pandas_ta as ta
# 构造大 Series
N = 2_000_000
s = pd.Series(np.random.rand(N))
span = 20

def by_ewm():
    return s.ewm(span=span, adjust=False).mean()

# 如果你装了 pandas_ta：
import pandas_ta as pta
def by_pandas_ta():
    return pta.rma(s, length=span)

# 测试
print("ewm:",    timeit.timeit(by_ewm,    number=3))
print("pandas_ta:", timeit.timeit(by_pandas_ta, number=3))