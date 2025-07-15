
import pandas as pd
import numpy as np

def compute_dynamic_kama(
    df: pd.DataFrame,
    src_col: str = 'close',
    len_er: int = 30,
    fast: int = 6,
    slow2fast_times: float = 2.0,
    slow: int = 120,
    intervalP: float = 0.01,
    minLen: int = 10,
    maxLen: int = 60,
    volLen: int = 30
) -> pd.DataFrame:
    """
    Compute two KAMA lines with adaptive ER window lengths based on volatility.
    Returns the input DataFrame augmented with columns:
      - dynLen1, dynLen2 : adaptive ER window lengths
      - kama1, kama2     : the two KAMA series
    """
    src = df[src_col].astype(float).reset_index(drop=True)
    n = len(src)

    # Derived parameters
    len2   = int(len_er * slow2fast_times)
    fast2  = int(fast * slow2fast_times)
    slow2  = int(slow * slow2fast_times)  

    # 1. compute normalized volatility (clamped 0~1)
    stdev_vol = src.rolling(volLen, min_periods=1).std()
    avg_vol   = stdev_vol.rolling(volLen, min_periods=1).mean()
    norm_vol  = (stdev_vol - avg_vol) / avg_vol
    norm_vol_clamped = norm_vol.clip(lower=0, upper=1).fillna(0)

    # 2. dynamic ER window lengths
    dynLen1 = ((maxLen - norm_vol_clamped * (maxLen - minLen))
               .round().astype(int).clip(lower=1))
    dynLen2 = ((dynLen1 * (len2 / len_er))
               .round().astype(int).clip(lower=1))

    # 3. init arrays
    kama1 = np.full(n, np.nan, dtype=float)
    kama2 = np.full(n, np.nan, dtype=float)

    # initial value
    if n > 0:
        kama1[0] = src.iloc[0]
        kama2[0] = src.iloc[0]

    # precompute diffs
    abs_delta = src.diff().abs().fillna(0).values

    # 4. iterate
    for i in range(1, n):
        # ER1
        w1 = dynLen1.iloc[i]
        if i >= w1:
            sum_abs1 = abs_delta[i-w1+1:i+1].sum()
            price_chg1 = abs(src.iloc[i] - src.iloc[i-w1])
            er1 = price_chg1 / sum_abs1 if sum_abs1 != 0 else 0.0
        else:
            er1 = 0.0
        sc1 = (er1 * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2

        # update kama1
        kama1[i] = kama1[i-1] + sc1 * (src.iloc[i] - kama1[i-1])

        # ER2
        w2 = dynLen2.iloc[i]
        if i >= w2:
            sum_abs2 = abs_delta[i-w2+1:i+1].sum()
            price_chg2 = abs(src.iloc[i] - src.iloc[i-w2])
            er2 = price_chg2 / sum_abs2 if sum_abs2 != 0 else 0.0
        else:
            er2 = 0.0
        sc2 = (er2 * (2/(fast2+1) - 2/(slow2+1)) + 2/(slow2+1)) ** 2

        # update kama2
        kama2[i] = kama2[i-1] + sc2 * (src.iloc[i] - kama2[i-1])

    # attach to DataFrame
    out = df.copy().reset_index(drop=True)
    out['dynLen1'] = dynLen1
    out['dynLen2'] = dynLen2
    out['kama1']    = kama1
    out['kama2']    = kama2

    return out

# Example usage:
if __name__ == "__main__":
    # load your data
    df = pd.read_csv("your_price_data.csv", parse_dates=True, index_col=0)
    res = compute_dynamic_kama(df)
    print(res[['dynLen1','dynLen2','kama1','kama2']].tail())
