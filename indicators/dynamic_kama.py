
import pandas as pd
import numpy as np

import pandas as pd

def convert_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    将传统OHLCV数据转换为Heikin Ashi格式。

    参数:
        df: 包含普通OHLCV数据的DataFrame，列名包含 ['open', 'high', 'low', 'close', 'volume']

    返回:
        heikin_ashi_df: 包含Heikin Ashi OHLCV数据的DataFrame，索引与df相同
    """
    ha_df = pd.DataFrame(index=df.index, columns=['datetime', 'open', 'high', 'low', 'close', 'vol'])

    # 计算Heikin Ashi Close：各价格均值
    ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # 初始化Heikin Ashi Open，第一根bar取普通开盘价，后续用前一根heikin ashi开盘与收盘均值
    ha_df.iloc[0, ha_df.columns.get_loc('open')] = df.iloc[0]['open']

    for i in range(1, len(df)):
        ha_df.iloc[i, ha_df.columns.get_loc('open')] = (ha_df.iloc[i-1]['open'] + ha_df.iloc[i-1]['close']) / 2

    # Heikin Ashi High：取当期普通最高价、HA开盘价、HA收盘价中最大值
    ha_df['high'] = pd.concat([df['high'], ha_df['open'], ha_df['close']], axis=1).max(axis=1)

    # Heikin Ashi Low：取当期普通最低价、HA开盘价、HA收盘价中最小值
    ha_df['low'] = pd.concat([df['low'], ha_df['open'], ha_df['close']], axis=1).min(axis=1)

    # 音量一般保持不变，直接复制原数据
    ha_df['vol'] = df['vol']

    return ha_df


def calculate_vwma(price, volume, period):
    """
    计算成交量加权移动平均（VWMA）

    参数:
        price: pd.Series 或 np.array，价格序列，如收盘价
        volume: pd.Series 或 np.array，成交量序列
        period: int，计算周期

    返回:
        pd.Series，VWMA序列
    """
    price = pd.Series(price)
    volume = pd.Series(volume)

    pv = price * volume
    pv_sum = pv.rolling(window=period).sum()
    vol_sum = volume.rolling(window=period).sum()

    vwma = pv_sum / vol_sum
    return vwma

def compute_dynamic_kama(
    df: pd.DataFrame,
    src_col: str = 'close',
    len_er: int = 30,
    fast: int = 6,
    second2first_times: float = 2.0,
    slow: int = 120,
    intervalP: float = 0.01,
    minLen: int = 10,
    maxLen: int = 60,
    volLen: int = 30,
    hl: int = 3
) -> pd.DataFrame:
    """
    Compute two KAMA lines with adaptive ER window lengths based on volatility.
    Returns the input DataFrame augmented with columns:
      - dynLen1, dynLen2 : adaptive ER window lengths
      - kama1, kama2     : the two KAMA series
    """
    # df = convert_to_heikin_ashi(df=df)
    src_orig = df[src_col].astype(float).reset_index(drop=True)
    n = len(src_orig)

    # 1. 预先 EWMA 衰减
    alpha = 1 - 2 ** (-1.0 / hl)
    src = src_orig.ewm(alpha=alpha, adjust=False).mean()
    # src = src_orig
    # src = calculate_vwma(src_orig, df['vol'].astype(float).reset_index(drop=True), 3)    ######这一行
    n = len(src)
    # print('length of src=',n)
    # print(src.head(10),'--------', src.tail(10))
    # Derived parameters
    len2   = int(len_er * second2first_times)
    fast2  = int(fast * second2first_times)
    slow2  = int(slow * second2first_times)  

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
    out = df.copy(deep=True)
    
    out['dynLen1'] = dynLen1
    out['dynLen2'] = dynLen2
    out['kama1']    = kama1
    out['kama2']    = kama2
    
    return out

import numpy as np
import pandas as pd

def rolling_lr_slope_last_n(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute slope of linear regression on last `window` points for each index,
    with x = 0..window-1. If not enough points, slope = 0 (matches Pine f_slope_seq).
    """
    n = len(arr)
    slopes = np.zeros(n, dtype=float)
    if window <= 1:
        return slopes

    xs = np.arange(window, dtype=float)          # 0..window-1
    sum_x = xs.sum()
    sum_xx = (xs * xs).sum()
    denom = window * sum_xx - sum_x * sum_x
    if denom == 0:
        return slopes

    for i in range(n):
        if i < window - 1:
            slopes[i] = 0.0
        else:
            y = arr[i - window + 1 : i + 1].astype(float)
            sum_y = y.sum()
            sum_xy = (xs * y).sum()
            slopes[i] = (window * sum_xy - sum_x * sum_y) / denom
    return slopes


def compute_dynamic_kama_with_predictions(
    df: pd.DataFrame,
    src_col: str = 'close',
    len_er: int = 30,
    fast: int = 6,
    second2first_times: float = 2.0,
    slow: int = 30,
    intervalP: float = 0.01,
    minLen: int = 10,
    maxLen: int = 60,
    volLen: int = 30,
    hl: int = 3,
    # Prediction params (mapping from Pine)
    fastSlopeLen: int = 20,
    slowSlopeLen: int = 30,
    m: float = 1.0,
    w: float = 0.6
) -> pd.DataFrame:
    """
    Compute two dynamic KAMA lines (kama1/kama2) as before, but also:
      - save sc1/sc2 (smoothing constants) per bar
      - compute slopes on sc1/sc2
      - compute sc1_predict_exact/sc2_predict_exact (Pine formula)
      - compute kama1_predict_exact/kama2_predict_exact by recursive formula

    Returns DataFrame with columns:
      - dynLen1, dynLen2, kama1, kama2, sc1, sc2,
      - sc1_slope_fast, sc1_slope_slow, sc2_slope_fast, sc2_slope_slow,
      - sc1_predict_exact, sc2_predict_exact,
      - kama1_predict_exact, kama2_predict_exact
    """
    src_orig = df[src_col].astype(float).reset_index(drop=True)
    n = len(src_orig)

    # 1. EWMA pre-smooth
    alpha = 1 - 2 ** (-1.0 / hl)
    src = src_orig.ewm(alpha=alpha, adjust=False).mean()
    src = src.reset_index(drop=True)
    n = len(src)

    # Derived params
    len2   = int(len_er * second2first_times)
    fast2  = int(fast * second2first_times)
    slow2  = int(slow * second2first_times)

    # normalized volatility
    stdev_vol = src.rolling(volLen, min_periods=1).std()
    avg_vol   = stdev_vol.rolling(volLen, min_periods=1).mean()
    norm_vol  = (stdev_vol - avg_vol) / avg_vol
    norm_vol_clamped = norm_vol.clip(lower=0, upper=1).fillna(0)

    dynLen1 = ((maxLen - norm_vol_clamped * (maxLen - minLen))
               .round().astype(int).clip(lower=1))
    dynLen2 = ((dynLen1 * (len2 / len_er))
               .round().astype(int).clip(lower=1))

    kama1 = np.full(n, np.nan, dtype=float)
    kama2 = np.full(n, np.nan, dtype=float)
    sc1_arr = np.full(n, np.nan, dtype=float)
    sc2_arr = np.full(n, np.nan, dtype=float)

    if n > 0:
        kama1[0] = src.iloc[0]
        kama2[0] = src.iloc[0]

    abs_delta = src.diff().abs().fillna(0).values

    # main loop computing kama and sc arrays
    for i in range(1, n):
        # ER1
        w1 = int(dynLen1.iloc[i])
        if i >= w1:
            sum_abs1 = abs_delta[i-w1+1:i+1].sum()
            price_chg1 = abs(src.iloc[i] - src.iloc[i-w1])
            er1 = price_chg1 / sum_abs1 if sum_abs1 != 0 else 0.0
        else:
            er1 = 0.0
        sc1 = (er1 * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        sc1_arr[i] = sc1
        kama1[i] = kama1[i-1] + sc1 * (src.iloc[i] - kama1[i-1])

        # ER2
        w2 = int(dynLen2.iloc[i])
        if i >= w2:
            sum_abs2 = abs_delta[i-w2+1:i+1].sum()
            price_chg2 = abs(src.iloc[i] - src.iloc[i-w2])
            er2 = price_chg2 / sum_abs2 if sum_abs2 != 0 else 0.0
        else:
            er2 = 0.0
        sc2 = (er2 * (2/(fast2+1) - 2/(slow2+1)) + 2/(slow2+1)) ** 2
        sc2_arr[i] = sc2
        kama2[i] = kama2[i-1] + sc2 * (src.iloc[i] - kama2[i-1])

    # build output DataFrame
    out = df.copy(deep=True).reset_index(drop=True)
    out['dynLen1'] = dynLen1
    out['dynLen2'] = dynLen2
    out['kama1']    = kama1
    out['kama2']    = kama2
    out['sc1'] = sc1_arr
    out['sc2'] = sc2_arr
    out['src_ewma'] = src  # optional: expose the pre-smoothed src

    # Now compute slopes on sc1/sc2 for predictions
    sc1_vals = out['sc1'].fillna(0).values
    sc2_vals = out['sc2'].fillna(0).values

    sc1_slope_fast = rolling_lr_slope_last_n(sc1_vals, fastSlopeLen)
    sc1_slope_slow = rolling_lr_slope_last_n(sc1_vals, slowSlopeLen)
    sc2_slope_fast = rolling_lr_slope_last_n(sc2_vals, fastSlopeLen)
    sc2_slope_slow = rolling_lr_slope_last_n(sc2_vals, slowSlopeLen)

    out['sc1_slope_fast'] = sc1_slope_fast
    out['sc1_slope_slow'] = sc1_slope_slow
    out['sc2_slope_fast'] = sc2_slope_fast
    out['sc2_slope_slow'] = sc2_slope_slow

    # sc_predict_exact
    sc1_predict_exact = (out['sc1'].fillna(0).values + m * (w * sc1_slope_fast + (1 - w) * sc1_slope_slow)) ** 2
    sc2_predict_exact = (out['sc2'].fillna(0).values + m * (w * sc2_slope_fast + (1 - w) * sc2_slope_slow)) ** 2
    out['sc1_predict_exact'] = sc1_predict_exact
    out['sc2_predict_exact'] = sc2_predict_exact

    # recursive kama_predict_exact
    n = len(out)
    kama1_predict_exact = np.full(n, np.nan, dtype=float)
    kama2_predict_exact = np.full(n, np.nan, dtype=float)
    src_vals = out[src_col].astype(float).values

    for i in range(n):
        if i == 0:
            kama1_predict_exact[i] = out['kama1'].iloc[0]
            kama2_predict_exact[i] = out['kama2'].iloc[0]
        else:
            kama1_predict_exact[i] = kama1_predict_exact[i-1] + sc1_predict_exact[i] * (src_vals[i] - kama1_predict_exact[i-1])
            kama2_predict_exact[i] = kama2_predict_exact[i-1] + sc2_predict_exact[i] * (src_vals[i] - kama2_predict_exact[i-1])

    out['kama1_predict_exact'] = kama1_predict_exact
    out['kama2_predict_exact'] = kama2_predict_exact

    return out

def anchored_momentum_via_kama(kama1: pd.Series, kama2: pd.Series, signal_period: int):
    df = pd.DataFrame({
        'amom': kama1 - kama2
    })
    df['amoms'] = df['amom'].ewm(span=signal_period, adjust=False).mean()
    df['hl'] = df['amom'] - df['amoms']

    # 颜色判断
    df['hlc'] = np.where(
        df['amom'] > df['amoms'], 
        np.where(df['amom'] >= 0, 'green', 'orange'),
        np.where(df['amom'] >= 0, 'orange', 'red')
    )
    return df


# Example usage:
if __name__ == "__main__":
    # load your data
    df = pd.read_csv("your_price_data.csv", parse_dates=True, index_col=0)
    res = compute_dynamic_kama(df)
    print(res[['dynLen1','dynLen2','kama1','kama2']].tail())
