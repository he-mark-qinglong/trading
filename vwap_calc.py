import pandas as pd
import numpy as np
import traceback

from numba import njit


@njit
def _boundary_vwap_numba(accum, lowv, binsize, start_i, end_i):
    w, pv = 0.0, 0.0
    for kk in range(start_i, end_i + 1):
        av = accum[kk]
        pr = lowv + binsize * (kk + 0.5)
        w += av
        pv += av * pr
    return pv / w if w > 0 else np.nan

@njit
def vpvr_pct_band_vwap_log_decay_njit(
    open_prices, close_prices, vol, length, bins, pct, decay,
    vwap_series=None, use_delta=False, use_vol_filter=False,
    use_external_vwap=True
):
    n = len(close_prices)
    low_arr  = np.full(n, np.nan)
    high_arr = np.full(n, np.nan)

    vwap_given = vwap_series is not None
    # 预分配vwap数组以兼容numba
    if vwap_given:
        vwap = vwap_series
    else:
        vwap = np.zeros(n)

    for end in range(length - 1, n):
        s = close_prices[end - length + 1:end + 1]
        o_ = open_prices[end - length + 1:end + 1]
        c_ = close_prices[end - length + 1:end + 1]
        vv = vol[end - length + 1:end + 1]
        vol_log = np.log(vv + 1.0)

        lowv = np.min(s)
        highv = np.max(s)
        binsize = max((highv - lowv) / bins, 1e-8)
        vmean = np.mean(vol_log) if use_vol_filter else -1e20

        accum = np.zeros(bins)
        for j in range(length):
            lj = vol_log[j]
            if use_vol_filter and lj <= vmean:
                continue
            pricej = s[j]
            weight = decay ** (length - 1 - j)
            bi = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bi = max(0, min(bins - 1, bi))
            if use_delta:
                sign = 1 if c_[j] > o_[j] else -1 if c_[j] < o_[j] else 0
                deltaj = sign * lj
                accum[bi] += abs(deltaj) * weight
            else:
                accum[bi] += lj * weight

        total = accum.sum()
        if total <= 0:
            continue

        # 中心bin
        if use_external_vwap and vwap_given:
            vv0 = vwap[end]
            ci = int(np.floor((bins - 1) * (vv0 - lowv) / max(highv - lowv, 1e-8)))
            ci = max(0, min(bins - 1, ci))
        else:
            ci = int(np.argmax(accum))

        threshold = total * pct

        # 低端通道
        cum = 0.0
        idx0 = 0
        while idx0 < bins and cum < threshold:
            cum += accum[idx0]
            idx0 += 1
        i0, low_end = 0, min(bins - 1, idx0 - 1)

        # 高端通道
        cum = 0.0
        idx1 = bins - 1
        while idx1 >= 0 and cum < threshold:
            cum += accum[idx1]
            idx1 -= 1
        high_start, i1 = max(0, idx1 + 1), bins - 1

        # 边界vwap
        low_arr[end]  = _boundary_vwap_numba(accum, lowv, binsize, i0, low_end)
        high_arr[end] = _boundary_vwap_numba(accum, lowv, binsize, high_start, i1)

    return low_arr, high_arr

def vpvr_pct_band_vwap_log_decay(
    open_prices, close_prices, vol, length, bins, pct, decay, 
    vwap_series=None, use_delta=False, use_vol_filter=False, 
    use_external_vwap=True
):
    """
    增强版 VPVR 百分位带状 VWAP 边界计算（log‐vol 版）

    参数:
      src, open_prices, close_prices, vol: pd.Series
      length, bins, pct, decay: 同 Pine
      use_delta       = 是否用有向成交量（只是决定 log(量) 前用不使用 delta 方向，累积时仍用绝对值）
      use_vol_filter  = 是否只累积 log‐vol>平均的那部分
      use_external_vwap = 是否用外部 vwap 定中心点
    返回:
      band_low, band_high, center_vwap (pd.Series)
    """
    import sys
    if sys.gettrace() is None:
        try:
            low_arr, high_arr = vpvr_pct_band_vwap_log_decay_njit(
                open_prices.values, close_prices.values, vol.values, length, bins, pct, decay,
                vwap_series=vwap_series.values if vwap_series is not None else None,
                use_delta=use_delta,
                use_vol_filter=use_vol_filter,
                use_external_vwap=use_external_vwap
            )
            # 转为pd.Series附加index后返回
            band_low = pd.Series(low_arr, index=open_prices.index)
            band_high = pd.Series(high_arr, index=open_prices.index)
            return band_low, band_high
        except Exception as e:
            print("Error in vpvr_pct_band_vwap_log_decay:")
            traceback.print_exc()
            print("Exception message:", repr(e))

    

    src = pd.Series(close_prices)
    o   = pd.Series(open_prices)
    c   = pd.Series(close_prices)
    v   = pd.Series(vol)
    if vwap_series is not None:
        vw = pd.Series(vwap_series)
    idx = src.index

    low_arr  = np.full(len(src), np.nan)
    high_arr = np.full(len(src), np.nan)

    for end in range(length - 1, len(src)):
        slc = slice(end - length + 1, end + 1)
        s = src.iloc[slc].values
        o_ = o.iloc[slc].values
        c_ = c.iloc[slc].values
        vv = v.iloc[slc].values

        # 1) log-vol
        vol_log = np.log(vv + 1.0)

        # 2) 价格区间 & binsize
        lowv, highv = s.min(), s.max()
        binsize = max((highv - lowv) / bins, 1e-8)

        # 3) 量过滤阈值
        vmean = vol_log.mean() if use_vol_filter else -np.inf

        # 4) 累积到 bins
        accum = np.zeros(bins)
        for j in range(length):
            lj = vol_log[j]
            if lj <= vmean:
                continue
            pricej = s[j]
            weight = decay ** (length - 1 - j)
            bi = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bi = max(0, min(bins - 1, bi))
            # 如果选了 delta，就用有向再取 abs，否则直接用 lj
            if use_delta:
                sign = 1 if c_[j] > o_[j] else -1 if c_[j] < o_[j] else 0
                deltaj = sign * lj
                accum[bi] += abs(deltaj) * weight
            else:
                accum[bi] += lj * weight

        total = accum.sum()
        if total <= 0:
            continue

        # 5) 确定中心 bin
        if use_external_vwap and vwap_series is not None:
            vv0 = vw.iloc[end]
            ci = int(np.floor((bins - 1) * (vv0 - lowv) / max(highv - lowv, 1e-8)))
            ci = max(0, min(bins - 1, ci))
        else:
            ci = int(accum.argmax())

        # 6) 从边缘向内累积到 pct
        threshold = total * pct

        # 低端通道
        cum = 0.0
        i0, i1 = 0, 0
        if True:  # 先算下边界
            cum = 0.0
            idx0 = 0
            while idx0 < bins and cum < threshold:
                cum += accum[idx0]
                idx0 += 1
            i0, low_end = 0, min(bins - 1, idx0 - 1)

        # 高端通道
        cum = 0.0
        idx1 = bins - 1
        while idx1 >= 0 and cum < threshold:
            cum += accum[idx1]
            idx1 -= 1
        high_start, i1 = max(0, idx1 + 1), bins - 1

        # 7) 量价加权均价
        def boundary_vwap(start_i, end_i):
            w, pv = 0.0, 0.0
            for kk in range(start_i, end_i + 1):
                av = accum[kk]
                pr = lowv + binsize * (kk + 0.5)
                w += av
                pv += av * pr
            return pv / w if w > 0 else np.nan

        low_arr[end]  = boundary_vwap(i0, low_end)
        high_arr[end] = boundary_vwap(high_start, i1)

    return (
        pd.Series(low_arr, index=idx),
        pd.Series(high_arr, index=idx),
    )

@njit
def center_vwap_log_decay_njit(
    open_prices, close_prices, vol, length, bins, pct, decay
):
    n = len(close_prices)
    res = np.full(n, np.nan)
    for end in range(length - 1, n):
        s = close_prices[end - length + 1:end + 1]
        o_ = open_prices[end - length + 1:end + 1]
        c_ = close_prices[end - length + 1:end + 1]
        vv = vol[end - length + 1:end + 1]
        vol_log = np.log(vv + 1.0)
        lowv, highv = s.min(), s.max()
        binsize = max((highv - lowv) / bins, 1e-8)
        accum = np.zeros(bins)
        for j in range(length):
            lj = vol_log[j]
            sign = 1 if c_[j] > o_[j] else -1 if c_[j] < o_[j] else 0
            deltaj = sign * lj
            pricej = s[j]
            # 计算 bin 的索引
            bi = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bi = max(0, min(bins - 1, bi))
            weight = decay ** (length - 1 - j)
            accum[bi] += abs(deltaj) * weight
        total = accum.sum()
        if total <= 0:
            continue
        ci = int(np.argmax(accum))
        # 两侧扩展
        sumv = accum[ci]
        l, r = ci, ci
        while sumv < total * pct:
            left_v = accum[l - 1] if l > 0 else -1
            right_v = accum[r + 1] if r < bins - 1 else -1
            if left_v >= right_v and l > 0:
                l -= 1
                sumv += accum[l]
            elif r < bins - 1:
                r += 1
                sumv += accum[r]
            else:
                break
        # VWAP
        wsum, pvsum = 0.0, 0.0
        for kk in range(l, r + 1):
            av = accum[kk]
            pr = lowv + binsize * (kk + 0.5)
            wsum += av
            pvsum += av * pr
        res[end] = pvsum / wsum if wsum > 0 else np.nan
    return res
def vpvr_center_vwap_log_decay(open_prices, close_prices, vol, length, bins, pct, decay):
    """
    基于 log‐vol 的 δVPVR 中心 VWAP 计算
    对应 Pine f_delta_vpvr_vwap_center
    """
    import sys
    if sys.gettrace() is None:
        try:
            center_vwap = center_vwap_log_decay_njit(
                open_prices.values, close_prices.values, vol.values, length, bins, pct, decay
            )
            # 若需要pd.Series形式
            center_vwap = pd.Series(center_vwap, index=close_prices.index)
            return center_vwap
        except Exception as e:
            print("Error in vpvr_center_vwap_log_decay:")
            traceback.print_exc()
            print("Exception message:", repr(e))

    src = pd.Series(close_prices)
    o   = pd.Series(open_prices)
    c   = pd.Series(close_prices)
    v   = pd.Series(vol)
    idx = src.index
    res = np.full(len(src), np.nan)

    for end in range(length - 1, len(src)):
        slc = slice(end - length + 1, end + 1)
        s = src.iloc[slc].values
        o_ = o.iloc[slc].values
        c_ = c.iloc[slc].values
        vv = v.iloc[slc].values

        # 1) log‐vol
        vol_log = np.log(vv + 1.0)

        # 2) 价格区间
        lowv, highv = s.min(), s.max()
        binsize = max((highv - lowv) / bins, 1e-8)

        # 3) 初始化
        accum = np.zeros(bins)

        # 4) 按 bin 累积有向 log‐vol
        for j in range(length):
            lj = vol_log[j]
            pricej = s[j]
            sign = 1 if c_[j] > o_[j] else -1 if c_[j] < o_[j] else 0
            deltaj = sign * lj
            bi = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bi = max(0, min(bins - 1, bi))
            weight = decay ** (length - 1 - j)
            accum[bi] += abs(deltaj) * weight

        total = accum.sum()
        if total <= 0:
            continue

        # 5) 找 POC
        ci = int(accum.argmax())

        # 6) 向两侧扩展
        sumv = accum[ci]
        l, r = ci, ci
        while sumv < total * pct:
            left_v  = accum[l - 1] if l > 0 else -1
            right_v = accum[r + 1] if r < bins - 1 else -1
            if left_v >= right_v and l > 0:
                l -= 1
                sumv += accum[l]
            elif r < bins - 1:
                r += 1
                sumv += accum[r]
            else:
                break

        # 7) 中轨 VWAP
        wsum, pvsum = 0.0, 0.0
        for kk in range(l, r + 1):
            av = accum[kk]
            pr = lowv + binsize * (kk + 0.5)
            wsum  += av
            pvsum += av * pr
        res[end] = pvsum / wsum if wsum > 0 else np.nan

    return pd.Series(res, index=idx)
