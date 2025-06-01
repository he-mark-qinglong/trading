import pandas as pd
import numpy as np

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


def vpvr_center_vwap_log_decay(open_prices, close_prices, vol, length, bins, pct, decay):
    """
    基于 log‐vol 的 δVPVR 中心 VWAP 计算
    对应 Pine f_delta_vpvr_vwap_center
    """
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

def vpvr_pct_band_vwap_decay(open_prices, close_prices, vol, length, bins, pct, decay, 
                                 vwap_series=None, use_delta=False, use_vol_filter=False, 
                                 use_external_vwap=False):
    """
    增强版VPVR百分位带状VWAP边界计算
    
    参数:
        src: pd.Series, 价格序列(通常是典型价格)
        open_prices: pd.Series, 开盘价
        close_prices: pd.Series, 收盘价  
        vol: pd.Series, 成交量
        length: int, 滑窗长度
        bins: int, 分桶数量
        pct: float, 累计包含的成交量比例(0~1)
        decay: float, 时间加权衰减因子
        vwap_series: pd.Series, 外部VWAP序列(可选)
        use_delta: bool, 是否使用delta成交量(有向成交量绝对值)
        use_vol_filter: bool, 是否启用成交量过滤(只计算大于均值的K线)
        use_external_vwap: bool, 是否使用外部VWAP作为中心(否则使用POC)
    
    返回:
        tuple: (band_low, band_high, center_vwap) 三个pd.Series
    """
    import pandas as pd
    import numpy as np
    
    src = pd.Series(src)
    open_prices = pd.Series(open_prices)
    close_prices = pd.Series(close_prices)
    vol = pd.Series(vol)
    
    if vwap_series is not None:
        vwap_series = pd.Series(vwap_series)
    
    index = src.index
    band_low = np.full(len(src), np.nan)
    band_high = np.full(len(src), np.nan) 
    
    for end in range(length-1, len(src)):
        slc = slice(end-length+1, end+1)
        s = src.iloc[slc].values
        o = open_prices.iloc[slc].values
        c = close_prices.iloc[slc].values
        v = vol.iloc[slc].values
        
        # 计算价格范围
        lowv = np.min(s)
        highv = np.max(s)
        binsize = max((highv - lowv) / bins, 1e-8)
        accum = np.zeros(bins)
        
        # 量过滤：计算成交量均值
        vmean = np.mean(v) if use_vol_filter else 0
        
        # 累积到bins
        for j in range(length):
            pricej = s[j]
            openj = o[j]
            closej = c[j]
            volj = v[j]
            
            # 量过滤判断
            if use_vol_filter and volj <= vmean:
                continue
                
            # 计算权重
            weightj = decay**(length - 1 - j)
            
            # 计算bin索引
            bini = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bini = max(0, min(bins - 1, bini))
            
            # 根据是否使用delta选择累积方式
            if use_delta:
                # 计算有向成交量的绝对值
                if closej > openj:
                    deltaj = volj
                elif closej < openj:
                    deltaj = -volj
                else:
                    deltaj = 0
                accum[bini] += abs(deltaj) * weightj
            else:
                # 普通成交量
                accum[bini] += volj * weightj
        
        total_vol = np.sum(accum)
        if total_vol == 0:
            continue
            
        # 确定中心点
        if use_external_vwap and vwap_series is not None:
            # 使用外部VWAP确定中心
            vv = vwap_series.iloc[end]
            center_idx = int(np.floor((bins - 1) * (vv - lowv) / max(highv - lowv, 1e-8)))
            center_idx = max(0, min(bins - 1, center_idx))
        else:
            # 使用POC(最大成交量bin)作为中心
            center_idx = np.argmax(accum)
        
        # 从中心向两侧扩展
        sum_vol = accum[center_idx]
        left_idx = center_idx
        right_idx = center_idx
        
        while sum_vol < total_vol * pct:
            add_left = accum[left_idx - 1] if left_idx > 0 else -1
            add_right = accum[right_idx + 1] if right_idx < bins - 1 else -1
            
            if add_left >= add_right and left_idx > 0:
                left_idx -= 1
                sum_vol += accum[left_idx]
            elif right_idx < bins - 1:
                right_idx += 1
                sum_vol += accum[right_idx]
            elif left_idx > 0:
                left_idx -= 1
                sum_vol += accum[left_idx]
            else:
                break
        # 计算左边界附近5个bins的加权均价
        def calc_boundary_vwap(boundary_idx, is_left=True):
            # 优先取boundary_idx上下各2个bins
            start_i = max(0, boundary_idx - 2)
            end_i = min(bins - 1, boundary_idx + 2)
            
            # 检查bins数量，如果不足5个则扩展
            bins_count = end_i - start_i + 1
            if bins_count < 5:
                if is_left and end_i < bins - 1:
                    # 左边界优先向上扩展
                    end_i = min(bins - 1, start_i + 4)
                elif not is_left and start_i > 0:
                    # 右边界优先向下扩展  
                    start_i = max(0, end_i - 4)
            
            # 再次检查，还不足则向另一侧扩展
            bins_count = end_i - start_i + 1
            if bins_count < 5:
                if is_left and start_i > 0:
                    start_i = max(0, end_i - 4)
                elif not is_left and end_i < bins - 1:
                    end_i = min(bins - 1, start_i + 4)
            
            # 计算这些bins的加权均价
            wsum = 0.0
            pvsum = 0.0
            for i in range(start_i, end_i + 1):
                v_i = accum[i]
                price_i = lowv + binsize * (i + 0.5)
                wsum += v_i
                pvsum += v_i * price_i
            
            return pvsum / wsum if wsum > 0 else np.nan
        
        # 计算左右边界的加权均价
        band_low[end] = calc_boundary_vwap(left_idx, is_left=True)
        band_high[end] = calc_boundary_vwap(right_idx, is_left=False)
    
    return (pd.Series(band_low, index=index), 
            pd.Series(band_high, index=index),
            )
    
def vpvr_center_vwap_decay(src, vol, length, bins, pct, decay):
    """
    输入:
        src, vol: pd.Series (或可转Series的一维结构)
        length: 窗口长度
        bins: 分桶数
        pct: 累计volume覆盖比例 (如0.7)
        decay: 衰减系数 (如0.995)
    输出:
        pd.Series: 每根K线vwap, index与输入对齐
    """
    src = pd.Series(src)
    vol = pd.Series(vol)
    index = src.index
    result = np.full(len(src), np.nan)

    for end in range(length - 1, len(src)):
        slc = slice(end - length + 1, end + 1)
        s = src.iloc[slc].values
        v = vol.iloc[slc].values

        lowv, highv = np.min(s), np.max(s)
        binsize = max((highv - lowv) / bins, 1e-8)
        accum = np.zeros(bins)

        for j in range(length):
            pricej = s[j]
            volj = v[j]
            weightj = decay ** (length - 1 - j)
            bini = int(np.floor((bins - 1) * (pricej - lowv) / max(highv - lowv, 1e-8)))
            bini = max(0, min(bins - 1, bini))
            accum[bini] += volj * weightj

        total_vol = np.sum(accum)
        if total_vol == 0:
            continue

        center_idx = np.argmax(accum)
        sum_vol = accum[center_idx]
        left_idx = center_idx
        right_idx = center_idx

        while sum_vol < total_vol * pct:
            add_left = accum[left_idx - 1] if left_idx > 0 else -1
            add_right = accum[right_idx + 1] if right_idx < bins - 1 else -1
            if add_left >= add_right:
                if left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                elif right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                else:
                    break
            else:
                if right_idx < bins - 1:
                    right_idx += 1
                    sum_vol += accum[right_idx]
                elif left_idx > 0:
                    left_idx -= 1
                    sum_vol += accum[left_idx]
                else:
                    break

        wsum, pvsum = 0.0, 0.0
        for i in range(left_idx, right_idx + 1):
            v_i = accum[i]
            price_i = lowv + binsize * (i + 0.5)
            wsum += v_i
            pvsum += v_i * price_i
        result[end] = pvsum / wsum if wsum > 0 else np.nan

    return pd.Series(result, index=index)