import time
from dataclasses import dataclass
from typing import Protocol, List, Optional, Set, Dict
import numpy as np
import support_resistance  # 你的 consecutive_above/below 函数
import pandas as pd

# —— 1) Condition 接口 —— 
class Condition(Protocol):
    def check(self, close_series: pd.Series, data_src, cur_close: float) -> bool:
        ...

    def price(self, close_series: pd.Series, data_src) -> Optional[float]:
        return None

# —— 2) 并列 OR：subs 中任意一个通过即 True —— 
@dataclass
class OrCondition:
    subs: List[Condition]

    def check(self, close_series, data_src, cur_close: float) -> bool:
        return any(c.check(close_series, data_src, cur_close) for c in self.subs)

    def price(self, close_series, data_src) -> Optional[float]:
        for c in self.subs:
            p = c.price(close_series, data_src)
            if p is not None:
                return p
        return None

# —— 3) 并列 AND：subs 全部通过才 True —— 
@dataclass
class AndCondition:
    subs: List[Condition]

    def check(self, close_series, data_src, cur_close: float) -> bool:
        return all(c.check(close_series, data_src, cur_close) for c in self.subs)

    def price(self, close_series, data_src) -> Optional[float]:
        for c in self.subs:
            p = c.price(close_series, data_src)
            if p is not None:
                return p
        return None

# —— 4) 连续突破/破位过滤 —— 
@dataclass
class ConsecutiveCondition:
    data_attr: str       # e.g. "HFrame_vwap_down_getin"
    comparator: str      # "above"/"below"
    count: int

    def check(self, close_series, data_src, cur_close: float) -> bool:
        thresh = getattr(data_src, self.data_attr)
        if self.comparator == "above":
            return support_resistance.consecutive_above_resistance(
                close_series, thresh, self.count)
        else:
            return support_resistance.consecutive_below_support(
                close_series, thresh, self.count)

# —— 5) 成交量突增过滤 —— 
@dataclass
class VolumeSpikeCondition:
    df_attr: str     # data_src 上 DataFrame 属性名, e.g. "df"
    vol_key: str     # 列名, e.g. "vol"
    window: int      # lookback length
    mult: float      # std multiplier

    def check(self, close_series, data_src, cur_close: float) -> bool:
        df: pd.DataFrame = getattr(data_src, self.df_attr)
        arr = df[self.vol_key].iloc[-self.window:].to_numpy().astype(float)
        if arr.size < self.window:
            return False
        return arr[-1] > arr.mean() + self.mult * arr.std(ddof=0)

# —— 6) 单根大振幅 Bar 过滤 —— 
@dataclass
class BarSpikeCondition:
    df_attr:      str   # e.g. "df"
    open_thresh:  str   # e.g. "SFrame_vp_poc"
    open_cmp:     str   # "above"/"below"
    close_thresh: str   # e.g. "HFrame_vwap_down_getin"
    close_cmp:    str   # "above"/"below"
    atr_attr:     str   # e.g. "atr"
    mult:         float # ATR multiplier

    def check(self, close_series, data_src, cur_close: float) -> bool:
        df: pd.DataFrame = getattr(data_src, self.df_attr)
        if len(df) < 1:
            return False
        last = df.iloc[-1]
        ot = getattr(data_src, self.open_thresh).iloc[-1]
        ct = getattr(data_src, self.close_thresh).iloc[-1]
        o_ok = (last["open"] < ot) if self.open_cmp == "below" else (last["open"] > ot)
        c_ok = (last["close"] < ct) if self.close_cmp == "below" else (last["close"] > ct)
        if not (o_ok and c_ok):
            return False
        hi, lo = last["high"], last["low"]
        atr = getattr(data_src, self.atr_attr).iloc[-1]
        return (hi - lo) >= self.mult * atr

# —— 7) 触及价格来源（可选，用于 sliding-window max/min） —— 
@dataclass
class PriceTouchCondition:
    data_attr: str
    comparator: str
    window: int = 20

    def check(self, close_series, data_src, cur_close: float) -> bool:
        return True

    def price(self, close_series, data_src) -> Optional[float]:
        arr = getattr(data_src, self.data_attr).iloc[-self.window:]
        if arr.empty:
            return None
        return float(np.max(arr) if self.comparator == "above" else np.min(arr))

# —— 8) 信号与规则的数据结构 —— 
@dataclass
class EntryTier:
    name:             str
    amount:           int
    conds:            List[Condition]
    limit_price_attr: Optional[str] = None

@dataclass
class EntryRule:
    tiers: List[EntryTier]

@dataclass
class OrderSignal:
    side:       str      # "long"/"short"
    action:     bool     # True=发 limit 单
    price:      float
    amount:     int
    order_type: str      # "limit"
    order_time: float
    tier_explain: EntryTier

# —— 9) 核心策略 —— 
class MultiFramePOCStrategy:
    def __init__(self,
                 long_rule: EntryRule,
                 short_rule: EntryRule,
                 timeout: float = 60.0):
        self.long_rule   = long_rule
        self.short_rule  = short_rule
        self.timeout     = timeout
        self._opened: Dict[str, Set[str]] = {"long": set(), "short": set()}
        self._has_order: Dict[str, bool] = {"long": False, "short": False}
        self._order_time: Dict[str, Optional[float]] = {"long": None, "short": None}

    def evaluate(self,
                 side: str,
                 cur_close: float,
                 close_series: pd.Series,
                 data_src,
                 open2equity_pct: int
                ) -> Optional[OrderSignal]:
        # 已有挂单未撤或累计过多单时跳过
        if self._has_order[side] or open2equity_pct >= 0.2:
            return None

        rule = self.long_rule if side == "long" else self.short_rule
        now  = time.time()

        for tier in rule.tiers:
            if tier.name in self._opened[side]:
                continue

            ok = True
            price_t: Optional[float] = None
            for cond in tier.conds:
                if not cond.check(close_series, data_src, cur_close):
                    ok = False
                    break
                p = cond.price(close_series, data_src)
                if p is not None:
                    price_t = p
            if not ok:
                continue

            # 如果配置了 limit_price_attr，覆盖 price_t
            if tier.limit_price_attr:
                series = getattr(data_src, tier.limit_price_attr)
                price_t = float(series.iloc[-1])

            if price_t is None:
                continue

            # 发单
            self._opened[side].add(tier.name)
            self._has_order[side]   = True
            self._order_time[side]  = now
            return OrderSignal(
                side=side,
                action=True,
                price=price_t,
                amount=tier.amount,
                order_type="limit",
                order_time=now,
                tier_explain=tier
            )
        return None

    def should_cancel(self, side: str) -> bool:
        """挂单超时撤销"""
        if not self._has_order[side]:
            return False
        ts = self._order_time[side]
        if ts is None or time.time() - ts >= self.timeout:
            self._order_time[side] = None
            return True
        return False

    def clear_order(self, side: str):
        """外部撤单或成交时调用，清空状态"""
        self._has_order[side]  = False
        self._order_time[side] = None
        self._opened[side].clear()

# —— 10) 配置示例 —— 
long_rule = EntryRule([
    EntryTier(
        name="or_consec_or_spike",
        amount=1,
        conds=[
            AndCondition([
                VolumeSpikeCondition("df", "vol", window=20, mult=2.0),
                OrCondition([
                    ConsecutiveCondition("HFrame_vwap_down_getin", "below", 5),
                    BarSpikeCondition(
                        df_attr="df",
                        open_thresh="SFrame_vp_poc", open_cmp="below",
                        close_thresh="HFrame_vwap_down_getin", close_cmp="below",
                        atr_attr="atr", mult=6.0
                    )
                ])
            ])
        ],
        limit_price_attr="HFrame_vwap_down_getin"
    ),
])

short_rule = EntryRule([
    EntryTier(
        name="or_consec_or_spike_short",
        amount=1,
        conds=[
            AndCondition([
                VolumeSpikeCondition("df", "vol", window=20, mult=2.0),
                
                OrCondition([  
                    BarSpikeCondition(
                        df_attr="df",
                        open_thresh="SFrame_vp_poc", open_cmp="below",
                        close_thresh="HFrame_vwap_down_getin", close_cmp="below",
                        atr_attr="atr", mult=6.0
                    ),
                    ConsecutiveCondition("HFrame_vwap_up_getin", "above", 5), 
                    ConsecutiveCondition("HFrame_vwap_up_sl", "above", 1)
                ])
            ])
        ],
        limit_price_attr="HFrame_vwap_up_getin"
    ),
    # … 你可以继续添加其他 tiers …
])

# —— 11) 使用示例 —— 
# strategy = MultiFramePOCStrategy(long_rule, short_rule, timeout=60)
# 每根新 Bar 调用:
# signal = strategy.evaluate(
#     side="long",
#     cur_close=cur_close,
#     close_series=close_series,
#     data_src=my_data_src,
#     record_buy_total=record_buy_total
# )
# if signal:
#     send_limit_order(signal)
# # 撤单检查
# if strategy.should_cancel("long"):
#     cancel_order("long")
#     strategy.clear_order("long")