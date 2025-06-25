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
            if not hasattr(c, "price"):
                return None
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
            if not hasattr(c, "price"):
                return None
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
        is_uppon_std = arr[-1] > arr.mean() + self.mult * arr.std(ddof=0)
        return is_uppon_std

# —— 6) 单根大振幅 Bar 过滤 —— 
@dataclass
class BarSpikeCondition:
    df_attr:      str   # e.g. "df"
    direction:    str   # e.g. "up"/"down"
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
        direction_ok = (last["close"] < last["open"]) if self.direction == "down" else (last["close"] < last["open"])
        if not (o_ok and c_ok and direction_ok):
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

        self.eval_history: List[dict] = []
    def evaluate(self,
                 side: str,
                 cur_close: float,
                 close_series: pd.Series,
                 data_src,
                 open2equity_pct: float
                ) -> Optional[OrderSignal]:

        now = time.time()
        record = {
            "time": now,
            "side": side,
            "open2equity": open2equity_pct,
            "tiers": []
        }

        # 1. 先判断整体能否继续
        if self._has_order[side] or open2equity_pct >= 3:
            record["skipped"] = True
            self.eval_history.append(record)
            return None

        rule = self.long_rule if side=="long" else self.short_rule

        # 2. 逐 tier 试条件
        for tier in rule.tiers:
            tier_rec = {"name": tier.name, "skipped": False, "conds": [], "ok": None, "selected_price": None}
            # 如果已经开过头，则跳过
            if tier.name in self._opened[side]:
                tier_rec["skipped"] = True
                record["tiers"].append(tier_rec)
                continue

            price_t = None
            ok = True

            # 2.1 每个 cond 的结果
            for cond in tier.conds:
                passed = cond.check(close_series, data_src, cur_close)
                p = cond.price(close_series, data_src)
                tier_rec["conds"].append({
                    "cond_type": cond.__class__.__name__,
                    "passed": passed,
                    "price": p
                })
                if not passed:
                    ok = False
                    # 一旦失败就不再 check 后续 cond
                    break
                # 如果 price 方法给了价，用作候选
                if p is not None:
                    price_t = p

            tier_rec["ok"] = ok

            # 2.2 如果所有 cond 都过了，看看 limit_price_attr
            if ok:
                if tier.limit_price_attr:
                    price_t = float(getattr(data_src, tier.limit_price_attr).iloc[-1])
                tier_rec["selected_price"] = price_t

            record["tiers"].append(tier_rec)

            # 2.3 真正执行下单
            if ok and price_t is not None:
                self._opened[side].add(tier.name)
                self._has_order[side] = True
                self._order_time[side] = now

                record["order_placed"] = {
                    "side": side,
                    "tier": tier.name,
                    "price": price_t,
                    "amount": tier.amount,
                    "time": now
                }
                self.eval_history.append(record)

                return OrderSignal(
                    side=side,
                    action=True,
                    price=price_t,
                    amount=tier.amount,
                    order_type="limit",
                    order_time=now,
                    tier_explain=tier
                )

        # 3. 全部 tier 都没下单
        self.eval_history.append(record)
        
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

from dataclasses import dataclass, field

@dataclass
class SeqStep:
    """一个“正事件”及其与下一个正事件之间的‘禁止事件’列表。"""
    positive: Condition
    forbidden_next: List[Condition] = field(default_factory=list)

@dataclass
class SequenceWithForbidden(Condition):
    """
    在 window 根 K 中，依序触发 steps[i].positive，
    且在每次 steps[i]→steps[i+1] 的区间里，
    不能触发 forbidden_next。
    """
    steps: List[SeqStep]
    window: int = 50

    def check(self, close_series: pd.Series, data_src, cur_close: float) -> bool:
        N = len(close_series)
        if N < self.window:
            return False
        idxs = list(close_series.index[-self.window:])
        last_pos = -1

        # 对每一步
        for i, step in enumerate(self.steps):
            found = False
            # 在 last_pos+1 ... window-1 里找正事件
            for off, ts in enumerate(idxs):
                if off <= last_pos:
                    continue
                if step.positive.check(close_series.loc[:ts], data_src, close_series.loc[ts]):
                    # 找到第 i 步的触发 bar
                    # 在它与上一步触发 bar 之间，检查 forbidden_next（只针对 i-1→i）
                    if i>0:
                        for neg in self.steps[i-1].forbidden_next:
                            # 扫描上一个触发点 last_pos 到当前 off 之间
                            for off2 in range(last_pos+1, off):
                                ts2 = idxs[off2]
                                if neg.check(close_series.loc[:ts2], data_src, close_series.loc[ts2]):
                                    return False
                    last_pos = off
                    found = True
                    break
            if not found:
                return False
        return True

    def price(self, close_series: pd.Series, data_src) -> Optional[float]:
        # 默认以倒数第二个 positive 条件的 price()
        # 也可以根据业务返回任意一步的 price()
        step = self.steps[-2]  
        return step.positive.price(close_series, data_src)
    



# 假设已经有 PriceTouchCondition：
mid_d2u_touch   = PriceTouchCondition("SFrame_vp_poc",      comparator="above", window=200)
up_sl_touch = PriceTouchCondition("HFrame_vwap_up_sl",   comparator="above", window=200)
up_poc_touch = PriceTouchCondition("HFrame_vwap_up_poc", comparator="above", window=200)

mid_u2d_touch   = PriceTouchCondition("SFrame_vp_poc",      comparator="below", window=200)
down_sl_touch  = PriceTouchCondition("HFrame_vwap_down_sl", comparator="below", window=200)
down_poc_touch = PriceTouchCondition("HFrame_vwap_down_poc", comparator="bellow", window=200)

short_touch_up_then_below_center__notouch_down = SequenceWithForbidden(
    steps=[
        SeqStep(positive=up_poc_touch, forbidden_next=[]),       # 步骤1: 先触及POC

        SeqStep(positive=mid_d2u_touch,  forbidden_next=[]),         # 步骤2: 回归中轨  
        # 步骤3: 再触及POC；在“中轨→上轨”间 禁止触及下轨
        SeqStep(positive=up_poc_touch, forbidden_next=[down_poc_touch]),      
    ],
    window=300
)

long_touch_down_then_uppon_center__notouch_up = SequenceWithForbidden(
    steps=[
        SeqStep(positive=down_poc_touch, forbidden_next=[]),       # 步骤1: 先触及POC

        SeqStep(positive=mid_u2d_touch,  forbidden_next=[]),         # 步骤2: 回归中轨
         # 步骤3: 再触及POC 在“中轨→下轨”间 禁止触及下轨
        SeqStep(positive=down_poc_touch, forbidden_next=[up_poc_touch]),      
    ],
    window=300
)

@dataclass
class RuleConfig:
    # —— 10) 配置示例 —— 
    long_rule = EntryRule([
        # EntryTier(
        #     name="or_consec_or_spike",
        #     amount=1,
        #     conds=[
        #         AndCondition([
        #             VolumeSpikeCondition("df", "vol", window=80, mult=2.0),
        #             OrCondition([
        #                 #单根从SFrame_vwap_down_poc之下，暴跌 3 x atr
        #                 BarSpikeCondition(
        #                     df_attr="df",
        #                     direction="down",
        #                     open_thresh="SFrame_vwap_down_poc", open_cmp="below",
        #                     close_thresh="SFrame_vwap_down_getin", close_cmp="below",
        #                     atr_attr="atr", mult=3.0
        #                 ),
                        
        #                 short_touch_up_then_below_center__notouch_down,
        #                 ConsecutiveCondition("SFrame_vwap_down_sl", "below", 4),
        #                 ConsecutiveCondition("SFrame_vwap_down_poc", "below", 10)
        #             ])
        #         ])
        #     ],
        #     limit_price_attr="SFrame_vwap_down_sl2"
        # ),

        EntryTier(
            name="below_down_sl2_and_volspike",
            amount=1,
            conds=[
                AndCondition([
                    # VolumeSpikeCondition("df", "vol", window=80, mult=2),
                    # ConsecutiveCondition("SFrame_vwap_down_poc", "below", 8),
                    ConsecutiveCondition("SFrame_vp_poc", "below", 2),
                ])
            ],
            limit_price_attr="HFrame_vwap_down_sl2"
        ),

    ])

    short_rule = EntryRule([
    #     EntryTier(
    #         name="or_consec_or_spike_short",
    #         amount=1,
    #         conds=[
    #             AndCondition([
    #                 VolumeSpikeCondition("df", "vol", window=80, mult=2.0),
    #                 OrCondition([  
    #                     #单根暴涨x倍 atr
    #                     BarSpikeCondition(
    #                         df_attr="df",
    #                         direction="up",
    #                         open_thresh="SFrame_vwap_up_poc", open_cmp="below",
    #                         close_thresh="SFrame_vwap_down_getin", close_cmp="below",
    #                         atr_attr="atr", mult=3.0
    #                     ),
    #                     long_touch_down_then_uppon_center__notouch_up, 
    #                     ConsecutiveCondition("SFrame_vwap_up_sl", "above", 4),
    #                     ConsecutiveCondition("SFrame_vwap_up_poc", "above", 10),
    #                 ])
    #             ])
    #         ],
    #         limit_price_attr="SFrame_vwap_up_poc"
    #     ),

        EntryTier(
            name="uppon_up_sl2_and_volspike",
            amount=1,
            conds=[
                AndCondition([
                    # VolumeSpikeCondition("df", "vol", window=80, mult=1),
                    # ConsecutiveCondition("SFrame_vwap_up_poc", "above", 2),
                    ConsecutiveCondition("SFrame_vp_poc", "above", 2),
                    
                    # ConsecutiveCondition("SFrame_vwap_up_sl2", "above", 1),
                ])
            ],
            limit_price_attr="HFrame_vwap_up_sl2"
        ),
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