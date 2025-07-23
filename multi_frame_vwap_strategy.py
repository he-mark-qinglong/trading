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
from dataclasses import dataclass
import pandas as pd
from typing import Optional

@dataclass
class BarsAwayFromThresholdCondition:
    """
    判断从最近一次“触及阈值”到当前，是否已过去至少 count 根 K 线。
    data_attr:   data_src 上存阈值的 Series 名称
    comparator:  "above" 表示 Price <= thresh 时视为触及；“below”则 Price >= thresh
    count:       至少要脱离阈值多少根 Bar
    """
    data_attr: str
    comparator: str
    count: int

    def check(self,
              close_series: pd.Series,
              data_src,
              cur_close: float) -> bool:
        # 1) 拿到收盘价和阈值，两端都 dropna
        closes = close_series.dropna()
        thresh = getattr(data_src, self.data_attr).dropna()
        if len(thresh) == 0:
            return False

        # 2) 尾部对齐：取最近 len(thresh) 根收盘价
        if len(closes) < len(thresh):
            return False
        closes_tail = closes.iloc[-len(thresh):]

        # 3) 从后往前找“最后一次触及”位置
        last_touch = None  # 在 closes_tail 中的位置 index
        for i in range(len(thresh) - 1, -1, -1):
            p = closes_tail.iat[i]
            t = thresh.iat[i]
            if self.comparator == "above":
                # 触及阈值意味着 p <= t
                if p <= t:
                    last_touch = i
                    break
            else:
                # 触及阈值意味着 p >= t
                if p >= t:
                    last_touch = i
                    break

        # 4) 计算脱离后的 Bar 数
        if last_touch is None:
            # 从来没触及过，则全部 len(closes_tail) 都算“脱离”
            bars_away = len(closes_tail)
        else:
            bars_away = len(closes_tail) - 1 - last_touch

        return bars_away >= self.count

    def price(self,
              close_series: pd.Series,
              data_src) -> Optional[float]:
        return None

    def price(self,
              close_series: pd.Series,
              data_src) -> Optional[float]:
        return None
    
# —— 5) 成交量突增过滤 —— 
@dataclass
class VolumeSpikeCondition:
    df_attr: str     # data_src 上 DataFrame 属性名, e.g. "df"
    vol_key: str     # 列名, e.g. "vol"

    def check(self, close_series, data_src, cur_close: float) -> bool:
        df: pd.DataFrame = getattr(data_src, self.df_attr)

        vol_normal = df['vol'].iloc[-3:] 
        vol_higher = df[self.vol_key].iloc[-3:]
        vol_higher_accum = sum(vol_normal - vol_higher)
        return vol_higher_accum > 2

# —— 6) 单根大振幅 Bar 过滤 —— 
@dataclass
class BarSpikeCondition:
    df_attr:      str   # e.g. "df"
    direction:    str   # e.g. "up"/"down"
    open_thresh:  str   # e.g. "SFrame_vwap_poc"
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
                 timeout: float = 60.0,
                 max_open2equity_pct = 0.4):
        self.long_rule   = long_rule
        self.short_rule  = short_rule
        self.timeout     = timeout
        self.max_open2equity_pct = max_open2equity_pct
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
        if self._has_order[side] or open2equity_pct >= self.max_open2equity_pct:
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

from dataclasses import dataclass

@dataclass
class RuleConfig:
    # —— 10) 配置示例 —— 
    long_rule = EntryRule([
        EntryTier(
            name="below_down_poc_and_volspike",
            amount=1,
            conds=[
                OrCondition([
                    AndCondition([
                        # VolumeSpikeCondition("vol_df", "lower"),
                        ConsecutiveCondition("SFrame_vwap_down_poc", "below", 2),
                        # ConsecutiveCondition("HFrame_vwap_down_poc", "below", 5),
                        #价格已经连续 30 根 K 线在 SFrame_vwap_up_getin 之下,以避免短期极强的动能冲击，太早介入可能浮亏比较大。
                        BarsAwayFromThresholdCondition("SFrame_vwap_poc", "below", 50),
                    ]),
                    AndCondition([
                        ConsecutiveCondition("SFrame_vwap_down_sl2", "below", 4),
                        VolumeSpikeCondition("vol_df", "lower"),
                    ])
                ])
            ],
            limit_price_attr="SFrame_vwap_down_poc",
            # limit_price_attr="SFrame_vwap_down_sl2"
        ),

    ])

    short_rule = EntryRule([
        EntryTier(
            name="uppon_up_sl2_and_volspike",
            amount=1,
            conds=[
                OrCondition([
                    AndCondition([
                        # VolumeSpikeCondition("vol_df", "lower"),
                        ConsecutiveCondition("SFrame_vwap_up_poc", "above", 2),
                        # ConsecutiveCondition("HFrame_vwap_up_poc", "above", 5),
                    #     # #价格已经连续 30 根 K 线在 SFrame_vwap_down_getin 之上，以避免短期极强的动能冲击，太早介入可能浮亏比较大。
                        BarsAwayFromThresholdCondition("SFrame_vwap_poc", "above", 50),
                        
                    ]),
                    AndCondition([
                        ConsecutiveCondition("SFrame_vwap_up_sl2", "above", 4),
                        VolumeSpikeCondition("vol_df", "lower"),
                    ])
                ])
            ],
            limit_price_attr="SFrame_vwap_up_sl2",
            # limit_price_attr="HFrame_vwap_up_poc"
        ),
    ])


from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class SimpleTouchCondition:
    trigger_attr: str      # 存储触发线的数据属性名（上边或下边）
    history_touched_attr: str  # 存储历史触及线的数据属性名
    vol_key: str           # 存储成交量的列名
    direction: str         # 方向，'long' 或 'short'

    def check(self, close_series: pd.Series, data_src, cur_close: float) -> bool:
        trigger_line = getattr(data_src, self.trigger_attr)
        history_touched_line = getattr(data_src, self.history_touched_attr)

        # 根据方向找到最近一次触及的历史线
        last_touch_history_line = None
        for i in range(-1, -len(history_touched_line), -1):
            if (self.direction == 'long' and close_series.iloc[i] <= history_touched_line.iloc[i]) or \
            (self.direction == 'short' and close_series.iloc[i] >= history_touched_line.iloc[i]):
                last_touch_history_line = i
                break

        # 如果没有触及过历史线，则返回 False
        if last_touch_history_line is None:
            return False

        # 根据 last_touch_history_line 计算负索引范围
        for j in range(last_touch_history_line, -1, 1):
            # 直接使用负索引 j
            if (self.direction == 'long' and close_series.iloc[j] >= trigger_line.iloc[j]) or \
            (self.direction == 'short' and close_series.iloc[j] <= trigger_line.iloc[j]):
                # 检查在触及时是否放量
                vol_df = getattr(data_src, self.vol_key)
                avg_volume = np.mean(vol_df['lower'].iloc[j])  # 获取触及前5根K线的平均成交量
                current_volume = vol_df['vol'].iloc[j]

                # 若当前成交量大于过去5根K线的均值，返回 True
                if current_volume > avg_volume:
                    return True

        return False  # 如果条件不满足返回 False

    def price(self, close_series: pd.Series, data_src) -> Optional[float]:
        return max(close_series) if self.direction == 'short' else min(close_series)
    

@dataclass
class weakLongRuleConfig:
    long_rule = EntryRule([
        EntryTier(
            name="long_condition",
            amount=1,
            conds=[
                SimpleTouchCondition(
                    trigger_attr="SFrame_vwap_down_poc",  # 用于做多的触发上边线
                    history_touched_attr="SFrame_vwap_up_poc",  # 用于做多的历史触及下边线
                    vol_key="vol_df",
                    direction='long'  # 指定为做多方向
                ),
            ],
            limit_price_attr=None,  # 限价使用触及的最高价
        ),
    ])

    short_rule = EntryRule([
        EntryTier(
            name="short_condition",
            amount=1,
            conds=[
                SimpleTouchCondition(
                    trigger_attr="SFrame_vwap_up_getin",  # 用于做空的触发下边线
                    history_touched_attr="HFrame_vwap_down_poc",  # 用于做空的历史触及上边线
                    vol_key="vol_df",
                    direction='short'  # 指定为做空方向
                ),
            ],
            limit_price_attr=None,  # 限价使用触及的最低价
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