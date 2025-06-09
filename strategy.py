import time
from dataclasses import dataclass
from typing import List, Callable, Optional, Set
import support_resistance
import numpy as np

@dataclass
class EntryTier:
    name: str
    price_attr: str                         # e.g. "SFrame_vwap_down_sl"
    # compare: Callable[[float, float], bool] # lambda cur, thresh: cur < thresh or >
    amount: int                             # 挂单手数
    consec_attr: Optional[str]              # e.g. "SFrame_vwap_down_poc"
    consec_compare: Optional[str]           # "above" 或 "below"
    consec_count: Optional[int]             # 连续 bar 数

@dataclass
class EntryRule:
    tiers: List[EntryTier]                  # 按保守→激进排列

@dataclass
class OrderSignal:
    side:       str                         # "long"/"short"
    action:     bool                        # True=发 limit 单, False=无动作
    price:      float                       # 挂单价格
    amount:     int                         # 挂单手数
    order_type: str                         # "limit"
    order_time: float                       # 发单时间戳
    tier_explain: EntryTier                 # 明确解释开仓的原因
class RuleConfig:
    #注意：conservative应该要在前面，因为后续循环索引规则的时候需要先判断突破次数多的情况，
    #     如果顺序是aggressive在前面，conservative可能会因为consec_count更大而一直被aggressive的规则给提前判断return,而无法被循环遍历到。
    long_rule = EntryRule([
        EntryTier(
            name="conservative",
            price_attr="SFrame_vwap_down_sl2",
            amount=2,
            consec_attr="SFrame_vwap_down_poc",
            consec_compare="below",
            consec_count=2
        ),
        EntryTier(
            name="neutral",
            price_attr="SFrame_vwap_down_sl",
            amount=1,
            consec_attr="SFrame_vwap_down_poc",
            consec_compare="below",
            consec_count=2
        ),
        EntryTier(
            name="aggressive",
            price_attr="HFrame_vwap_down_getin",
            amount=1,
            consec_attr="SFrame_vwap_down_poc",
            consec_compare="below",
            consec_count=5
        ),

        EntryTier(
            name="aggressive",
            price_attr="SFrame_vwap_down_poc",
            amount=1,
            consec_attr="SFrame_vwap_down_poc",
            consec_compare="below",
            consec_count=10
        ),
        
    ])

    short_rule = EntryRule([
        EntryTier(
            name="conservative",
            price_attr="SFrame_vwap_up_sl2",
            amount=2,
            consec_attr="SFrame_vwap_up_poc",
            consec_compare="above",
            consec_count=2
        ),
        EntryTier(
            name="neutral",
            price_attr="SFrame_vwap_up_sl",
            amount=1,
            consec_attr="SFrame_vwap_up_poc",
            consec_compare="above",
            consec_count=2
        ),
        EntryTier(
            name="aggressive",
            price_attr="HFrame_vwap_up_getin",
            amount=1,
            consec_attr="SFrame_vwap_up_poc",
            consec_compare="above",
            consec_count=5
        ),
        
        EntryTier(
            name="aggressive",
            price_attr="SFrame_vwap_up_poc",
            amount=1,
            consec_attr="SFrame_vwap_up_poc",
            consec_compare="above",
            consec_count=10
        ),
    ])

class MultiFramePOCStrategy:
    def __init__(self,
                 long_rule: EntryRule,
                 short_rule: EntryRule,
                 timeout: float = 60.0):
        self.long_rule   = long_rule
        self.short_rule  = short_rule
        self.timeout     = timeout

        # 哪些 tier 已挂过，防止重复
        self._opened: dict[str, Set[str]] = {
            "long": set(), "short": set()
        }
        # 当前侧有没有挂单（超时前都视为“有挂单”）
        self._has_order: dict[str, bool] = {
            "long": False, "short": False
        }
        # 挂单时间戳
        self._order_time: dict[str, Optional[float]] = {
            "long": None, "short": None
        }

    def evaluate(self,
                 side: str,
                 cur_close: float,
                 close_series,            # 你的 close 序列
                 data_src,                # multFramevp_poc 对象
                 record_buy_total        # 你的 record_buy_total_*（内部不使用，仅配接口）
        )-> Optional[OrderSignal]:
        """        
        1) 如果已有挂单且未超时 → 返回 None  
        2) 否则遍历 tiers，第一个满足条件的 → 标记挂单并返回 OrderSignal  
        """
        # 1) 如果已有挂单，且没超时，跳过
        if self._has_order[side]:
            # 外部会继续调用 should_cancel 判断是否超时
            return None
        if record_buy_total >= 20:
            return None  #最多开20个订单，以免爆仓。
        
        rule   = self.long_rule if side == "long" else self.short_rule
        opened = self._opened[side]
        now    = time.time()

        for tier in rule.tiers:
            if tier.name in opened:
                continue

            # (1) 连续突破/破位
            if tier.consec_attr:
                thresh = getattr(data_src, tier.consec_attr)
                if tier.consec_compare == "above":
                    ok = support_resistance.consecutive_above_resistance(
                        close_series, thresh, tier.consec_count)
                else:
                    ok = support_resistance.consecutive_below_support(
                        close_series, thresh, tier.consec_count)
                if not ok:
                    continue

            # (2) 价格触及
            if tier.consec_compare == "above":
                price_t = np.max(getattr(data_src, tier.price_attr).iloc[-20:])
            else:
                price_t = np.min(getattr(data_src, tier.price_attr).iloc[-20:])
            # if not tier.compare(cur_close, price_t):
            #     continue

            # (3) 触发挂单
            opened.add(tier.name)
            self._has_order[side]  = True
            self._order_time[side] = now
            return OrderSignal(
                side=side,
                action=True,
                price=price_t,
                amount=tier.amount,
                order_type="limit",
                order_time=now,
                tier_explain=tier
            )

        # 没触发
        return None

    def should_cancel(self, side: str) -> bool:
        """已有挂单且超时，就撤单。"""
        if not self._has_order[side]:
            return False

        ts = self._order_time[side]
        if ts is None:
            # 如果意外没记时间，可选地当“肯定要撤”
            return True

        # 超时就撤，并重置时间戳
        if time.time() - ts >= self.timeout:
            self._order_time[side] = time.time()  # 如果要继续监控，再重置
            return True

        return False

    def clear_order(self, side: str):
        """外部撤单或成交回报时调用，清除挂单状态"""
        self._has_order[side]  = False
        self._order_time[side] = None

        self._opened[side].clear()

