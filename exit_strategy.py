import time
from dataclasses import dataclass
from typing import List, Optional, Callable
import operator
import support_resistance
import numpy as np

# —— 配置参数 —— 
STOP_PROFIT         = 0.01    # 中心线止盈最小利润
FEE_REQUIRE_PROFIT  = 0.005   # SFrame 止盈最小利润

# —— 数据结构 —— 
@dataclass
class ExitTier:
    name: str
    # 连续突破条件
    consec_attr:    Optional[str]        # 阈值属性名，如 "HFrame_vwap_up_sl"
    consec_compare: Optional[str]        # "above" 或 "below"
    consec_count:   Optional[int]        # 连续多少根 K 线

    # 价格触及条件
    price_attr:     Optional[str]        # 属性名，如 "center_tp_poc"
    price_compare:  Optional[Callable]   # operator.gt, operator.lt, …

    # 利润门槛
    profit_threshold: Optional[float]
    profit_compare:   Optional[Callable]

    # 平仓手数（根据持仓 pos 计算）
    amount:         Callable[[int], int]
    # 平仓价格取法："cur"=cur_close；"level"=price_attr 对应的值
    price_type:     str                  # "cur" or "level"

@dataclass
class ExitRule:
    tiers: List[ExitTier]

@dataclass
class OrderSignal:
    side:       str        # "long"/"short"
    action:     bool       # True=下单
    price:      float
    amount:     int
    order_type: str        # "market"/"ioc"/"fok"
    order_time: float

# —— 策略类 —— 
class MultiFramePOCStrategy:
    def __init__(self,
                 long_exit_rule: ExitRule,
                 short_exit_rule: ExitRule):
        self.long_exit_rule  = long_exit_rule
        self.short_exit_rule = short_exit_rule

    def evaluate_exit(self,
                      side: str,
                      cur_close: float,
                      profit: float,
                      position: int,
                      close_series,       # pandas.Series
                      data_src            # MultiFrameVWAP/POC 实例
                     ) -> Optional[OrderSignal]:

        rule = self.long_exit_rule if side=="long" else self.short_exit_rule

        for tier in rule.tiers:
            # 1) 连续突破
            if tier.consec_attr:
                thresh = getattr(data_src, tier.consec_attr)
                if tier.consec_compare=="above":
                    ok = support_resistance.consecutive_above_resistance(
                             close_series, thresh, tier.consec_count)
                else:
                    ok = support_resistance.consecutive_below_support(
                             close_series, thresh, tier.consec_count)
                if not ok:
                    continue

            # 2) 价格触及
            level = None
            if tier.price_attr:
                level = getattr(data_src, tier.price_attr)[-1]
                if not tier.price_compare(cur_close, level):
                    continue

            # 3) 利润门槛
            if tier.profit_threshold is not None:
                if not tier.profit_compare(profit, tier.profit_threshold):
                    continue

            # 4) 生成平仓信号
            amt = tier.amount(position)
            if amt <= 0:
                return None

            price = cur_close if tier.price_type=="cur" else level
            return OrderSignal(
                side=side,
                action=True,
                price=price,
                amount=amt,
                order_type="market",
                order_time=time.time()
            )
        return None

# —— 出场规则配置 —— 
from operator import ge, le, gt, lt

short_exit_rule = ExitRule([
    # ① 连续突破 HFrame_vwap_down_sl → 全平
    ExitTier("consec_down_sl", "HFrame_vwap_down_sl", "above", 2,
             None, None, None, None,
             lambda pos: pos, "cur"),
    # ② 中心线止盈
    ExitTier("center_tp", None, None, None,
             "center_tp_poc", lt, STOP_PROFIT, gt,
             lambda pos: 1, "cur"),
    # ③ SFrame 下轨止盈
    ExitTier("sframe_tp", None, None, None,
             "SFrame_vwap_down_poc", le, FEE_REQUIRE_PROFIT, gt,
             lambda pos: min(2, pos), "level"),
    # ④ 常规止损（跌破 HFrame 下轨）
    ExitTier("down_sl", None, None, None,
             "HFrame_vwap_down_sl", le, None, None,
             lambda pos: min(3, pos), "cur"),
])

long_exit_rule = ExitRule([
    # ① 连续突破 HFrame_vwap_up_sl → 全平
    ExitTier("consec_up_sl", "HFrame_vwap_up_sl", "below", 2,
             None, None, None, None,
             lambda pos: pos, "cur"),
    # ② 中心线止盈
    ExitTier("center_tp", None, None, None,
             "center_tp_poc", gt, STOP_PROFIT, gt,
             lambda pos: 1, "cur"),
    # ③ SFrame 上轨止盈
    ExitTier("sframe_tp", None, None, None,
             "SFrame_vwap_up_poc", ge, FEE_REQUIRE_PROFIT, gt,
             lambda pos: min(2, pos), "level"),
    # ④ 常规止损（突破 HFrame 上轨）
    ExitTier("up_sl", None, None, None,
             "HFrame_vwap_up_sl", ge, None, None,
             lambda pos: min(3, pos), "cur"),
])

# —— 示例：初始化与调用 —— 
strategy = MultiFramePOCStrategy(
    long_exit_rule=long_exit_rule,
    short_exit_rule=short_exit_rule
)

# # 在每根 Bar 后：
# sig = strategy.evaluate_exit(
#     side="long",                 # or "short"
#     cur_close=current_price,
#     profit=upl_long/notionalUsd_long,
#     position=amount_long,
#     close_series=close_series,
#     data_src=multi_frame_vwap_obj
# )
# if sig and sig.action:
#     place_order(sig)  # 直接市价/IOC 平仓