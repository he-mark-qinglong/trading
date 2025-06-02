import time
from dataclasses import dataclass
from typing import List, Callable, Optional, Set

@dataclass
class EntryTier:
    name: str
    price_attr: str                         # e.g. "SFrame_vwap_down_sl"
    compare: Callable[[float, float], bool] # lambda cur, thresh: cur < thresh or >
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

class MultiFramePOCStrategy:
    def __init__(self,
                 long_rule: EntryRule,
                 short_rule: EntryRule,
                 support_resistance,
                 timeout: float = 20.0):
        self.long_rule   = long_rule
        self.short_rule  = short_rule
        self.sr          = support_resistance
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
        if len(record_buy_total) >= 20:
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
                    ok = self.sr.consecutive_above_resistance(
                        close_series, thresh, tier.consec_count)
                else:
                    ok = self.sr.consecutive_below_support(
                        close_series, thresh, tier.consec_count)
                if not ok:
                    continue

            # (2) 价格触及
            price_t = getattr(data_src, tier.price_attr)
            if not tier.compare(cur_close, price_t):
                continue

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
                order_time=now
            )

        # 没触发
        return None

    def should_cancel(self, side: str) -> bool:
        """如果已有挂单且超时，返回 True"""
        if not self._has_order[side]:
            return False
        return (time.time() - (self._order_time[side] or 0)) >= self.timeout

    def clear_order(self, side: str):
        """外部撤单或成交回报时调用，清除挂单状态"""
        self._has_order[side]  = False
        self._order_time[side] = None

'''
SFrame_vwap_up_getin = multFramevp_poc.SFrame_vwap_up_getin.iloc[-1]
                SFrame_vwap_down_getin = multFramevp_poc.SFrame_vwap_down_getin.iloc[-1]

                SFrame_vwap_up_poc = multFramevp_poc.SFrame_vwap_up_poc.iloc[-1]
                SFrame_vwap_down_poc = multFramevp_poc.SFrame_vwap_down_poc.iloc[-1]
                SFrame_vwap_down_sl = multFramevp_poc.SFrame_vwap_down_sl.iloc[-1]
                SFrame_vwap_up_sl = multFramevp_poc.SFrame_vwap_up_sl.iloc[-1]

                HFrame_vwap_down_getin = multFramevp_poc.HFrame_vwap_down_getin.iloc[-1]
                HFrame_vwap_up_getin = multFramevp_poc.HFrame_vwap_up_getin.iloc[-1]
                HFrame_vwap_up_poc = multFramevp_poc.HFrame_vwap_up_poc.iloc[-1]
                HFrame_vwap_down_poc = multFramevp_poc.HFrame_vwap_down_poc.iloc[-1]
                HFrame_vwap_down_sl = multFramevp_poc.HFrame_vwap_down_sl.iloc[-1]
                HFrame_vwap_up_sl = multFramevp_poc.HFrame_vwap_up_sl.iloc[-1]

                cur_high = hh2.iloc[-1]
                is_short_un_opend = len(record_buy_total_short) == 0
                hard_short = False
                if hard_short:
                    if is_short_un_opend:
                        conecutive_above_s = support_resistance.consecutive_above_resistance(close, multFramevp_poc.SFrame_vwap_up_getin, 10)
                        #首次开仓要大周期和中周期的getin都触摸才算。补仓则是价格比开仓价格更优并且触摸中周期的getin
                        close_above_vwap = cur_close > SFrame_vwap_up_poc
                        if  ( conecutive_above_s and close_above_vwap) or (cur_close >= HFrame_vwap_up_sl*1.001): 
                            multiFrame_vp_poc_short=1
                            self.short_order_record_time = time.time()

                            print("open", "-"*100)
                    else:  #加仓条件
                        betterThanPreLong = float(record_buy_total_short['price'].iloc[-1]) < cur_close  #更高的价格才加空仓
                        time_cond =  time.time() - float(record_buy_total_short['record_time'].iloc[-1]) > 20*len(record_buy_total_short)
                        conecutive_above_s = support_resistance.consecutive_above_resistance(close, multFramevp_poc.SFrame_vwap_up_getin, 7 )
                        close_above_vwap = (cur_close > SFrame_vwap_up_poc)
                        if ( conecutive_above_s and close_above_vwap)\
                                and time_cond and betterThanPreLong:
                            multiFrame_vp_poc_short=1

                            self.short_order_record_time = time.time()
                            print("append open","-"*100)
                else:
                    if is_short_un_opend:
                        conecutive_above_s = support_resistance.consecutive_above_resistance(close, multFramevp_poc.SFrame_vwap_up_poc, 10)
                        #首次开仓要大周期和中周期的getin都触摸才算。补仓则是价格比开仓价格更优并且触摸中周期的getin
                        close_above_getin = cur_close > SFrame_vwap_up_getin 
                        if  ( conecutive_above_s and close_above_getin) or (cur_close >= HFrame_vwap_up_sl*0.999): 
                            multiFrame_vp_poc_short=1
                            self.short_order_record_time = time.time()

                            print("open", "-"*100)
                    else:  #加仓条件
                        betterThanPreLong = float(record_buy_total_short['price'].iloc[-1]) < cur_close  #更高的价格才加空仓
                        time_cond =  time.time() - float(record_buy_total_short['record_time'].iloc[-1]) > 20*len(record_buy_total_short)
                        conecutive_above_s = support_resistance.consecutive_above_resistance(close, multFramevp_poc.SFrame_vwap_up_poc, 7 )
                        close_above_getin = (cur_close > SFrame_vwap_up_getin)
                        if ( conecutive_above_s and close_above_getin)\
                                and time_cond and betterThanPreLong:
                            multiFrame_vp_poc_short=1

                            self.short_order_record_time = time.time()
                            print("append open","-"*100)
                    
                is_long_un_opend = len(record_buy_total_long) == 0
                if is_long_un_opend:
                    consecutive_break_resistance_l = support_resistance.consecutive_below_support(close, multFramevp_poc.SFrame_vwap_down_poc, 10)
                    #首次开仓要大周期和中周期的getin都触摸才算。补仓则是价格比开仓价格更优并且触摸中周期的getin
                    close_below_getin =  SFrame_vwap_down_getin > cur_close
                    if (consecutive_break_resistance_l and close_below_getin) or (SFrame_vwap_down_sl >= cur_close and HFrame_vwap_down_sl >= cur_close):  
                        
                        multiFrame_vp_poc_long=1
                        self.long_order_record_time = time.time()

                        print("open", "+"*100)
                else:  #加仓条件
                    betterThanPreShort = float(record_buy_total_long['price'].iloc[-1])  > cur_close  #更低的价格才加多仓
                    time_cond = time.time()-float(record_buy_total_long['record_time'].iloc[-1]) > 20*len(record_buy_total_long)
                    consecutive_break_resistance_l = support_resistance.consecutive_below_support(close, multFramevp_poc.SFrame_vwap_down_poc, 7)
                    close_below_getin = SFrame_vwap_down_getin > cur_close
                    if (consecutive_break_resistance_l and close_below_getin) \
                        and time_cond and betterThanPreShort:

                        multiFrame_vp_poc_long=1
                        self.long_order_record_time = time.time()
                        print("append open", "+"*100)
'''