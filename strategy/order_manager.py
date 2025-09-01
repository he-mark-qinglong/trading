from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class LimitOrder:
    action: str               # "open_long" / "close_long" / "open_short" / "close_short" / "stop_loss_long" / "stop_loss_short"
    side: str                 # "long" or "short"
    price: float
    qty: float
    created_pos: int
    created_ts: Union[int, float]
    ttl_bars: int = 160
    id: int = field(default_factory=int)
    status: str = "pending"   # pending / filled / canceled / expired
    filled_pos: Optional[int] = None
    filled_ts: Optional[Union[int, float]] = None
    filled_price: Optional[float] = None
    note: str = ""

    @property
    def activate_pos(self) -> int:
        return self.created_pos + 1

    @property
    def expire_pos(self) -> int:
        return self.created_pos + self.ttl_bars

    def is_buy(self) -> bool:
        return self.action in ("open_long", "close_short", "stop_loss_short")  # 添加 stop_loss_short（假设它是买单平空）

    def is_sell(self) -> bool:
        return self.action in ("open_short", "close_long", "stop_loss_long")   # 添加 stop_loss_long（假设它是卖单平多）

    def is_close(self) -> bool:
        return self.action in ("close_long", "close_short", "stop_loss_long", "stop_loss_short")  # 添加 stop_loss


class OrderManager:
    def __init__(self, ttl_bars: int = 100):
        self.ttl_bars = ttl_bars
        self._next_id = 1
        self.pending: Dict[int, LimitOrder] = {}
        self.filled: List[LimitOrder] = []
        self.canceled: List[LimitOrder] = []
        self.expired: List[LimitOrder] = []

    def submit_limit(self, action: str, side: str, price: float, qty: float, created_pos: int, created_ts: Union[int, float], note: str = "") -> LimitOrder:
        if qty <= 0:  # 新增：无效 qty 校验
            raise ValueError("Order qty must be positive")
        order = LimitOrder(
            action=action,
            side=side,
            price=float(price),
            qty=float(qty),
            created_pos=created_pos,
            created_ts=created_ts,
            ttl_bars=self.ttl_bars,
            id=self._next_id,
            note=note,
        )
        self._next_id += 1
        self.pending[order.id] = order
        return order

    def cancel(self, order_id: int, reason: str = "manual"):
        order = self.pending.pop(order_id, None)
        if order:
            order.status = "canceled"
            order.note = f"{order.note} | cancel:{reason}"
            self.canceled.append(order)

    def cancel_pending_by(self, action: Optional[str] = None, side: Optional[str] = None, reason: str = "replace"):
        to_cancel = [oid for oid, o in self.pending.items()
                     if (action is None or o.action == action) and (side is None or o.side == side)]
        for oid in to_cancel:
            self.cancel(oid, reason)

    def has_pending(self, action: str, side: str) -> bool:
        return any(o.action == action and o.side == side for o in self.pending.values())

    def _should_fill_price(self, order: LimitOrder, bar_row) -> bool:
        if order.is_buy():
            return bar_row["low"] <= order.price
        else:
            return bar_row["high"] >= order.price

    def _has_closable_position(self, order: LimitOrder, long_qty: float, short_qty: float) -> bool:
        # 平多/止损多需要有多头；平空/止损空需要有空头；开仓不需要持仓
        if order.action in ("close_long", "stop_loss_long"):
            return (long_qty or 0.0) > 0.0
        if order.action in ("close_short", "stop_loss_short"):
            return (short_qty or 0.0) > 0.0
        return True

    def on_bar(self, cur_pos: int, cur_ts: Union[int, float], bar_row,
               long_qty: float = 0.0, short_qty: float = 0.0) -> List[LimitOrder]:
        filled_now: List[LimitOrder] = []
        to_remove: List[int] = []

        avail_long = float(long_qty or 0.0)
        avail_short = float(short_qty or 0.0)

        for oid, order in list(self.pending.items()):
            
            if cur_pos < order.activate_pos:
                continue

            if cur_pos >= order.expire_pos:
                order.status = "expired"
                order.note = f"{order.note} | expired@{cur_pos}"
                self.expired.append(order)
                to_remove.append(oid)
                continue

            price_ok = self._should_fill_price(order, bar_row)
            # print(f"  Price check: {price_ok} (is_buy={order.is_buy()}, low={bar_row.get('low')}, high={bar_row.get('high')})")
            if not price_ok:
                continue

            pos_ok = self._has_closable_position(order, avail_long, avail_short)
            # print(f"  Position check: {pos_ok} (avail_long={avail_long}, avail_short={avail_short})")
            if not pos_ok:
                self.cancel(oid, reason="no_position_to_close")  # 改为直接取消，防止遗留
                continue

            # if order.action in ("stop_loss_long"):
            #     print('has stop_loss_long in pending')

            # 限制平仓数量（扩展到 stop_loss）
            if order.action in ("close_long", "stop_loss_long"):
                if order.qty > avail_long:
                    order.note = f"{order.note} | clipped:{order.qty}->{avail_long}"
                    order.qty = avail_long
                avail_long -= order.qty

                # print(f'order.action {order.action }')

            elif order.action in ("close_short", "stop_loss_short"):
                if order.qty > avail_short:
                    order.note = f"{order.note} | clipped:{order.qty}->{avail_short}"
                    order.qty = avail_short
                avail_short -= order.qty

                # print(f'order.action {order.action }')
            order.status = "filled"
            order.filled_pos = cur_pos
            order.filled_ts = cur_ts
            order.filled_price = order.price
            filled_now.append(order)
            self.filled.append(order)
            to_remove.append(oid)

        for oid in to_remove:
            self.pending.pop(oid, None)

        # 清理剩余平仓/止损挂单（扩展到 stop_loss）
        if avail_long <= 0.0:
            self.cancel_pending_by(action=None, side="long", reason="no_long_position")  # action=None 取消所有 long side（包括 close/stop_loss）
        if avail_short <= 0.0:
            self.cancel_pending_by(action=None, side="short", reason="no_short_position")

        return filled_now