import pandas as pd
from datetime import datetime

class Portfolio:
    def __init__(self, init_cash, margin_rate=0.05):
        self.fee_rate = 15/10000
        self.cash = init_cash      # 可用现金
        self.margin = 0.0          # 冻结保证金
        self.position_long = 0
        self.position_short = 0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0
        self.margin_rate = margin_rate  # 保证金比例，默认0.5表示2倍杠杆
        self.history = []
        self.trade_log = []
        
        self.total_asset = init_cash

    def upnl(self, price):
        upnl_long  = (price - self.avg_price_long)  * self.position_long if self.position_long  > 0 else 0.0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0.0
        return upnl_long + upnl_short

    def free_margin(self, price):
        # 全仓：可用保证金 = 现金 + 未实现盈亏（初始保证金已从现金锁定到 self.margin）
        return self.cash + self.upnl(price)

    def update(self, price, long_change=0, short_change=0, cur_time=None, action=None):
        rpnl = 0  # 已实现盈亏（平仓时生效）

        if long_change > 0:
            fee_rate = getattr(self, "fee_rate", 0.0)  # 若无手续费，可不设置或设为0
            cost = price * long_change * self.margin_rate   # 初始保证金（固定锁定）
            fee  = price * long_change * fee_rate           # 手续费（可选）
            free = self.free_margin(price)
            need = cost + fee

            if free >= need:
                prev = self.position_long
                self.position_long += long_change
                # 加权平均开仓价（用于后续UPnL；保证金基数也随均价/仓位变化）
                self.avg_price_long = ((self.avg_price_long * prev) + price * long_change) / self.position_long

                # 记账：锁定保证金到 margin（固定），现金扣除 保证金+手续费
                self.cash   -= need
                self.margin += cost
            else:
                print(f'Free margin not enough for {action} (long): need {need:.3f}, free={free:.3f}')
                return self.get_total_value(price)

        elif long_change < 0:  # 平多头仓
            close_amount = -long_change
            if close_amount > self.position_long:
                close_amount = self.position_long
            rpnl = (price - self.avg_price_long) * close_amount  * (1 - self.fee_rate)
            self.position_long -= close_amount
            margin_return = self.avg_price_long * close_amount * self.margin_rate
            self.cash += margin_return + rpnl
            self.margin -= margin_return

            if self.position_long == 0:
                self.avg_price_long = 0.0

        if short_change > 0:
            # 可选：手续费
            fee_rate = getattr(self, "fee_rate", 0.0)
            cost = price * short_change * self.margin_rate         # 初始保证金（固定锁定）
            fee  = price * short_change * fee_rate                 # 手续费（若有）

            free = self.free_margin(price)
            need = cost + fee
            if free >= need:
                prev = self.position_short
                self.position_short += short_change
                # 加权平均开仓价（用于计算UPnL；注意：均价改变 => 未来“固定保证金”的基数也随之改变）
                self.avg_price_short = ((self.avg_price_short * prev) + price * short_change) / self.position_short

                # 记账：锁定保证金到 margin（固定），现金扣除保证金+手续费
                self.cash   -= need
                self.margin += cost
            else:
                print(f'Free margin not enough for {action} (short): need {need:.3f}, free={free:.3f}')
                return self.get_total_value(price)

        elif short_change < 0:  # 平空头仓
            close_amount = -short_change
            if close_amount > self.position_short:
                close_amount = self.position_short
            rpnl = (self.avg_price_short - price) * close_amount * (1 - self.fee_rate)
            self.position_short -= close_amount
            margin_return = self.avg_price_short * close_amount * self.margin_rate
            self.cash += margin_return + rpnl
            self.margin -= margin_return
            if self.position_short == 0:
                self.avg_price_short = 0.0

        # 未实现盈亏计算
        upnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0

        # 总资产=现金+冻结保证金+未实现盈亏
        self.total_asset = self.cash + self.margin + upnl_long + upnl_short

        # 时间处理
        if cur_time is not None:
            cur_time_dt = datetime.utcfromtimestamp(cur_time) if isinstance(cur_time, (int, float)) else cur_time
        else:
            cur_time_dt = datetime.now()
        self.history.append((cur_time_dt, self.total_asset))

        # 记录交易日志（只在有动作时记录）
        if action is not None and (long_change != 0 or short_change != 0):
            record = {
                'datetime': cur_time_dt,
                'action': action,
                'price': round(price, 3),
                'amount': round(long_change if long_change != 0 else short_change, 2),
                'position_long': round(self.position_long, 2),
                'position_short': round(self.position_short, 2),
                'cash': round(self.cash, 3),
                'margin': round(self.margin, 3),
                'asset': round(self.total_asset, 3),
                'rpnl': round(rpnl, 3),
                'upnl_long': round(upnl_long, 3),
                'upnl_short': round(upnl_short, 3),
            }
            self.trade_log.append(record)
            # if rpnl != 0:
            print(record)

        return self.total_asset

    def get_total_value(self, price):
        upnl_long = (price - self.avg_price_long) * self.position_long if self.position_long > 0 else 0
        upnl_short = (self.avg_price_short - price) * self.position_short if self.position_short > 0 else 0
        return self.cash + self.margin + upnl_long + upnl_short
