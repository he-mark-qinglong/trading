
class OrderSignal:
    def __init__(self, side, action, price, amount, order_type="limit", order_time=None, tier_explain=""):
        self.side = side
        self.action = action
        self.price = price
        self.amount = amount
        self.order_type = order_type
        self.order_time = order_time
        self.tier_explain = tier_explain