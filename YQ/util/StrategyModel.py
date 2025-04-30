# encoding: utf-8
from makerengine import OrderManager_E,MakerEngine,OrderClass
import time
from constant import *
from vtobject import *
import prettytable as pt
class StrategyModel(MakerEngine):
    def __init__(self):
        super(StrategyModel,self).__init__()
        self.ordermanager = OrderManager_E()
        
    def on_init(self):
        super(StrategyModel,self).on_init()
        pass
    
    def on_tick(self,tick):
        super(StrategyModel,self).on_tick(tick)

    def on_order(self,order):        
        self.ordermanager.order_amend(order)
        self.ordermanager.order_deletedone(order=order)

    def on_trade(self,trade):
        pass

    def on_position(self):
        pass

    def on_error(self):
        pass

    def on_timer(self):
        pass

    def printout(self,*args):
        outstr = str(time.strftime("%H:%M:%S")) + ' '
        for i in args:
            outstr += str(i) + ' '
        print(outstr) 

if __name__ == '__main__':
    pass