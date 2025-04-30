# encoding: utf-8
import time
import threading
import json
import logging
import numpy as np
import random
from constant import *
from vtobject import *


class MakerEngine(object):
    def __init__(self):
        self.ontimerperiod = 0.5
        
        self.log = logging
        self.logger = self.log.getLogger(__name__)
        handler = self.log.FileHandler("log.txt")
        self.logger.addHandler(handler)


        self.flag = {
            'tickrec':False,
            'init':False
        }
        self.timeT = threading.Thread(target=self._on_timer)
        self.timeT.start()
        self.cancelT = threading.Thread(target=self._thread_cancel)
        self.cancelT.start()
        self.calcuT = threading.Thread(target=self._thread_calculate)
        self.calcuT.start()
        self.selftradeT = threading.Thread(target=self._thread_selftrade)
        self.selftradeT.start()
        self.hedgeT = threading.Thread(target=self._thread_hedge)
        self.hedgeT.start()

    def on_init(self):#没有使用？
        pass 

    def _thread_hedge(self):
        while True:
            try:
                self.thread_hedge()
            except:
                pass
            time.sleep(self.ontimerperiod)

    def thread_hedge(self):
        pass

    def _thread_selftrade(self):
        while True:
            try:
                self.thread_selftrade()
            except:
                pass
            time.sleep(self.ontimerperiod)

    def thread_selftrade(self):
        pass

    def _thread_calculate(self):
        while True:
            try:
                self.thread_calculate()
            except:
                pass
            time.sleep(self.ontimerperiod)

    def thread_calculate(self):
        pass

    def thread_cancel(self):
        pass

    def _thread_cancel(self):
        while True:
            try:
                self.thread_cancel()
            except:
                pass
            time.sleep(self.ontimerperiod)

    def on_tick(self,tick):
        self.flag['tickrec'] = True#tickrec 获取到数据的标志

    def _on_timer(self):
        while True:
            if not self.flag['init'] and self.flag['tickrec']:#flag['init']？，flag['tickrec']？
                self.flag['init'] = True
                self.on_init()

            if self.flag['tickrec']:
                try:
                    self.on_timer()
                except:
                    pass
            else:
                # self.log.warning('no tick...')
                pass
            time.sleep(self.ontimerperiod)

    def on_timer(self):
        pass

    def symbolconver_STE(self,symbol,exchange):
        'input BTC_USDT'
        strlist = symbol.split('_')
        if exchange == Exchange.OKEX:
            return strlist[0]+'-'+strlist[1]
        if exchange == Exchange.MXC or exchange == Exchange.BETAEX:
            return symbol
        if exchange == Exchange.BITMAX:
            return strlist[0]+'/'+strlist[1]
        if exchange == Exchange.WTZ:
            return strlist[0]+'/'+strlist[1]
        if exchange == Exchange.HUOBI:
            return (strlist[0]+strlist[1]).lower()
        if exchange ==  Exchange.GOOD:
            return symbol.lower()
        if exchange == Exchange.BITMEX:
            if 'btc' in symbol.lower():
                return 'XBTUSD'
            if 'eth' in symbol.lower():
                return 'ETHUSD'
        if exchange == Exchange.NEWTON:
            return (strlist[0]+strlist[1]).lower()
        if exchange == Exchange.COINW:
            if 'BCH' in symbol:
                if 'USDT' in symbol:
                    return 'BCHABC/USDT'
                if 'CNYT' in symbol:
                    return 'BCHABC/CNYT'
            return strlist[0] + '/' + strlist[1]
        if exchange == Exchange.BIGONE:
            return strlist[0] + '-' + strlist[1]
        if exchange == Exchange.VC:
            return strlist[0] + '-' + strlist[1]


    def symbolconver_ETS(self,symbol,exchange):
        '"out put BTC_USDT"'
        if exchange == Exchange.OKEX:
            strlist = symbol.split('-')
            return strlist[0]+'_'+strlist[1]
        if exchange == Exchange.MXC or exchange == Exchange.BETAEX:
            return symbol
        if exchange == Exchange.BITMAX:
            strlist = symbol.split('/')
            return strlist[0]+'_'+strlist[1]
        if exchange == Exchange.WTZ:
            strlist = symbol.split('/')
            return strlist[0]+'_'+strlist[1]
        if exchange == Exchange.HUOBI:
            if symbol[-4:] == 'usdt':
                return symbol[:-4].upper() + '_' + 'USDT'
            if symbol[-3:] == 'btc':
                return symbol[:-3].upper() + '_' + 'BTC'
            if symbol[-3:] == 'eth':
                return symbol[:-3].upper() + '_' + 'ETH'
        if exchange ==  Exchange.GOOD:
            return symbol.upper()
        if exchange == Exchange.BITMEX:
            if symbol == 'XBTUSD':
                return 'BTC_USDT'
            if symbol == 'ETHUSD':
                return 'ETH_USDT'
        if exchange == Exchange.NEWTON:
            if symbol[-2:] == 'nt':
                return symbol[:-2].upper() + '_' + 'NT'
        if exchange == Exchange.COINW:
            strlist = symbol.split('/')
            if 'BCH' in symbol:
                return 'BCH_USDT'
            return strlist[0] + '_' + strlist[1]
        if exchange == Exchange.COINW:
            strlist = symbol.split('-')
            return strlist[0] + '_' + strlist[1]
        if exchange == Exchange.VC:
            strlist = symbol.split('-')
            return strlist[0] + '_' + strlist[1]
        return symbol

    def w_log(self,*data):
        wdata = str(time.ctime())+' '
        for i in data:
            wdata += str(i) + ' '

        self.logger.warning(wdata)

class OrderClass(VtOrderData):
    def __init__(self):
        super(OrderClass,self).__init__()
        self.canceltimes = 0                # 撤单次数
        self.canceldatetime = 0             # 最后撤单时间
        self.createtime = None              # 订单创建时间
        self.orderflag = None               # 订单标识(所属策略及参数)  
        self.nature = None                  # 其它属性
        self.marketlevel = None             # 档次要求

class OrderManager_E(object):
    def __init__(self):
        self.orderdict = {}
        self.position = {}

    def clear_order(self):
        self.orderdict = {}

    def order_add(self,order):
        if order.orderid:
            if order.orderid not in self.orderdict:
                self.orderdict[order.orderid] = order
            else:
                print ('order exist',order.symbol,order.direction,order.price,order.quantity)
        if order.clientid:
            if order.clientid not in self.orderdict:
                self.orderdict[order.clientid] = order
            else:
                print ('order exist',order.symbol,order.direction,order.price,order.quantity)

    def has_order(self,orderid):
        ishas = False
        if orderid in self.orderdict:
            ishas = True
        return ishas

    def get_allorderid(self):
        odidlist = list(self.orderdict.keys())
        return odidlist

    def delete_order(self,orderid):
        if orderid in self.orderdict:
            del self.orderdict[orderid]

    def get_pricelist(self,direction):
        try:
            pricelist = [self.orderdict[orderid].price for orderid in self.orderdict if
                        self.orderdict[orderid].direction == direction and not self.orderdict[orderid].nature]
        except:
            pricelist = None
        return pricelist

    def get_vollist(self,direction):
        try:
            vollist = [self.orderdict[orderid].quantity for orderid in self.orderdict if
                        self.orderdict[orderid].direction == direction and not self.orderdict[orderid].nature]
        except:
            vollist = None
        return vollist

    def get_orderidlist_inpriceinterval(self,interval):
        minprice = min(interval)
        maxprice = max(interval)
        try:
            orderidlist = [orderid for orderid in self.orderdict if minprice<=self.orderdict[orderid].price<=maxprice]
        except:
            orderidlist = None
        return orderidlist

    def get_totalvol(self,direction):
        vollist = self.get_vollist(direction)
        totalvol = 0
        if vollist:
            totalvol = sum(vollist)
        return totalvol

    def get_maxvol(self,direction):
        vollist = self.get_vollist(direction)
        maxvol = 0
        if vollist:
            maxvol = max(vollist)
        return maxvol

    def get_bidaskprice(self):
        bidprice = None
        askprice = None
        buypricelist = self.get_pricelist(Direction.LONG)
        sellpricelist = self.get_pricelist(Direction.SHORT)
        if buypricelist:
            bidprice = max(buypricelist)
        if sellpricelist:
            askprice = min(sellpricelist)

        return [bidprice,askprice]

    def get_L1_orderid(self):
        buypricelist = self.get_pricelist(Direction.LONG)
        sellpricelist = self.get_pricelist(Direction.SHORT)
        try:
            maxbuyprice = max(buypricelist)
            minsellprice = min(sellpricelist)
        except:
            maxbuyprice = None
            minsellprice = None
        buyid = None
        sellid = None
        buyorder = None
        sellorder = None
        if maxbuyprice and minsellprice:
            for orderid in self.orderdict:
                order  = self.orderdict[orderid]
                if order.direction == Direction.LONG and order.price == maxbuyprice:
                    buyid = orderid
                    buyorder = order
                if order.direction == Direction.SHORT and order.price == minsellprice:
                    sellid = orderid
                    sellorder = order
            return buyid,buyorder,sellid,sellorder
        else:
            return None,None,None,None

    def get_LN_price(self,N):
        N = N-1
        buypricelist = self.get_pricelist(Direction.LONG)
        sellpricelist = self.get_pricelist(Direction.SHORT)
        buypricelist = sorted(buypricelist)
        sellpricelist = sorted(sellpricelist)
        try:
            Nbuy = buypricelist[-N]
            Nsell = sellpricelist[N]
        except:
            Nbuy = None
            Nsell = None
        # print(Nbuy,Nsell)
        return Nbuy,Nsell

    def get_N_totalvol(self,N):
        nbuyp,nsellp = self.get_LN_price(N)
        # print(nbuyp,nsellp)
        Nbuyvol = None
        Nsellvol = None
        if nbuyp and nsellp:
            Nbuyvol = 0
            Nsellvol = 0
            for orderid in self.orderdict:
                order  = self.orderdict[orderid]
                if order.direction == Direction.LONG and order.price>=nbuyp:
                    Nbuyvol += order.quantity
                if order.direction == Direction.SHORT and order.price<=nsellp:
                    Nsellvol += order.quantity
        # print(Nbuyvol,Nsellvol)
        return Nbuyvol,Nsellvol

    def get_ordernumber(self,direction = None):
        if direction is None:
            buylist = self.getorderid_fromdirection(Direction.LONG)
            selllist = self.getorderid_fromdirection(Direction.SHORT)
            return  len(buylist),len(selllist)
        if direction is not None:
            orderlist = self.getorderid_fromdirection(direction)
            return len(orderlist)

    def getorderid_fromdirection(self,direction):
        try:
            orderidlist = [orderid for orderid in self.orderdict if self.orderdict[orderid].direction == direction]
        except:
            orderidlist = []
        return orderidlist

    def get_mindistance(self,price,direction):
        mindis = 0.00
        pricelist = self.get_pricelist(direction)
        if pricelist:
            dislist = [abs(price - i) for i in pricelist]
            mindis = min(dislist)
        if not pricelist:
            mindis = None
        return mindis

    def order_amend(self,order):#order ，<LQS.util.vtobject.VtOrderData at 0x9f774a8>
        # if order.status == Status.ALLTRADED or order.status == Status.PARTTRADED:
        #     print('---new trade',self.has_order(order.orderid),order.status,order.price,order.quantity)
        if order and order.orderid not in self.orderdict:
            # print('--unknow order:',order.orderid,order.status)
            pass

        if order.orderid in self.orderdict:# <LQS.util.makerengine.OrderClass at 0xa0b4fd0>
            self.orderdict[order.orderid].status = order.status #Status.NOTTRADED ？
            self.orderdict[order.orderid].tradevol = order.tradevol
            self.orderdict[order.orderid].filledquantity = order.filledquantity#成交量

            # if order.status == Status.ALLTRADED or order.status == Status.PARTTRADED:
            #     print('---amend',order.orderid,order.status,order.tradevol,order.quantity)
            
            # if order.symbol not in self.position:
            #     self.position[order.symbol] = {
            #                         'buyvol':0,
            #                         'sellvol':0,
            #                         'buyprice':0,
            #                         'sellprice':0,
            #                         }
            # if order.direction == Direction.LONG and (order.status ==  Status.ALLTRADED or order.status == Status.PARTTRADED):
            #     totalvol = self.position[order.symbol]['buyvol'] + self.orderdict[order.orderid].tradevol
            #     self.position[order.symbol]['buyprice'] = (self.position[order.symbol]['buyprice']*self.position[order.symbol]['buyvol'] + self.orderdict[order.orderid].tradevol*order.price)/totalvol
            #     self.position[order.symbol]['buyvol'] = self.orderdict[order.orderid].tradevol

            # if order.direction == Direction.SHORT and (order.status ==  Status.ALLTRADED or order.status == Status.PARTTRADED):
            #     totalvol = self.position[order.symbol]['sellvol'] + self.orderdict[order.orderid].tradevol
            #     self.position[order.symbol]['sellprice'] = (self.position[order.symbol]['sellprice']*self.position[order.symbol]['sellvol'] + self.orderdict[order.orderid].tradevol*order.price)/totalvol
            #     self.position[order.symbol]['sellvol'] = self.orderdict[order.orderid].tradevol

    def order_deletedone(self,order = None):
        # neworderidct = {}
        # for clientid in self.orderdict:
        #     state = self.orderdict[clientid].status
        #     if state != Status.ALLTRADED and state != Status.CANCELLED and state != Status.REJECTED:
        #         neworderidct[clientid] = self.orderdict[clientid]
        # self.orderdict = neworderidct
        
        if order and order.orderid in self.orderdict:
            state = order.status
            if state == Status.ALLTRADED or state == Status.CANCELLED or state == Status.PARTCANCELED or state == Status.REJECTED:
                del self.orderdict[order.orderid]

    def get_status(self):
        state = [self.orderdict[clientid].status for clientid in self.orderdict if not self.orderdict[clientid].nature]
        return state

    def get_clientid_levelone(self):
        [bidprice,askprice] = self.get_bidaskprice()
        buylist = []
        selllist = []
        for clientid in self.orderdict:
            if self.orderdict[clientid].price == bidprice:
                buylist.append(clientid)
            if self.orderdict[clientid].price == askprice:
                selllist.append(clientid)
        return buylist,selllist

    def getorder_from_orderid(self,orderid):
        return self.orderdict[orderid]

    def get_orderidlist_frompriceinterval(self,direction):
        clientlist = []
        pricelist = self.get_pricelist(direction)
        midp = np.median(pricelist)
        for clientid in self.orderdict:
            if direction == Direction.LONG and not self.orderdict[clientid].nature:
                if self.orderdict[clientid].price <= midp:
                    clientlist.append(clientid)
            if direction == Direction.SHORT and not self.orderdict[clientid].nature:
                if self.orderdict[clientid].price >= midp:
                    clientlist.append(clientid)
        return clientlist

    def __iter__(self):
        return iter(self.orderdict.items())

    def create_clientid(self):
        ordernum = self.get_ordernumber()
        clientid = 'LQ_' + str(int(time.time() * 1000)) + '_' + str(sum(ordernum))
        return clientid

    def price_tooclose(self,direction,price,limit):
        isclose = False
        pricelist = self.get_pricelist(direction)
        distance = price
        for p in pricelist:
            distance = min(distance,abs(p-price))
        if distance<limit:
            isclose = True
        return isclose

class OrderManager(object):
    def __init__(self):
        self.orderdict = {}
        self.position = {}

    def clear_order(self):
        self.orderdict = {}

    def order_add(self,order):
        if order.clientid not in self.orderdict and order.clientid:
            self.orderdict[order.clientid] = order
        else:
            print ('add order failed:',order.clientid)

    def has_order(self,orderid = None,clientid = None):
        ishas = False
        if clientid and clientid in self.orderdict:
            ishas = True
        if orderid:
            orderidlist = [self.orderdict[clientid].orderid for clientid in self.orderdict]
            if orderid in orderidlist:
                ishas = True
        return ishas

    def delete_order(self,clientid):
        if clientid in self.orderdict:
            del self.orderdict[clientid]

    def get_pricelist(self,direction):
        try:
            pricelist = [self.orderdict[clientid].price for clientid in self.orderdict if
                        self.orderdict[clientid].direction == direction and not self.orderdict[clientid].nature]
        except:
            pricelist = None
        return pricelist

    def get_bidaskprice(self):
        bidprice = None
        askprice = None
        buypricelist = self.get_pricelist(Direction.LONG)
        sellpricelist = self.get_pricelist(Direction.SHORT)
        if buypricelist:
            bidprice = max(buypricelist)
        if sellpricelist:
            askprice = min(sellpricelist)

        return [bidprice,askprice]

    def get_L1_clientid(self):
        buypricelist = self.get_pricelist(Direction.LONG)
        sellpricelist = self.get_pricelist(Direction.SHORT)
        try:
            maxbuyprice = max(buypricelist)
            minsellprice = min(sellpricelist)
        except:
            maxbuyprice = None
            minsellprice = None
        buyid = None
        sellid = None
        buyorder = None
        sellorder = None
        if maxbuyprice and minsellprice:
            for clientid in self.orderdict:
                order  = self.orderdict[clientid]
                if order.direction == Direction.LONG and order.price == maxbuyprice:
                    buyid = clientid
                    buyorder = order
                if order.direction == Direction.SHORT and order.price == minsellprice:
                    sellid = clientid
                    sellorder = order
            return buyid,buyorder,sellid,sellorder
        else:
            return None,None,None,None

    def get_ordernumber(self,direction = None):
        if direction is None:
            buylist = self.getorderid_fromdirection(Direction.LONG)
            selllist = self.getorderid_fromdirection(Direction.SHORT)
            return  len(buylist),len(selllist)
        if direction is not None:
            orderlist = self.getorderid_fromdirection(direction)
            return len(orderlist)

    def getorderid_fromdirection(self,direction):
        try:
            orderidlist = [clientid for clientid in self.orderdict if self.orderdict[clientid].direction == direction]
        except:
            orderidlist = []
        return orderidlist

    def get_mindistance(self,price,direction):
        mindis = 0
        pricelist = self.get_pricelist(direction)
        if pricelist:
            dislist = [abs(price - i) for i in pricelist]
            mindis = min(dislist)
        if not pricelist:
            mindis = None
        return mindis

    def order_amend(self,order):
        if order.clientid in self.orderdict:
            self.orderdict[order.clientid].status = order.status
            self.orderdict[order.clientid].filledquantity = order.filledquantity
            self.orderdict[order.clientid].tradevol = order.filledquantity
            self.cal_orderposition(order)

    def cal_orderposition(self,order):
        if order.symbol not in self.position:
            self.position[order.symbol] = {
                                'buyvol':0,
                                'sellvol':0,
                                'buyprice':0,
                                'sellprice':0,
                                }
        if order.direction == Direction.LONG and (order.status ==  Status.ALLTRADED or order.status == Status.PARTCANCELED):
            totalvol = self.position[order.symbol]['buyvol'] + self.orderdict[order.clientid].tradevol
            self.position[order.symbol]['buyprice'] = (self.position[order.symbol]['buyprice']*self.position[order.symbol]['buyvol'] + self.orderdict[order.clientid].tradevol*order.price)/totalvol
            self.position[order.symbol]['buyvol'] = self.orderdict[order.clientid].tradevol
            # print('cal position,buy',self.orderdict[order.clientid].tradevol)

        if order.direction == Direction.SHORT and (order.status ==  Status.ALLTRADED or order.status == Status.PARTCANCELED):
            totalvol = self.position[order.symbol]['sellvol'] + self.orderdict[order.clientid].tradevol
            self.position[order.symbol]['sellprice'] = (self.position[order.symbol]['sellprice']*self.position[order.symbol]['sellvol'] + self.orderdict[order.clientid].tradevol*order.price)/totalvol
            self.position[order.symbol]['sellvol'] = self.orderdict[order.clientid].tradevol
            # print('cal position,sell',self.orderdict[order.clientid].tradevol)

    def order_deletedone(self):
        neworderidct = {}
        for clientid in self.orderdict:
            state = self.orderdict[clientid].status
            if state != Status.ALLTRADED and state != Status.CANCELLED and state != Status.REJECTED:
                neworderidct[clientid] = self.orderdict[clientid]
        self.orderdict = neworderidct

    def get_status(self):
        state = [self.orderdict[clientid].status for clientid in self.orderdict if not self.orderdict[clientid].nature]
        return state

    def get_clientid_levelone(self):
        [bidprice,askprice] = self.get_bidaskprice()
        buylist = []
        selllist = []
        for clientid in self.orderdict:
            if self.orderdict[clientid].price == bidprice:
                buylist.append(clientid)
            if self.orderdict[clientid].price == askprice:
                selllist.append(clientid)
        return buylist,selllist

    def getorder_from_clientid(self,clientid):
        return self.orderdict[clientid]

    def getorder_from_orderid(self,orderid):
        data = None
        for clientid in self.orderdict:
            if self.orderdict[clientid].orderid == orderid:
                data = self.orderdict[clientid]
                break
        return data

    def __iter__(self):
        return iter(self.orderdict.items())

    def create_clientid(self):
        ordernum = self.get_ordernumber()
        clientid = 'LQ_' + str(int(time.time() * 1000)) + '_' + str(sum(ordernum))
        return clientid