# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:22:08 2024

@author: yb
"""

# -- coding: UTF-8 --
import requests
import json
import pickle
import datetime 
import time
import pandas as pd
import numpy as np
import hashlib
from urllib.parse import urlencode
from decimal import Decimal
#from OkcoinSpotAPI import OKCoinSpot
#from OkcoinFutureAPI import OKCoinFuture
#import json
import YQ.API.okex_v5.Account_api as account
import YQ.API.okex_v5.Funding_api as funding
import YQ.API.okex_v5.Market_api as market
import YQ.API.okex_v5.Public_api as public
import YQ.API.okex_v5.Trade_api as trade 
import json
from pytz import timezone #timezone('Asia/Shanghai') #东八区
import warnings
import os
import sys
import gc
import pprint
import threading
import random
lock = threading.Lock()
import logging
from websocket import create_connection
import hmac
import base64
from threading import Lock
from collections import deque
from BinanceFutures import  BinanceFut
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)
#import bs4 #pip install --upgrade beautifulsoup4 
#bs4.__version__
#import html5lib #pip install -U html5lib=="0.9999999
#html5lib.__version__
#from urllib.request import  urlopen
#ex = ccxt.binance()
flag='0'
user='j02'
ABSPATH=os.path.abspath(sys.argv[0])  
ABSPATH=os.path.dirname(ABSPATH)+"/"    
Asset0=2000
follow_ex='okx'#okx,binance
class get_depth(object):
    def __init__(self,login_info,user,ex,asset0):
         self.shut=1
         self.user=user
         self.ex=ex
         self.lev=1#跟单杠杆
         self.delta_time=time.time()
         self.asset0=asset0
         self.jt=0
         acclist = self.import_account()
        
         if follow_ex=='okx':
             okexacc = acclist['okex_v5_hql']
             self.api_key=okexacc['apikey']
             self.seceret_key=okexacc['secretkey']
             self.passphrase=okexacc['passphrase']
             # self.root=okexacc['root']
             # self.root1=okexacc['root1']
             self.login()
             # if 'BTC' in symbol_ex:
             #      product=self.get_product_swap()
             #      self.symbol_main=product['instrument_id_SWAP_USDT_BTC']
             # elif  'ETH' in symbol_ex:
             #       product=self.get_product_swap()
             #       self.symbol_main=product['instrument_id_SWAP_USDT_ETH']
         elif  follow_ex=='binance':
             binanceacc = acclist['binance_yyyyy1']
             self.api_key=binanceacc['apikey']
             self.seceret_key=binanceacc['secretkey']
             # self.root=binanceacc['root']
             # self.root1=okexacc['root1']
             self.ex_binance_main=BinanceFut(self.api_key,self.seceret_key)
             # if 'BTC' in symbol_ex:
             #     self.symbol_main='BTCUSDT'
             # if 'ETH' in symbol_ex:
             #     self.symbol_main='ETHUSDT'
             
         if ex=='okx':
             self.api_key=login_info['apikey']
             self.seceret_key=login_info['secretkey']
             self.passphrase=login_info['passphrase']
             self.login1()
             
             try:
                 self.accountAPI1.get_position_mode('long_short_mode')
                 account_config=self.accountAPI1.get_account_config()
                 self.autoLoan=account_config['data'][0]['autoLoan']#true为自动借币
                 self.margin_model=account_config['data'][0]['acctLv']#'3'3为跨币种保证金模式
                 symbol_list=self.get_product_swap1()
                 # print(symbol_list)
                 # for i in symbol_list:
                 #       print(i)
                 #       self.accountAPI1.set_leverage('3', 'cross', i,'','long')
                 #       self.accountAPI1.set_leverage('3', 'cross', i,'','short')
                 #       time.sleep(1)
                
             except Exception as e:
                 print('设置出错',e)
                 time.sleep(5)
                 pass
             # product=self.get_product_swap1()
             # symbol_list=self.get_positions_okx_all()
             # print('symbol_list'*100,symbol_list)
             # fv=self.get_face_value('BTC-USDT-SWAP')
             # print('fv'*100,fv)
             # return
             # symbol_ex=product[symbol_ex]
         elif ex=='binance':
              self.api_key=login_info['apikey']
              self.seceret_key=login_info['secretkey']
              self.ex_binance=BinanceFut(self.api_key,self.seceret_key)
              precision=self.ex_binance.get_u_ExchangeInfo().get('symbols')
              self.precision_dict={}
              for i in range(len(precision)):
                  symbol=precision[i].get('symbol')
                  # if 'ETH' in symbol:
                  #      print(precision[i],symbol,'006'*100)
                  #      return
                  self.precision_dict[symbol]={}
                  amountPrecision=int(precision[i].get('quantityPrecision'))
                  # pricePrecision=int(precision[i].get('pricePrecision'))
                  filters=precision[i].get('filters')
                  for j in filters:
                    if j['filterType']=='PRICE_FILTER':
                        tickSize=j['tickSize'].rstrip('0')
                        try:
                            print(tickSize,111,str(tickSize),str(tickSize).split('.')[1])
                        except:
                            print('tickSize',tickSize,symbol)
                        print('tickSize',tickSize)
                        if float(tickSize)<1:
                            pricePrecision=len(tickSize.split('.')[1])
                        else:
                            pricePrecision=0
                       
                        break
                  self.precision_dict[symbol]['amountPrecision']=amountPrecision  
                  self.precision_dict[symbol]['pricePrecision']=pricePrecision
               
         # usdt_bal=self.fundingAPI1.get_balances('USDT')['data'][0]['availBal']
         # print(usdt_bal,33333)
         # if float(usdt_bal)>0:
         #     self.fundingAPI1.funds_transfer('USDT',usdt_bal,'6','18')
         # if ex=='okx':
         #     try:
         #          self.accountAPI1.get_position_mode('long_short_mode')
         #          account_config=self.accountAPI1.get_account_config()
         #          self.autoLoan=account_config['data'][0]['autoLoan']#true为自动借币
         #          self.margin_model=account_config['data'][0]['acctLv']#'3'3为跨币种保证金模式
         #          self.accountAPI1.set_leverage('3', 'cross', self.symbol_ex,'','long')
         #          self.accountAPI1.set_leverage('3', 'cross', self.symbol_ex,'','short')
                  
         #     except Exception as e:
         #          print('设置出错',e)
         #          time.sleep(5)
         #          pass
    
    def connect(self):
        # self.get_future_ticker0.start()
        
        self.trade0.start()
    def _trade1(self):
         while True:
            try:
                self.trade1()
            except Exception as e:
                time.sleep(5)
                print(e,'trade1_erro',self.symbol_type)
                pass
    def trade1(self):
        # try:
             # self.setsystem_time()
             # future_ticker=(bidsdict_swap[0]['price']+asksdict_swap[0]['price'])*0.5
             # spot_ticker=(bidsdict_swap_inv[0]['price']+asksdict_swap_inv[0]['price'])*0.5
             sleep_time = random.randint(1, 2)
             time.sleep(sleep_time)
             print('检测中',self.user,self.ex,follow_ex)
             # adjEq,imr,mmr,mgnRatio,notionalUsd,totalEq,lever=self.get_account()
             if follow_ex=='okx':
                 usdt_total=self.get_usdt_total()
             elif  follow_ex=='binance':
                 usdt_total=self.get_usdt_binance_main()
             print(usdt_total)
             if self.ex=='okx':
                 usdt_total1=self.get_usdt_total1()
             elif self.ex=='binance':
                 usdt_total1=self.get_usdt_binance()
             print(usdt_total1)
             usdt_total10=usdt_total1
             
             if  self.user== 'qh':
                 rate=usdt_total1*1/usdt_total # 0.2
             elif self.user== '1106':
                 rate=usdt_total1*1/usdt_total # 0.5
            
             else:
                 rate=usdt_total1/usdt_total
             print('rate',rate,'usdt_total',usdt_total,'usdt_total1',usdt_total1)
             
             if follow_ex=='okx':
                 symbol_list_main,positionAmount_dict=self.get_positions_okx_main()
             elif  follow_ex=='binance':
                 try:
                     symbol_list_main,positionAmount_dict=self.get_positions_binance_main()
                 except Exception as e:
                     logging.info((self.user,'获取binance跟单者仓位失败',e))
             # print('symbol_list_main','positionAmount_dict',symbol_list_main,positionAmount_dict)
             # symbol_list_main0=[i for i in symbol_list_main if i in ['BTCUSDT','ETHUSDT']]
             # if len(symbol_list_main0)>0:
             #     entryprice_dict, markprice_dict , positionAmount_dict, direct_dict=self.get_entryprice_binance_main()
             #     for i in symbol_list_main0:
             #         symbol=i
             #         entryprice=entryprice_dict[symbol]
             #         markprice=markprice_dict[symbol]
             #         amount=abs(positionAmount_dict[symbol])
             #         direct= direct_dict[symbol]
             #         if direct=='SHORT' and (markprice/entryprice>1.03 or markprice/entryprice<0.992):
             #             price=markprice*1.01
             #             price = round(price, self.precision_dict[symbol]['pricePrecision'])
             #             self.ex_binance_main.create_futures_order(symbol,'BUY',price,amount,'SHORT')
             #         if direct=='LONG' and (markprice/entryprice<0.97 or markprice/entryprice>1.008):
             #             price=markprice*0.99
             #             price = round(price, self.precision_dict[symbol]['pricePrecision'])
             #             self.ex_binance_main.create_futures_order(symbol,'SELL',price,amount,'LONG')
                     
                     
             if self.ex=='okx':
                 symbol_list_sub=self.get_positions_okx_all()
             elif self.ex=='binance':
                 symbol_list_sub=self.get_positions_binance_all()
             if  time.time()-self.delta_time>60*2:
                 self.delta_time=time.time()
                 for i in symbol_list_sub:
                     try:
                         self.cancel_all_binance(i)
                     except:
                         pass
                     time.sleep(0.5)
          
             if self.user=='jan' :
                symbol_list_main=[i for i in symbol_list_main if i in ['BTCUSDT','ETHUSDT']]
                symbol_list_sub=[i for i in symbol_list_sub if i in ['BTCUSDT','ETHUSDT']]
             if self.user=='qh' :
               symbol_list_main=[i for i in symbol_list_main if i in ['BTCUSDT','ETHUSDT']]
               symbol_list_sub=[i for i in symbol_list_sub if i in ['BTCUSDT','ETHUSDT']]
             if self.user=='1106' :
                symbol_list_main=[i for i in symbol_list_main if i in ['BTCUSDT','ETHUSDT']]
                symbol_list_sub=[i for i in symbol_list_sub if i in ['BTCUSDT','ETHUSDT']]
             symbol_list_sub_only=list(set(symbol_list_sub)-set(symbol_list_main))
             # if len(symbol_list_sub_only)>0:
             #     symbol_list=symbol_list_sub_only
             # elif len(symbol_list_sub_only)==0:
             #     symbol_list=symbol_list_main
             if self.jt==0:
                self.jt=1
                symbol_list=symbol_list_sub
             elif  self.jt==1:
                self.jt=0
                symbol_list=symbol_list_main
             print('symbol_list',symbol_list,'symbol_list_main',symbol_list_main)
             # print('symbol_list'*100,symbol_list)
             if len(symbol_list)>0:
                 for i in range(len(symbol_list)):
                     symbol_ex=symbol_list[i]
                     if len(symbol_list_sub_only)>0:
                         buy_amount=0
                         sell_amount=0
                     else:
                         if positionAmount_dict[symbol_ex]>0:
                            buy_amount= positionAmount_dict[symbol_ex]
                            sell_amount=0
                         elif positionAmount_dict[symbol_ex]<0:
                            sell_amount= abs(positionAmount_dict[symbol_ex])
                            buy_amount=0
                     
                     if self.ex=='okx':
                         buy_amount1,sell_amount1=self.get_positions_okx(symbol_ex)
                         print(self.ex,symbol_ex,'主账户持仓','资金为',usdt_total,'多单为',buy_amount,'空单为',sell_amount)
                         delta_buy_amount=int(buy_amount*rate-buy_amount1)
                         delta_sell_amount=int(sell_amount*rate-sell_amount1)
                         print(self.ex,symbol_ex,'%s跟单账户持仓'%self.user,'资金为',usdt_total1/self.lev,'多单为',buy_amount1,'空单为',sell_amount1,'持仓多单差额为',delta_buy_amount,'持仓空单差额为',delta_sell_amount)
                     elif self.ex=='binance':
                         buy_amount1,sell_amount1=self.get_positions_binance(symbol_ex)  
                         print(self.ex,symbol_ex,'主账户持仓','资金为',usdt_total,'多单为',buy_amount,'空单为',sell_amount)
                         delta_buy_amount=round(buy_amount*rate-buy_amount1,2)
                         delta_sell_amount=round(sell_amount*rate-sell_amount1,2)
                         print(self.ex,symbol_ex,'%s跟单账户持仓'%self.user,'资金为',usdt_total1/self.lev,'多单为',buy_amount1,'空单为',sell_amount1,'持仓多单差额为',delta_buy_amount,'持仓空单差额为',delta_sell_amount)
                     if delta_buy_amount>0 and self.shut==0 :
                         if self.ex=='okx':
                             self.cancel_order1()
                             amount=max(delta_buy_amount,1)
                             symbol=symbol_ex
                             price=self.get_future_ticker(symbol)*1.01
                             model='buylong'
                             if amount>buy_amount1*0.08 and  amount>0:
                                 self.create_order1(symbol,price,amount,model)
                         elif  self.ex=='binance' and delta_buy_amount>=0.01:
                             symbol=symbol_ex
                             amount=round(delta_buy_amount,self.precision_dict[symbol]['amountPrecision'])
                             price = float(self.ex_binance.get_future_tickers(symbol).get('lastPrice'))
                             price = round(price*1.002, self.precision_dict[symbol]['pricePrecision'])
                             if amount>buy_amount1*0.08 and  amount>0:
                                self.ex_binance.create_futures_order(symbol,'BUY',price,amount,'LONG')
                     if  delta_sell_amount>0 and self.shut==0 :
                         if self.ex=='okx':
                             self.cancel_order1()
                             amount=max(delta_sell_amount,1)
                             symbol=symbol_ex
                             price=self.get_future_ticker(symbol)*0.99
                             model='buyshort'
                             if amount>sell_amount1*0.08 and  amount>0:
                                 self.create_order1(symbol,price,amount,model)
                         elif  self.ex=='binance'  and delta_sell_amount>=0.01:
                             symbol=symbol_ex
                             amount=round(delta_sell_amount,self.precision_dict[symbol]['amountPrecision'])
                             price = float(self.ex_binance.get_future_tickers(symbol).get('lastPrice'))
                             price = round(price*0.998, self.precision_dict[symbol]['pricePrecision'])
                             print(symbol,price,amount,self.precision_dict[symbol]['pricePrecision'],'002'*100)
                             if amount>sell_amount1*0.08 and  amount>0:
                                 self.ex_binance.create_futures_order(symbol,'SELL',price,amount,'SHORT')
                     if delta_buy_amount<0 :
                         if self.ex=='okx':
                             self.cancel_order1()
                             amount=-delta_buy_amount
                             symbol=symbol_ex
                             price=self.get_future_ticker(symbol)*0.99
                             model='selllong'
                             if amount>buy_amount1*0.08 and  amount>0:
                                 self.create_order1(symbol,price,amount,model)
                         elif  self.ex=='binance'  and delta_buy_amount<=-0.01:
                             symbol=symbol_ex
                             amount=round(-delta_buy_amount,self.precision_dict[symbol]['amountPrecision'])
                             price = float(self.ex_binance.get_future_tickers(symbol).get('lastPrice'))
                             price = round(price*0.998, self.precision_dict[symbol]['pricePrecision'])
                             print(symbol,price,amount,self.precision_dict[symbol]['pricePrecision'],self.precision_dict[symbol]['amountPrecision'],'003'*100)
                             if amount>buy_amount1*0.08 and  amount>0: 
                                 self.ex_binance.create_futures_order(symbol,'SELL',price,amount,'LONG')
                     if delta_sell_amount<0:
                         if self.ex=='okx':
                             self.cancel_order1()
                             amount=-delta_sell_amount
                             symbol=symbol_ex
                             price=self.get_future_ticker(symbol)*1.01
                             model='sellshort'
                             if amount>sell_amount1*0.08 and  amount>0:
                                 self.create_order1(symbol,price,amount,model)
                         elif  self.ex=='binance'  and delta_sell_amount<=-0.01: 
                             symbol=symbol_ex
                             amount=round(-delta_sell_amount,self.precision_dict[symbol]['amountPrecision'])
                             price = float(self.ex_binance.get_future_tickers(symbol).get('lastPrice'))
                             price = round(price*1.002, self.precision_dict[symbol]['pricePrecision'])
                             if amount>sell_amount1*0.08 and  amount>0:
                                 self.ex_binance.create_futures_order(symbol,'BUY',price,amount,'SHORT')
                     if usdt_total10/self.asset0<0.95:
                         self.shut=1
                         print('self.user',self.user,'self.asset0',self.asset0,'usdt_total1',usdt_total1,'回撤超过10%暂停开平仓')
                     else:
                         self.shut=0
    def get_product_swap(self):
        a= self.publicAPI.get_instruments('SWAP')['data']
        time.sleep(2)
        for i in range(len(a)):
            type1=a[i].get('instId')
            # print(type1)
            locals()['instrument_id_'+a[i].get('instType')+'_'+a[i].get('settleCcy')+'_'+a[i].get('ctValCcy')]=a[i].get('instId')
        return locals()
    def get_product_swap1(self):
        a= self.publicAPI1.get_instruments('SWAP')['data']
        time.sleep(2)
        for i in range(len(a)):
            type1=a[i].get('instId')
            # print(type1)
            locals()['instrument_id_'+a[i].get('instType')+'_'+a[i].get('settleCcy')+'_'+a[i].get('ctValCcy')]=a[i].get('instId')
        return locals()
    def get_future_ticker(self,symbol):          
         while(1):
            try:
                time.sleep(0.5)#1
                print(symbol)
                future_ticker = float(self.marketAPI1.get_ticker(symbol)['data'][0]['last'])
                print(future_ticker)
                break
            except Exception as e:
                time.sleep(1)
                print('\t 获取合约现价超时',e,symbol)
                continue
         return future_ticker
    def   get_positions_okx_main(self):     
         while(1):
            try:
                positionAmount_dict={}
                symbol_list=[]
                position_amount = self.accountAPI.get_positions('')
                # print('position_amount',position_amount)
                if len(position_amount.get('data'))>0:
                    for i in range(len(position_amount.get('data'))):
                        mgnMode=position_amount.get('data')[i].get('mgnMode')
                        if position_amount.get('data')[i].get('posSide')=='long' and mgnMode=='cross':
                             positionAmount =float(position_amount.get('data')[i].get('pos'))
                        elif position_amount.get('data')[i].get('posSide')=='short' and mgnMode=='cross':
                             positionAmount =-float(position_amount.get('data')[i].get('pos'))
                        if self.ex=='okx':
                            symbol=position_amount.get('data')[i].get('instId')
                            positionAmount_dict[symbol]=positionAmount
                            symbol_list.append(symbol)
                        elif self.ex=='binance':
                            symbol=position_amount.get('data')[i].get('instId')
                            fv=self.get_face_value(symbol)
                            positionAmount=positionAmount*fv
                            symbol=symbol.split('-')[0]+symbol.split('-')[1]
                            positionAmount_dict[symbol]=positionAmount
                            symbol_list.append(symbol)
                    # print('\t 实际持有合约空单张数为%s'%sell_amount)#%f %s
                    # print('\t 实际持有合约多单张数为%s'%buy_amount)#%f %s
                else:
                    pass
                break
            except Exception as e:
                time.sleep(5)
                print('获取okx主账户合约持仓失败'*100,e)
                logging.info(('获取okx主账户合约持仓失败'))
                continue    
         return symbol_list,positionAmount_dict
    
    def  get_positions_okx(self,symbol):     
         while(1):
            try:
                time.sleep(0.5)
                position_amount = self.accountAPI1.get_positions('')
                buy_amount=0
                sell_amount=0
                # print(position_amount,symbol_ex)
                if len(position_amount.get('data'))>0:
                    for i in range(len(position_amount.get('data'))):
                        mgnMode=position_amount.get('data')[i].get('mgnMode')
                        instId=position_amount.get('data')[i].get('instId')
                        if instId==symbol and position_amount.get('data')[i].get('posSide')=='long' and mgnMode=='cross':
                             buy_amount= float(position_amount.get('data')[i].get('pos'))
                        elif instId==symbol and  position_amount.get('data')[i].get('posSide')=='short' and mgnMode=='cross':
                             sell_amount= abs(float(position_amount.get('data')[i].get('pos')))
                        if buy_amount>0 and sell_amount>0:
                            break
                    # print('\t 实际持有合约空单张数为%s'%sell_amount)#%f %s
                    # print('\t 实际持有合约多单张数为%s'%buy_amount)#%f %s
                else:
                    pass
#                        sell_amount_inv = logic_sell_future
#                        print('\t实际持有空单张数为%s'% sell_amount)
                break
            except:
                time.sleep(5)
                print('获取okx跟单账户合约实际持仓失败'*100)
                logging.info(('获取okx跟单账户合约实际持仓失败'))
                continue    
         return buy_amount,sell_amount
    def  get_positions_binance(self,symbol):   
        a11 = 1
        while(1):
            try:
                position_amount= self.ex_binance.get_position_risk(symbol)
                for i in range(len(position_amount)):
                    if position_amount[i].get('symbol')==symbol and position_amount[i].get('positionSide')=='LONG':
                        buy_amount =float(position_amount[i].get('positionAmt'))
                        print(position_amount[i],77777,'buy_amount',buy_amount)
                    elif position_amount[i].get('symbol')==symbol and position_amount[i].get('positionSide')=='SHORT':
                        sell_amount =abs(float(position_amount[i].get('positionAmt')))
                        print(position_amount[i],88888,'sell_amount',sell_amount)
                break
            except:
                a11+=1
                print('\t 获取持仓数量超时')
                print('\t a11=%s'%a11)
                time.sleep(1)
                if a11<5:
                    continue
                else:
                    buy_amount = 0
                    sell_amount = 0
                    break
        return buy_amount,sell_amount
    def  get_positions_okx_all(self,symbol='BTCUSDT'):   
        a11 = 1
        while(1):
            try:
                position_amount = self.accountAPI1.get_positions('')
                symbol_list=[]
                if len(position_amount)>0:
                    position_amount=position_amount['data']
                    for i in range(len(position_amount)):
                        if abs(float(position_amount[i].get('pos')))>0:
                            symbol=position_amount[i].get('instId')
                            symbol_list.append(symbol)
                break
            except:
                a11+=1
                print('\t 获取持仓数量超时')
                print('\t a11=%s'%a11)
                time.sleep(1)
                if a11<5:
                    continue
                else:
                    break
        return  symbol_list
    def  get_positions_binance_all(self,symbol='BTCUSDT'):   
        a11 = 1
        while(1):
            try:
                position_amount= self.ex_binance.get_position_risk(symbol)
                symbol_list=[]
                for i in range(len(position_amount)):
                    if abs(float(position_amount[i].get('positionAmt')))>0:
                        symbol=position_amount[i].get('symbol')
                        symbol_list.append(symbol)
                break
            except:
                a11+=1
                print('\t 获取持仓数量超时')
                print('\t a11=%s'%a11)
                time.sleep(1)
                if a11<5:
                    continue
                else:
                    break
        return  symbol_list
    def  get_entryprice_binance_main(self,symbol='BTCUSDT'):
         a11 = 1
         while(1):
             try:
                 positionAmount_dict={}
                 entryprice_dict={}
                 markprice_dict={}
                 direct_dict={}
                 position_amount= self.ex_binance_main.get_position_risk(symbol)
                 if len(position_amount)>0 :
                     for i in range(len(position_amount)):
                         if abs(float(position_amount[i].get('positionAmt')))>0:
                            positionAmount =float(position_amount[i].get('positionAmt'))
                            entryprice =float(position_amount[i].get('entryPrice'))
                            markprice =float(position_amount[i].get('markPrice'))
                            symbol=position_amount[i].get('symbol')
                            direct=position_amount[i].get('positionSide')
                            positionAmount_dict[symbol]=positionAmount
                            entryprice_dict[symbol]=entryprice
                            markprice_dict[symbol]=markprice
                            direct_dict[symbol]=direct
                 break
             except Exception as e:
                 a11+=1
                 print('\t 获取币安主账户持仓数量超时',e)
                 print('\t a11=%s'%a11)
                 time.sleep(1)
                 if a11<5:
                     continue
                 else:
                     break
         return entryprice_dict, markprice_dict , positionAmount_dict, direct_dict
    def  get_positions_binance_main(self,symbol='BTCUSDT'):   
        a11 = 1
        while(1):
            try:
                positionAmount_dict={}
                symbol_list=[]
                position_amount= self.ex_binance_main.get_position_risk(symbol)
               
                if len(position_amount)>0 :
                    for i in range(len(position_amount)):
                        if abs(float(position_amount[i].get('positionAmt')))>0:
                           
                            if self.ex=='binance':
                                positionAmount =float(position_amount[i].get('positionAmt'))
                                symbol=position_amount[i].get('symbol')
                                positionAmount_dict[symbol]=positionAmount
                                symbol_list.append(symbol)
                            elif self.ex=='okx':
                                positionAmount =float(position_amount[i].get('positionAmt'))
                                symbol=position_amount[i].get('symbol')
                                symbol=symbol.split('USDT')[0]+'-USDT-SWAP'
                                try:
                                    fv=self.get_face_value(symbol)
                                    if fv==0:
                                        continue
                                    # assert fv!=0
                                except Exception as e:
                                    print('面值报错',e)
                                    continue
                                positionAmount=int(positionAmount/fv)
                                positionAmount_dict[symbol]=positionAmount
                                symbol_list.append(symbol)
                                
                             
                            
                break
            except Exception as e:
                a11+=1
                print('\t 获取币安主账户持仓数量超时',e)
                print('\t a11=%s'%a11)
                time.sleep(1)
                if a11<5:
                    continue
                else:
                    break
        return symbol_list,positionAmount_dict
    def cancel_order(self):
        # lock.acquire()
        try:
            unfill=self.tradeAPI.get_order_list(instId='')['data']
            print('订单请求')
        except:
            pass
        # lock.release()
        if len(unfill)>0:
            for i in range(len(unfill)):
                instid=unfill[i]['instId']
                orderid=unfill[i]['ordId']
                print('撤单开始')
                self.tradeAPI.cancel_order(instid, orderid)
        
    
    def cancel_order1(self):
        # lock.acquire()
        try:
            unfill=self.tradeAPI1.get_order_list(instId='')['data']
            print('订单请求')
        except:
            pass
        # lock.release()
        if len(unfill)>0:
            for i in range(len(unfill)):
                instid=unfill[i]['instId']
                orderid=unfill[i]['ordId']
                print('撤单开始')
                self.tradeAPI1.cancel_order(instid, orderid)
    
    def get_position_risk(self): 
        a=self.accountAPI.get_position_risk('')
        print(a)   
    
    
    
    def login(self):
        while(1):
            try:
                self.accountAPI = account.AccountAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t资产账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 )         
        while(1):
            try:
                self.tradeAPI =trade.TradeAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 交易账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 ) 
        while(1):
            try:
                self.fundingAPI = funding.FundingAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 资金账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 ) 
        while(1):
            try:
                self.marketAPI = market.MarketAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 市场账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1)
        while(1):
            try:
                self.publicAPI = public.PublicAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 公共账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1)    
    def login1(self):
        while(1):
            try:
                self.accountAPI1 = account.AccountAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t资产账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 )         
        while(1):
            try:
                self.tradeAPI1 =trade.TradeAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 交易账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 ) 
        while(1):
            try:
                self.fundingAPI1 = funding.FundingAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 资金账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1 ) 
        while(1):
            try:
                self.marketAPI1 = market.MarketAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 市场账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1)
        while(1):
            try:
                self.publicAPI1 = public.PublicAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                break
            except:
                print('\t 公共账户登陆超时')
                time.sleep(1)
                continue
        time.sleep(0.1)
        
        
        
        
    def setsystem_time(self):
         timeNum=self.publicAPI.get_system_time()
         timeNum=timeNum['data'][0]['ts']
         timeStamp = float(timeNum)/1000
         ttime = time.localtime(timeStamp)
#         ttime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
         dat = "date %u-%02u-%02u" % (ttime.tm_year, ttime.tm_mon, ttime.tm_mday)
         tm = "time %02u:%02u:%02u" % (ttime.tm_hour, ttime.tm_min, ttime.tm_sec)
         os.system(dat)
         os.system(tm)
    
    def get_usdt_total(self):
         while(1):
            try:
                lock.acquire()
                try:
                    account_info = self.accountAPI.get_account()
                except:
                    lock.release()
                    pass
                lock.release()
                totalEq=float(account_info['data'][0]['totalEq'])
                
                break
            except:
                print('\t 获取总资金失败')
                self.setsystem_time()
                time.sleep(1)
                continue
         return totalEq 
    def get_usdt_total1(self):
         while(1):
            try:
                lock.acquire()
                try:
                    account_info = self.accountAPI1.get_account()
                except:
                    lock.release()
                    pass
                lock.release()
                totalEq=float(account_info['data'][0]['totalEq'])
                
                break
            except:
                print('\t 获取总资金失败')
                self.setsystem_time()
                time.sleep(1)
                continue
         return totalEq 
    def get_usdt_binance(self):
        a11 = 1
        while(1):
            try:
                time.sleep(1)
                capital1 = self.ex_binance.get_futures_balances('usdt').get('usdt_base_balance')
                for i in range(len(capital1)):
                    try:
                        if capital1[i].get('asset')=='USDT':
                            rights_usdt = float(capital1[i].get('balance')) #账户权益 etc数量
                            break
                    except:
                        rights_usdt = 0
                right_future = rights_usdt
                break
            except Exception as e:
                
                a11+=1
                print('\t 获取合约用户信息超时',e)
                print('\t a11=%s'%a11)
                time.sleep(3)
                if a11<3:
                    continue
                else:
                    break
        return rights_usdt  
    def get_usdt_binance_main(self):
        a11 = 1
        while(1):
            try:
                time.sleep(1)
                capital1 = self.ex_binance_main.get_futures_balances('usdt').get('usdt_base_balance')
                for i in range(len(capital1)):
                    try:
                        if capital1[i].get('asset')=='USDT':
                            rights_usdt = float(capital1[i].get('balance')) #账户权益 etc数量
                            break
                    except:
                        rights_usdt = 0
                right_future = rights_usdt
                break
            except Exception as e:
                
                a11+=1
                print('\t 获取合约用户信息超时',e)
                print('\t a11=%s'%a11)
                time.sleep(3)
                if a11<3:
                    continue
                else:
                    break
        return rights_usdt
    def get_face_value(self,symbol):
        try:
            face=self.publicAPI.get_instruments('SWAP')
        except:
            face=self.publicAPI1.get_instruments('SWAP')
        data=face['data']
        for i in range(len(data)):
            if data[i]['instId']==symbol:
                fv=float(data[i]['ctVal'])
                break
            else:
                fv=0
#        print(11111,fv,symbol,data[i]['instId'])
        return fv
    def get_face_value_f(self,symbol):
        face=self.publicAPI.get_instruments('FUTURES')
        data=face['data']
        for i in range(len(data)):
            if data[i]['instId']==symbol:
                fv=float(data[i]['ctVal'])
                break
            else:
                fv=0
#        print(11111,fv,symbol,data[i]['instId'])
        return fv
    def get_last_time(self):
        try:
            lock.acquire()
            self.last_time=time.time()
            print(self.last_time,99999)
        finally:
            lock.release()
    
    def create_order(self,symbol,price,amount,model):#model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
        if model=='buylong':
            place_order=self.tradeAPI.place_order(symbol,'isolated','buy','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='buyshort':
            place_order=self.tradeAPI.place_order(symbol,'isolated','sell','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='selllong':  
            place_order=self.tradeAPI.place_order(symbol,'isolated','sell','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='sellshort':  
            place_order=self.tradeAPI.place_order(symbol,'isolated','buy','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='buycash':
            place_order=self.tradeAPI.place_order(symbol,'isolated','buy','limit',str(amount),'','','','',str(price),'')
        elif model=='sellcash':
            place_order=self.tradeAPI.place_order(symbol,'isolated','sell','limit',str(amount),'','','','',str(price),'')
        return place_order
    def create_order1(self,symbol,price,amount,model):#model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
        if model=='buylong':
            place_order=self.tradeAPI1.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='buyshort':
            place_order=self.tradeAPI1.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='selllong':  
            place_order=self.tradeAPI1.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='sellshort':  
            place_order=self.tradeAPI1.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='buycash':
            place_order=self.tradeAPI1.place_order(symbol,'cross','buy','limit',str(amount),'','','','',str(price),'')
        elif model=='sellcash':
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(amount),'','','','',str(price),'')
        return place_order
    def create_order_binance(self,symbol,price,amount,model):#model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
        if model=='buylong':
            place_order=self.tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='buyshort':
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='selllong':  
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','','','long',str(price),'')
        elif model=='sellshort':  
            place_order=self.tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','','','short',str(price),'')
        elif model=='buycash':
            place_order=self.tradeAPI.place_order(symbol,'cash','buy','limit',str(amount),'','','','',str(price),'')
        elif model=='sellcash':
            place_order=self.tradeAPI.place_order(symbol,'cash','sell','limit',str(amount),'','','','',str(price),'')
        return place_order
    def sign_request(self,method, path, params):
        # if "apiKey" not in options or "secret" not in options:
        #     raise ValueError("Api key and secret must be set")
        if  'fapi' in path :
            base_path='https://fapi.binance.com'
        # elif 'papi' in path and 'cm' in path:
        #     base_path=self.c_future_path
        # elif  'papi' in path and 'um' in path:
        #     base_path=self.u_future_path
        # elif  'sapi' in path:
        #     base_path=self.spots_path
        query = urlencode(sorted(params.items()))
        query += "&timestamp={}".format(int(time.time() * 1000))
        secret = bytes(self.seceret_key.encode("utf-8"))
        signature = hmac.new(secret, query.encode("utf-8"),
                              hashlib.sha256).hexdigest()
        query += "&signature={}".format(signature)
        resp = requests.request(method,
                                base_path + path + "?" + query,
                                headers={"X-MBX-APIKEY": self.api_key})
        data = resp.json()
        return data
    def request(self,method, path, params=None):
        if 'fapi' in path :
            base_path='https://fapi.binance.com'
        elif  'dapi' in path:
            base_path='https://dapi.binance.com'
        elif  'fapi' not in path and 'dapi' not in path and 'api' in path:
            base_path='https://api.binance.com'
        resp = requests.request(method, base_path + path, params=params)
        data = resp.json()
        return data
    def cancel_all_binance(self,symbol,**kwargs):
        params = {
            "symbol":symbol
            }
        params.update(kwargs)
        data = self.sign_request('DELETE', '/fapi/v1/allOpenOrders', params)
        return data
    # 查询账户最大可借贷额度 (USER_DATA)
    def get_maxBorrowable(self,asset,**kwargs):
        params = {
            "asset":asset
            }
        params.update(kwargs)
        data = self.sign_request('GET', '/papi/v1/margin/maxBorrowable', params)
        return data
    def import_account(self):
        filepath = ABSPATH+'Account.json'
        f = open(filepath).read()
        accountlist = json.loads(f)
        return accountlist
    
    
    

if __name__=='__main__':

    filepath = ABSPATH+'Account.json'
    f = open(filepath).read()
    accountlist = json.loads(f)
    aa1=get_depth(accountlist['okex_v5_yyyyy1'],'bruce','okx',600*0.8)
    while(1):
        try:
            time1=datetime.datetime.now()
            time11=time1.hour*100+time1.minute
            week_day=time1.isoweekday()
            threads=[]
            t1 = threading.Thread(target = aa1.trade1) 
            threads.append(t1)
            i=0
            for t in threads:
                i+=1
                t.start()
                if i%3==0:
                    time.sleep(1)
            for t in threads:
                t.join()
        except:
            time.sleep(5)
            print('启动报错')
    
    