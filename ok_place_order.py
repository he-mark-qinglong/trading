# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:23:20 2025

@author: yb
"""

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
# import init_

ABSPATH=os.path.abspath(sys.argv[0])  
ABSPATH=os.path.dirname(ABSPATH)+"/"    
def import_account():
    filepath = ABSPATH+'Account.json'
    f = open(filepath).read()
    accountlist = json.loads(f)
    return accountlist
acclist = import_account()
user='okex_v5_excute'

okexacc = acclist[user]
api_key=okexacc['apikey']
seceret_key=okexacc['secretkey']
passphrase=okexacc['passphrase']
flag='0'

while(1):
    try:
        accountAPI = account.AccountAPI(api_key, seceret_key, passphrase, False, flag)
        break
    except:
        print('\t资产账户登陆超时')
        time.sleep(1)
        continue
time.sleep(0.1 )         
while(1):
    try:
        tradeAPI =trade.TradeAPI(api_key, seceret_key, passphrase, False, flag)
        break
    except:
        print('\t 交易账户登陆超时')
        time.sleep(1)
        continue
time.sleep(0.1 ) 
while(1):
    try:
        fundingAPI = funding.FundingAPI(api_key, seceret_key, passphrase, False, flag)
        break
    except:
        print('\t 资金账户登陆超时')
        time.sleep(1)
        continue
time.sleep(0.1 ) 
while(1):
    try:
        marketAPI = market.MarketAPI(api_key, seceret_key, passphrase, False, flag)
        break
    except:
        print('\t 市场账户登陆超时')
        time.sleep(1)
        continue
time.sleep(0.1)
while(1):
    try:
        publicAPI = public.PublicAPI(api_key, seceret_key, passphrase, False, flag)
        break
    except:
        print('\t 公共账户登陆超时')
        time.sleep(1)
        continue
time.sleep(0.1)    


def cancel_order1():
    # lock.acquire()
    try:
        unfill=tradeAPI.get_order_list(instId='')['data']
        print('订单请求')
    except:
        pass
    # lock.release()
    if len(unfill)>0:
        for i in range(len(unfill)):
            instid=unfill[i]['instId']
            orderid=unfill[i]['ordId']
            print('撤单开始')
            tradeAPI.cancel_order(instid, orderid)
def get_future_ticker(symbol):          
     while(1):
        try:
            time.sleep(0.5)#1
            print(symbol)
            future_ticker = float(marketAPI.get_ticker(symbol)['data'][0]['last'])
            print(future_ticker)
            break
        except Exception as e:
            time.sleep(1)
            print('\t 获取合约现价超时',e,symbol)
            continue
     return future_ticker
def create_order1(symbol,price,amount,model,tag="520ccb3f7df2SUDE"):#bieren 955a52fe8e73SUDE,model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
     if model=='buylong':
         place_order=tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','',tag,'long',str(price),'')
     elif model=='buyshort':
         place_order=tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','',tag,'short',str(price),'')
     elif model=='selllong':  
         place_order=tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','',tag,'long',str(price),'')
     elif model=='sellshort':  
         place_order=tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','',tag,'short',str(price),'')
     elif model=='buycash':
         place_order=tradeAPI.place_order(symbol,'cross','buy','limit',str(amount),'','',tag,'',str(price),'')
     elif model=='sellcash':
         place_order=tradeAPI.place_order(symbol,'cross','sell','limit',str(amount),'','',tag,'',str(price),'')
     return place_order
cancel_order1()
amount=100#177
symbol='BTC-USDT-SWAP'
model='buyshort'
# price=get_future_ticker(symbol)*0.997
while 1:
    try:
        cancel_order1()
        time.sleep(2)
        print('等待成交中'*10)
        price0=get_future_ticker(symbol)
        print('symbol',symbol,'price0',price0,'amount',amount,'model',model)
        price=price0*0.9998
        if price>8705.3 or price<87821.3:
            create_order1(symbol,price,amount,model)
            amount += 1
            time.sleep(1)
            break
    except:
        time.sleep(2)
        pass
    