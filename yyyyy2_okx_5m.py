
# -- coding: UTF-8 --
import pickle
import datetime 
import time
import pandas as pd
import numpy as np
import random
import scipy.ndimage.interpolation as ns
import YQ.API.okex_v5.Account_api as account
import YQ.API.okex_v5.Funding_api as funding
import YQ.API.okex_v5.Market_api as market
import YQ.API.okex_v5.Public_api as public
import YQ.API.okex_v5.Trade_api as trade 
import json
from pytz import timezone #timezone('Asia/Shanghai') #东八区
import warnings
import os
import gc
import sys
import pprint
import threading
lock = threading.Lock()
# from threading import Lock
import logging
import traceback# traceback.print_exc()
from collections import deque
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)

root = '/Users/a1234/Desktop/trade_excute'
ABSPATH=os.path.abspath(sys.argv[0])  
ABSPATH=os.path.dirname(ABSPATH)+"/"
flag='0'

from LHFrameStd import MultiTFvpPOC, plot_all_multiftfpoc_vars

class trade_coin(object):
    def __init__(self,symbol,user,asset ):
         acclist = self.import_account()
         okexacc = acclist['okex_v5_excute']
         self.api_key=okexacc['apikey']
         self.seceret_key=okexacc['secretkey']
         self.passphrase=okexacc['passphrase']
         self.login()
         self.asset=asset
         self.symbol=symbol
         self.user=user
         self.position_time=0
         self.asset_time=0
         self.asset_record = deque(maxlen=1440)

         self.save_pic_interval = 100
         self.save_pic_counter = 0

         if 'ETH' in self.symbol:
             self.asset_coe=200
         else:
             self.asset_coe=200  #资金分配系数，5/2000
         try:
              self.accountAPI.get_position_mode('long_short_mode')
              account_config=self.accountAPI.get_account_config()
              self.autoLoan=account_config['data'][0]['autoLoan']#true为自动借币
              self.margin_model=account_config['data'][0]['acctLv']#'3'3为跨币种保证金模式
              self.fv={}
              self.accountAPI.set_leverage('20', 'cross', symbol,'','long')
              self.accountAPI.set_leverage('20', 'cross', symbol,'','short')
              self.fv[symbol]=self.get_face_value(symbol)
             
         except Exception as e:
              print('设置出错',e)
              time.sleep(5)
              pass
         self.upl_open_condition()

         '''数据初始化'''
         while(1):
             try:
                 pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'rb')
                 buy_total = pickle.load(pk1)
                 pk1.close()
                 break
             except:
                 time.sleep(2)
                 print('buy_total出错')
                 buy_total = pd.DataFrame(np.zeros([1,6])) 
                 buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                 pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'wb')
                 pickle.dump(buy_total,pk1)
                 pk1.close()
         while(1):
             try:
                 pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'rb')
                 buy_total = pickle.load(pk1)
                 pk1.close()
                 break
             except:
                 print('buy_total出错')
                 time.sleep(2)
                 buy_total = pd.DataFrame(np.zeros([1,6])) 
                 buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                 pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'wb')
                 pickle.dump(buy_total,pk1)
                 pk1.close()

    def trade1(self):
        try:
            print('运行中',self.symbol)
            if time.time()-self.asset_time>60:
                self.asset_time=time.time()
                usdt_total=self.get_usdt_total()
                self.asset_record.append(usdt_total)
                self.upl_open_condition()
                # print('--1'*1000,self.symbol,self.asset_record)
            if len(self.asset_record)>=0:
                try:
                   usdt_total 
                except:
                   usdt_total=self.get_usdt_total()
                   self.asset_record.append(usdt_total)
                   if len(self.asset_record)<=0:
                       print('------------------if len(self.asset_record)>=0 false')
                       return
                max_draw=usdt_total/max(self.asset_record)
            else:
                max_draw=1
            if max_draw < 1 - 0.1:
                self.asset_normal=0
                print('当日资金回撤百分比超过%s'%(1-max_draw)*100,'当日停止开仓'*100)
            else:
                self.asset_normal=1
                print('当日资金回撤百分比为%s'%(1-max_draw)*10,usdt_total,max(self.asset_record))
            
            time11 = datetime.datetime.now()
            self.coin_date=self.get_kline()
            try:
                lock.acquire()
                _pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'rb')
                buy_total_long = pickle.load(_pk1)
                _pk1.close()
                time.sleep(0.1)
            except Exception as e:
                time.sleep(0.1)
                print('--3'*50,e)
            lock.release()
            
            try:
                lock.acquire()
                _pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'rb')
                buy_total_short = pickle.load(_pk1)
                _pk1.close()
                time.sleep(0.1)
            except Exception as e:
                time.sleep(0.1)
                print('--2'*50,e)  
            lock.release()
            '''
            索引	含义	说明
            0	时间（Timestamp）	日期时间，格式 "YYYY-MM-DD HH:MM:SS"
            1	开盘价 (Open)	该时间周期的开盘价格
            2	最高价 (High)	该时间周期的最高价格
            3	最低价 (Low)	该时间周期的最低价格
            4	收盘价 (Close)	该时间周期的收盘价格
            5	成交量 (Volume)	该时间周期的成交量或成交额
            '''
            window_tau_1m = 30
            window_tau_1h = window_tau_1m * 56
            multFramevpPOC = MultiTFvpPOC(window_LFrame=window_tau_1m, window_HFrame=window_tau_1h)
            multFramevpPOC.calculate_HFrame_vpPOC_and_std(self.coin_date)
            rsi_ema_recent = multFramevpPOC.rsi_with_ema_smoothing(self.coin_date).iloc[-1]
            
            LFrame_vpPOCs = multFramevpPOC.LFrame_vpPOC_series
            HFrame_vpPOCs = multFramevpPOC.HFrame_vpPOC

            if self.save_pic_counter % self.save_pic_interval == 0:
                plot_all_multiftfpoc_vars( multFramevpPOC, self.coin_date, self.symbol)
                time.sleep(0.1)
                # from vpvr_gmm_splited import VPVRAnalyzer
                # analyzer = VPVRAnalyzer(self.coin_date, n_bins=60, n_components=3)
                # result = analyzer.run(self.symbol)
                # print(result['regions'])
                
                if self.save_pic_counter >= 4294967296-1:  #2**32 - 1, avoid value overflow
                    self.save_pic_counter = 1
            self.save_pic_counter += 1

            if 1:
                '''开平仓信号计算'''
                print ("\t 开仓信号计算 开始 %s" %time.ctime())
                ohcl5 = self.coin_date.iloc[:,4] 
                hh2 = self.coin_date.iloc[:,2] 
                ll2 = self.coin_date.iloc[:,3] 
                open2 = self.coin_date.iloc[:,1] 
                vol = self.coin_date.iloc[:,5]
                ma5vol=vol.rolling(5).mean()
                ma10vol=vol.rolling(10).mean()
                # LFrame_vpPOCs=(ohcl5.iloc[-20:]*vol.iloc[-20:]).cumsum()/vol.iloc[-20:].cumsum()

                ohcl5 = hh2

                MidLine = ohcl5.rolling(20).mean()
                MidLine_degree=np.arctan((MidLine/(MidLine.shift(1))-1)*100)*180/np.pi
                # UpLine=MidLine+ Offset*Band
                UpLine = multFramevpPOC.HFrame_std_2_0_up
                UpLine_degree=np.arctan((UpLine/(UpLine.shift(1))-1)*100)*180/np.pi
                # DownLine=MidLine - Offset*Band
                DownLine = multFramevpPOC.HFrame_std_2_0_down
                DownLine_degree=np.arctan((DownLine/(DownLine.shift(1))-1)*100)*180/np.pi
                cur_close = ohcl5.iloc[-1]
                
                print('ohcl5[-1]',cur_close,'UpLine[-1]',UpLine.iloc[-1],'DownLine[-1]',DownLine.iloc[-1])

                LFrame_vpPOC_short = 0
                LFrame_vpPOC_long = 0
                HF_STDUpper = multFramevpPOC.HFrame_std_2_0_up.iloc[-1] 
                HF_STDLower = multFramevpPOC.HFrame_std_2_0_down.iloc[-1]
                #不考虑rsi的话，可能小周期的3倍标准差会安全一些，考虑rsi估计用小周期的2倍标准差就可以。
                LF_STDUpper = multFramevpPOC.LFrame_std_2_upper.iloc[-1]
                LF_STDLower = multFramevpPOC.LFrame_std_2_lower.iloc[-1]
                LFrame_vwap = LFrame_vpPOCs.iloc[-1]
                
                minus_LFramePOC_to_HFramePOC_percent_delta = 0.0001  #避免std收窄的时候还在开仓，结果开出了逆势的仓位，容易导致长期扛单。
                minus_single_direction_delta = 0.001

                if 'ETH' in self.symbol or 'BTC' in self.symbol:
                    HF_STDUpper = multFramevpPOC.HFrame_std_1_5_up.iloc[-1]
                    HF_STDLower = multFramevpPOC.HFrame_std_1_5_down.iloc[-1]
                    LF_STDUpper = multFramevpPOC.LFrame_std_2_upper.iloc[-1]
                    LF_STDLower = multFramevpPOC.LFrame_std_2_lower.iloc[-1]
                    minus_LFramePOC_to_HFramePOC_percent_delta = 0.00005
                    minus_single_direction_delta = 0.0005
                
                close_to_lframe_vwap_percent = cur_close/LFrame_vwap
                is_short_un_opend = len(buy_total_short)<2
                if is_short_un_opend:
                    minus_short_cond = close_to_lframe_vwap_percent > (1+minus_single_direction_delta) 
                    lf2hf_cond = multFramevpPOC.LFrame_vpPOC_series.iloc[-1]/multFramevpPOC.HFrame_std_0_5_down.iloc[-1] >= 1 + minus_LFramePOC_to_HFramePOC_percent_delta
                    if rsi_ema_recent >= 55 and minus_short_cond and HF_STDUpper <= cur_close and LF_STDUpper <= cur_close and lf2hf_cond:
                        LFrame_vpPOC_short=1
                else:  #加仓条件
                    previous_short_filllPx = buy_total_short.iloc[-1,2]
                    minus_short_cond = cur_close/previous_short_filllPx > 1+minus_single_direction_delta
                    time_cond =  time.time() - buy_total_short.iloc[-1,5]>59
                    if (rsi_ema_recent >= 55 and minus_short_cond and time_cond):
                        LFrame_vpPOC_short=1

                is_long_un_opend = len(buy_total_long)<2
                if is_long_un_opend:
                    minus_long_cond = close_to_lframe_vwap_percent < (1-minus_single_direction_delta) 
                    lf2hf_cond = multFramevpPOC.LFrame_vpPOC_series.iloc[-1]/multFramevpPOC.HFrame_std_0_5_up.iloc[-1] <= 1 - minus_LFramePOC_to_HFramePOC_percent_delta
                    if rsi_ema_recent <= 45 and minus_long_cond and HF_STDLower >= cur_close and LF_STDLower >= cur_close and lf2hf_cond:
                        LFrame_vpPOC_long=1
                else:  #加仓条件
                    previous_long_filllPx = buy_total_long.iloc[-1,2]
                    minus_long_cond = cur_close/previous_long_filllPx < 1-minus_single_direction_delta
                    time_cond = time.time()-buy_total_long.iloc[-1,5] > 5
                    if  (rsi_ema_recent <= 45 and time_cond and minus_long_cond):
                        LFrame_vpPOC_long=1

                if LFrame_vpPOC_long != 1 and LFrame_vpPOC_short != 1:
                    self.cancel_order()

                print(f'symbol={self.symbol}, self.upl_long_open=={self.upl_long_open}, LFrame_vpPOC_long=={LFrame_vpPOC_long\
                        }, self.upl_short_open=={self.upl_short_open==1}, LFrame_vpPOC_short=={LFrame_vpPOC_short \
                        }, close_to_lframe_vwap_percent={close_to_lframe_vwap_percent}, ',
                      '--3'*50)

                if self.asset_normal==1 and self.upl_short_open==1 and LFrame_vpPOC_short==1 :
                    #开空
                    try:  
                        if 1 :
                          usdt_total=self.get_usdt_total()
                          model='buyshort'
                          price0=hh2.iloc[-1]
                          price = price0*(1+0.0001) if len(buy_total_long)>=2 else price0*(1-0.0002)  #首次开仓立刻成交
                          fv=self.fv[self.symbol]
                          amount=max(float(round(usdt_total/self.asset_coe/price0/fv)), 0.01 if ('BTC' in self.symbol or symbol != 'ETH') else 1)
                          symbol=self.symbol
                          place_order=self.create_order1(symbol,price,amount,model)
                          print(place_order,symbol,model)
                          try:
                              logging.info((self.user,symbol,'MidLine.iloc[-1]',MidLine.iloc[-1],'LFrame_vpPOC',LFrame_vpPOCs.iloc[-1],'price0',price0,time11,model))
                          except:
                              pass
                          orderid2= place_order['data'][0]['ordId']
                          orderinfo2=self.tradeAPI.get_orders(self.symbol,orderid2)['data'][0]
                          isfill2=orderinfo2['state']
                          fillPx=float(orderinfo2['fillPx'])
                          fillSz=float(orderinfo2['fillSz'])
                          holdC0 = pd.DataFrame(np.zeros([1,6]))
                          holdC0.iloc[0,:]=time11,'short',fillPx,fillSz,fillPx*fillSz*fv,time.time()
                          buy_total_short=buy_total_short.append(holdC0,ignore_index=True)
                          print('buy_total_short',buy_total_short)
                          pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'wb')
                          pickle.dump(buy_total_short,pk1)
                          pk1.close()
                          print('orderid2',orderid2,'orderinfo2',orderinfo2,'fillPx',fillPx,'fillSz',fillSz)
                          time.sleep(2)
                    except Exception as e:
                          time.sleep(1)
                          print('buyshort erro1',e)
                    self.upl_open_condition()

                
                if self.asset_normal==1 and self.upl_long_open==1 and LFrame_vpPOC_long==1:
                    #开多
                    try:  
                        if 1:
                          usdt_total=self.get_usdt_total()
                          model='buylong'
                          price0=ll2.iloc[-1]
                          price=price0*(1+0.0002) if len(buy_total_long)>=2 else price0*(1-0.0001)  #首次开仓立刻成交
                          fv=self.fv[self.symbol]
                          amount=max(float(round(usdt_total/self.asset_coe/price0/fv)), 0.01 if ('BTC' in self.symbol or symbol != 'ETH') else 1)
                          symbol=self.symbol
                          place_order=self.create_order1(symbol,price,amount,model)
                          try:
                              logging.info((self.user,symbol,'MidLine.iloc[-1]',MidLine.iloc[-1],'LFrame_vpPOC',LFrame_vpPOCs.iloc[-1],'price0',price0,time11,model))
                          except:
                              pass
                          print(place_order,symbol,model)
                          orderid2= place_order['data'][0]['ordId']
                          orderinfo2=self.tradeAPI.get_orders(self.symbol,orderid2)['data'][0]
                          isfill2=orderinfo2['state']
                          fillPx=float(orderinfo2['fillPx'])
                          fillSz=float(orderinfo2['fillSz'])
                          holdC0 = pd.DataFrame(np.zeros([1,6]))
                          holdC0.iloc[0,:]=time11,'long',fillPx,fillSz,fillPx*fillSz*fv,time.time()
                          buy_total_long=buy_total_long.append(holdC0,ignore_index=True)
                          print('buy_total_long',buy_total_long)
                          pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'wb')
                          pickle.dump(buy_total_long,pk1)
                          pk1.close()
                          print('orderid2',orderid2,'orderinfo2',orderinfo2,'fillPx',fillPx,'fillSz',fillSz)
                          time.sleep(2)
                    except Exception as e:
                          time.sleep(1)
                          print('buylong erro1',e) 
                    self.upl_open_condition()
                #平仓
                if len(buy_total_long)>=2 or len(buy_total_short)>=2 or time.time()-self.position_time>10:
                    self.position_time=time.time()
                    usdt_total=self.get_usdt_total()
                    notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict,upl_long_dict,upl_short_dict=self.get_entryprice_okx()
                    print(notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict)
                    print('--2'*200,upl_long_dict,upl_short_dict,sum(upl_long_dict.values()),sum(upl_short_dict.values()))

                    symbol=self.symbol
                    try:
                        upl_short=upl_dict[symbol+'-SHORT']
                        amount_short=positionAmount_dict_sub[symbol+'-SHORT']
                        notionalUsd_short=notionalUsd_dict[symbol+'-SHORT']
                        price_short= markprice_dict[symbol+'-SHORT']
                    except:
                        upl_short=0
                        amount_short=0
                        notionalUsd_short=0
                    
                    try:
                        upl_long=upl_dict[symbol+'-LONG']
                        amount_long=positionAmount_dict_sub[symbol+'-LONG']
                        notionalUsd_long=notionalUsd_dict[symbol+'-LONG']
                        price_long= markprice_dict[symbol+'-LONG']
                    except:
                        upl_long=0
                        amount_long=0
                        notionalUsd_long=0
                    if  'SOL' in self.symbol:
                         stop_profit=0.01
                    else:
                        stop_profit=0.01

                    cross_hframe_poc_min_profit = 0.005
                    if amount_short>0 and (upl_short/notionalUsd_short>stop_profit or
                                           (upl_short/notionalUsd_short > cross_hframe_poc_min_profit and cur_close <= HFrame_vpPOCs.iloc[-1])):
                        model='sellshort'
                        price0=cur_close
                        price=price0*(1-0.0001)
                        amount=max(amount_short, 1)
                        symbol=self.symbol
                        place_order=self.create_order1(symbol,price,amount,model)
                        self.cancel_order()
                        print(place_order,symbol,model)
                        buy_total = pd.DataFrame(np.zeros([1,6])) 
                        buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                        pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'wb')
                        pickle.dump(buy_total,pk1)
                        pk1.close()
                    if amount_long>0 and (upl_long/notionalUsd_long>stop_profit or
                                          (upl_long/notionalUsd_long > cross_hframe_poc_min_profit and cur_close >= HFrame_vpPOCs.iloc[-1])):
                        model='selllong'
                        price0=cur_close
                        price=price0*(1.00 + 0.0001)
                        amount=max(amount_long, 1)
                        symbol=self.symbol
                        place_order=self.create_order1(symbol,price,amount,model)
                        self.cancel_order()
                        print(place_order,symbol,model)
                        buy_total = pd.DataFrame(np.zeros([1,6])) 
                        buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                        pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'wb')
                        pickle.dump(buy_total,pk1)
                        pk1.close()
                
                    if ((upl_long+ upl_short)/usdt_total<-0.15) or self.asset_normal==0 :
                        try:  
                            if self.asset_normal==0:
                               logging.info((self.user,'当日回撤过大所有仓位止损'))
                            if amount_short>0 :
                              model='sellshort'
                              price=price_short*(1.00 - 0.0003)
                              amount=amount_short
                              print('--1',symbol,model)
                              place_order=self.create_order1(symbol,price,amount,model)
                              buy_total = pd.DataFrame(np.zeros([1,6])) 
                              buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                              pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'wb')
                              pickle.dump(buy_total,pk1)
                              pk1.close()
                              logging.info(('空单止损',symbol,'整体亏损比例',(upl_long+ upl_short)/(notionalUsd_long+notionalUsd_short)))
                        except Exception as e:
                              logging.info(('空单止损e',symbol,e))
                              print('sellshort erro1',e)
                        try:  
                            if amount_long>0 :
                              model='selllong'
                              price=price_long*0.997
                              amount=amount_long
                              print('--1',symbol,model)
                              place_order=self.create_order1(symbol,price,amount,model)
                              buy_total = pd.DataFrame(np.zeros([1,6])) 
                              buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                              pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'wb')
                              pickle.dump(buy_total,pk1)
                              pk1.close()
                              logging.info(('多单止损',symbol,'整体亏损比例',(upl_long+ upl_short)/(notionalUsd_long+notionalUsd_short)))
                        except Exception as e:
                              logging.info(('多单止损e',symbol,e))
                              print('selllong erro1',e)
                
        except Exception as e:
            time.sleep(2)
            traceback.print_exc()
            print('运行出错',e)

    def upl_open_condition(self):
        usdt_total=self.get_usdt_total()
        notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict,upl_long_dict,upl_short_dict=self.get_entryprice_okx()
        if (sum(upl_long_dict.values())/usdt_total<-0.002 and sum(upl_long_dict.values())<sum(upl_short_dict.values())) :
            self.upl_long_open=0
        else:
            self.upl_long_open=1

        if sum(upl_short_dict.values())/usdt_total<-0.002 and sum(upl_short_dict.values())<sum(upl_long_dict.values()):
            self.upl_short_open=0
        else:
            self.upl_short_open=1  

    def get_kline(self):
        a11 = 1
        while(1):
            try:
                f_kline_1m =  self.marketAPI.get_candlesticks(self.symbol,'','','1m') #最新300个K
                assert len(f_kline_1m)>2
                break
            except:
                a11+=1
                time.sleep(1)
                print('\t 获取f_kline_1m数据超时')
                print('\t a11=%s'%a11)
                if a11<10:
                    continue
                else:
                    break
        time.sleep(0.1) #4
        f_kline_1m = f_kline_1m['data']
        coin_date = pd.DataFrame(np.zeros([len(f_kline_1m),len(f_kline_1m[0])])) #时间 开盘价 最高价 最低价 收盘价 交易量 交易量转化BTC数量
        print ("\t 实时K线数据更新 开始 %s" %time.ctime())
        coin_date= pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(f_kline_1m[i][0])/1000)),float(f_kline_1m[i][1]),float(f_kline_1m[i][2]),float(f_kline_1m[i][3]),float(f_kline_1m[i][4]),float(f_kline_1m[i][5])] for i in range(len(f_kline_1m))    ])    
        print ("\t 实时K线数据更新 结束 %s \n" %time.ctime())
        coin_date = coin_date.sort_index(axis = 0,ascending = False)
        coin_date.index = range(len(coin_date))
        return coin_date          
    
    def create_order1(self,symbol,price,amount,model,tag="520ccb3f7df2SUDE"):#bef23d76c2f8SUDE model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
        if model=='buylong':
            place_order=self.tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','',tag,'long',str(price),'')
        elif model=='buyshort':
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','',tag,'short',str(price),'')
        elif model=='selllong':  
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(int(float(amount))),'','',tag,'long',str(price),'')
        elif model=='sellshort':  
            place_order=self.tradeAPI.place_order(symbol,'cross','buy','limit',str(int(float(amount))),'','',tag,'short',str(price),'')
        elif model=='buycash':
            place_order=self.tradeAPI.place_order(symbol,'cross','buy','limit',str(amount),'','',tag,'',str(price),'')
        elif model=='sellcash':
            place_order=self.tradeAPI.place_order(symbol,'cross','sell','limit',str(amount),'','',tag,'',str(price),'')
        return place_order
    
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
    
    def get_future_ticker(self,symbol):          
         while(1):
            try:
                time.sleep(0.5)#1
                print(symbol)
                future_ticker = float(self.marketAPI.get_ticker(symbol)['data'][0]['last'])
                print(future_ticker)
                break
            except Exception as e:
                time.sleep(1)
                print('\t 获取合约现价超时',e,symbol)
                continue
         return future_ticker
    
    def  get_positions_okx_all(self,symbol='BTCUSDT'):   
        a11 = 1
        while(1):
            try:
                position_amount = self.accountAPI.get_positions('')
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
    
    def get_usdt_total(self):
         while(1):
            try:
                lock.acquire()
                try:
                    account_info = self.accountAPI.get_account()
                    time.sleep(0.5)
                except:
                    time.sleep(0.5)
                    pass
                lock.release()
                totalEq=float(account_info['data'][0]['totalEq'])
                
                break
            except Exception as e:
                print('\t 获取总资金失败',e)
                self.setsystem_time()
                time.sleep(10)
                continue
         return totalEq 
   
    def cancel_order(self):
        try:
            lock.acquire()
            unfill=self.tradeAPI.get_order_list(instId='')['data']
            time.sleep(0.5)
            print('查询订单请求')
        except:
            time.sleep(0.5)
        lock.release()
        if len(unfill)>0:
            print('撤单开始')
            for i in range(len(unfill)):
                instid=unfill[i]['instId']
                if instid == self.symbol:
                    orderid=unfill[i]['ordId']
                    self.tradeAPI.cancel_order(instid, orderid)

    def  get_entryprice_okx(self,symbol='BTCUSDT'):
         a11 = 1
         while(1):
             try:
                 positionAmount_dict={}
                 notionalUsd_dict={}
                 markprice_dict={}
                 upl_dict={}
                 upl_long_dict={}
                 upl_short_dict={}
                 position_amount = self.accountAPI.get_positions('')
                 position_amount=position_amount['data']
                 # print(position_amount)
                 if len(position_amount)>0 :
                     for i in range(len(position_amount)):
                         if abs(float(position_amount[i].get('pos')))>0:
                            positionAmount =float(position_amount[i].get('pos'))
                            notionalUsd =float(position_amount[i].get('notionalUsd'))
                            markprice =float(position_amount[i].get('last'))
                            symbol=position_amount[i].get('instId')
                            direct=position_amount[i].get('posSide')
                            upl=float(position_amount[i].get('upl'))
                            if direct=='long':
                                symbol= symbol+'-LONG'
                                upl_long_dict[symbol]=upl
                            elif direct=='short':
                                symbol= symbol+'-SHORT'
                                upl_short_dict[symbol]=upl
                            positionAmount_dict[symbol]=positionAmount
                            notionalUsd_dict[symbol]=notionalUsd
                            markprice_dict[symbol]=markprice
                            upl_dict[symbol]=upl
                            
                 break
             except Exception as e:
                 a11+=1
                 print('\t 获取okx主账户持仓数量超时1',e)
                 print('\t a11=%s'%a11)
                 time.sleep(1)
                 if a11<5:
                     continue
                 else:
                     break
         return notionalUsd_dict, markprice_dict , positionAmount_dict, upl_dict,upl_long_dict,upl_short_dict
    
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

    def get_usdt_total(self):
          lock.acquire()
          while(1):
             try:
                 try:
                     account_info = self.accountAPI.get_account()
                     time.sleep(0.5)
                 except:
                     time.sleep(0.5)
                 totalEq=float(account_info['data'][0]['totalEq'])
                 lock.release()
                 break
             except Exception as e:
                 print('\t 获取总资金失败',e)
                 self.setsystem_time()
                 time.sleep(1)
                 continue
          return totalEq 
    
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

    def import_account(self):
         filepath = ABSPATH+'Account.json'
         f = open(filepath).read()
         accountlist = json.loads(f)
         return accountlist 
    
if __name__=='__main__':
    aa1=trade_coin('ETH-USDT-SWAP','yyyyy2_okx',1500)
    aa2=trade_coin('XRP-USDT-SWAP','yyyyy2_okx',1500)
    aa3=trade_coin('SOL-USDT-SWAP','yyyyy2_okx',1500)
    # aa4=trade_coin('DOGE-USDT-SWAP','yyyyy2_okx',1500)
    aa5=trade_coin('TRUMP-USDT-SWAP','yyyyy2_okx',1500)
    aa6=trade_coin('BTC-USDT-SWAP','yyyyy2_okx',1500)
    while(1):
        try:
            time1=datetime.datetime.now()
            time11=time1.hour*100+time1.minute
            week_day=time1.isoweekday()
            threads=[]
            t1 = threading.Thread(target = aa1.trade1) 
            threads.append(t1)
            t2 = threading.Thread(target = aa2.trade1) 
            threads.append(t2)
            t3 = threading.Thread(target = aa3.trade1) 
            threads.append(t3)
            # t4 = threading.Thread(target = aa4.trade1) 
            # threads.append(t4)
            t5 = threading.Thread(target = aa5.trade1) 
            threads.append(t5)
            t6 = threading.Thread(target = aa6.trade1) 
            threads.append(t6)
            i=0
            for t in threads:
                i+=1
                t.start()
                time.sleep(0.1)
                if i%3==0:
                    time.sleep(3)
            for t in threads:
                t.join()
        except:
            time.sleep(5)
            print('启动报错')
   