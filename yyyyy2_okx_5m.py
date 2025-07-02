
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
import builtins


from db_client import SQLiteWALClient
# from strategy import MultiFramePOCStrategy, RuleConfig
from multi_frame_vwap_strategy import MultiFramePOCStrategy, RuleConfig

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)

root = '/Users/a1234/Desktop/trade_excute/trade_records'
ABSPATH=os.path.abspath(sys.argv[0])  
ABSPATH=os.path.dirname(ABSPATH)+"/"
flag='0'

from LHFrameStd import MultiTFvp_poc, WindowConfig
from plot_mtf import  plot_all_multiftfpoc_vars, plot_liquidation_vp
windowConfig = WindowConfig()
LIMIT_K_N_APPEND = max(windowConfig.window_tau_s, 310)
LIMIT_K_N = 400 + LIMIT_K_N_APPEND #+ 1000
DEBUG = False 
DEBUG = True
    
MAX_DAILY_DRAW = 0.06

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
        self.asset_time=0
        self.asset_record = deque(maxlen=1440)
        self.asset_normal = 1
        self.usdt_total = None

        if 'ETH' in self.symbol:
            self.asset_coe=20
        else:
            self.asset_coe=20  #资金分配系数，5/2000
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
        self.usdt_total=self.get_usdt_total()
        self.upl_open_condition()

        '''数据初始化'''
        DB_PATH = f'{symbol}.db'
        self.client = SQLiteWALClient(db_path=DB_PATH, table="combined_30x")
        self.coin_data = self.get_kline(init=True)

        self.multiFrameVwap = MultiTFvp_poc(window_LFrame=windowConfig.window_tau_l, 
                                        window_HFrame=windowConfig.window_tau_h, 
                                        window_SFrame=windowConfig.window_tau_s)
        # self.multiFrameVwap.calculate_SFrame_vwap_poc_and_std(self.coin_data, DEBUG)
       
        # 近期的最高或者最低的sl挂单2分钟后撤单为标准.
        self.strategy = MultiFramePOCStrategy(long_rule=RuleConfig.long_rule, 
                                              short_rule=RuleConfig.short_rule, 
                                              timeout=150,
                                              max_open2equity_pct=4)  #超出后，每分钟最多挂单3个
        self.strategy_log_interval = 0

        self.max_history_long_profit = 0
        self.max_history_short_profit = 0

        self.direction = self.mv_direction()
        self.direction_time = time.time()

        print("direction is: ", self.direction)

    

    def get_open_interest(self):
        oi_result = self.publicAPI.get_open_interest('SWAP', instId=self.symbol)
        
        funding_rate = self.publicAPI.get_funding_rate(instId=self.symbol)
        fundingRate = float(funding_rate['data'][0]['fundingRate'])
        print("######", f"fundingRate={fundingRate}  持仓手续费的正负代表--{'空头居多' if fundingRate <= 0 else '多头居多'}")
    
        print('****** oi_result:', oi_result)
        
        return 
    
        odb = self.marketAPI.get_orderbook(instId=self.symbol, sz=10)
        asks = odb['data'][0]['asks']
        bids = odb['data'][0]['bids']
        print('------orderbook')
        print(asks)
        print(bids)

    def get_liquidation(self):
        # 按最后一个 '-' 拆分
        inst_family = self.symbol.rsplit("-", 1)[0]
        # inst_family 现在就是 "ETH-USDT"
        liquidation_orders = self.publicAPI.get_liquidation_orders('SWAP', instId=self.symbol, uly=inst_family, state='filled')
        liquidation_details = liquidation_orders['data'][0]['details']
        long_px_sz_accum = 0
        long_sz_accum = 0

        short_px_sz_accum = 0
        short_sz_accum = 0

        for item in liquidation_details:
            px, sz = float(item['bkPx']), float(item['sz'])
            if 'long' == item['posSide']:
                long_px_sz_accum += px * sz
                long_sz_accum += sz
            else:
                short_px_sz_accum += px * sz
                short_sz_accum += sz
        liquidation_long_vwap = long_px_sz_accum / long_sz_accum
        liquidation_short_vwap = short_px_sz_accum / short_sz_accum
        print(f'liquidation_long_vwap = {liquidation_long_vwap}, liquidation_short_vwap = {liquidation_short_vwap}')
        
        plot_liquidation_vp(liquidation_details=liquidation_details)

    def trade1(self):
        try:
            print('运行中',self.symbol)
            if time.time() - self.direction_time >= 3600:
                self.direction = self.mv_direction()

            self.get_open_interest()

            self.usdt_total=self.get_usdt_total()
            if time.time()-self.asset_time>60:
                self.asset_time=time.time()
                self.asset_record.append(self.usdt_total)
                self.upl_open_condition()
                # print('--1'*1000,self.symbol,self.asset_record)
        
                if len(self.asset_record)<=0:
                    print('------------------if len(self.asset_record)>=0 false')
                    return
                max_draw=self.usdt_total/max(self.asset_record)
            else:
                max_draw=1

            if max_draw < 1 - MAX_DAILY_DRAW:
                self.asset_normal=0
                print('----当日资金回撤百分比超过%s'%(1-max_draw)*30,'当日停止开仓'*100)
            else:
                self.asset_normal=1
                print('++++当日资金回撤百分比为%s'%(1-max_draw),self.usdt_total,max(self.asset_record))

            self.coin_data = self.get_kline(init=True)
            self.multiFrameVwap.calculate_SFrame_vwap_poc_and_std(self.coin_data, DEBUG)
            
            #debug ploting content
            # plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, False)
            if 1:
                '''开平仓信号计算'''
                print ("\t 开仓信号计算 开始 %s" %time.ctime())
                close = self.coin_data['close']
                high = self.coin_data['high']
                low = self.coin_data['low']
                open = self.coin_data['open'] 
                vol = self.coin_data['vol']
                
                cur_close = close.iloc[-1]
                cur_low = low.iloc[-1]
                cur_high = high.iloc[-1]
                try:
                    notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict,upl_long_dict,upl_short_dict=self.get_entryprice_okx()
                    print(notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict)
                    print('--2'*10,upl_long_dict,upl_short_dict,sum(upl_long_dict.values()),sum(upl_short_dict.values()))
                except Exception as e:
                    print(e)
                    return 
                
                if 1:
                    # 1) 检查超时撤单
                    for side in ("long", "short"):
                        if self.strategy.should_cancel(side):
                            self.cancel_order(side)  
                            self.strategy.clear_order(side)

                    # 2) 评估新信号并下单
                    all_values = sum(positionAmount_dict_sub.values())
                    open2equity_pct = all_values/self.usdt_total

                    sig_short = self.strategy.evaluate("short", cur_close, close, self.multiFrameVwap, open2equity_pct)
                    sig_long  = self.strategy.evaluate("long",  cur_close, close, self.multiFrameVwap,  open2equity_pct)
                    record = self.strategy.eval_history[-1]
                    print(record['side'])
                    if record['tiers']:
                        print(record['tiers'][0])
                    if self.strategy_log_interval % 20 == 0:
                        with builtins.open("eval_logs.json","w") as f:
                            import json
                            json.dump(self.strategy.eval_history[-1], f, default=str)
                        if self.strategy_log_interval > 9999999:
                            self.strategy_log_interval = 0
                    self.strategy_log_interval += 1
                else:
                    time.sleep(60)
                    sig_long = None
                    sig_short = None
                    
                cur_SFrame_vwap_up_poc = self.multiFrameVwap.SFrame_vwap_up_poc.iloc[-1]
                cur_SFrame_vwap_down_poc = self.multiFrameVwap.SFrame_vwap_down_poc.iloc[-1]
                cur_HFrame_vwap_down_sl = self.multiFrameVwap.HFrame_vwap_down_sl.iloc[-1]
                cur_HFrame_vwap_up_sl = self.multiFrameVwap.HFrame_vwap_up_sl.iloc[-1]

                if cur_close < cur_SFrame_vwap_down_poc or cur_close > cur_SFrame_vwap_up_poc:
                    self.get_liquidation()
                if sig_long != None or sig_short != None:
                    print(f'symbol={self.symbol}, self.upl_long_open=={self.upl_long_open}, \n\t sig_long=={sig_long\
                            },\nself.upl_short_open=={self.upl_short_open==1}, \n\tsig_short=={sig_short}, ',
                        '--3'*100)
                else:
                    if time.time() - self.asset_time > 120:
                        self.cancel_order()


                atr_1x =  self.multiFrameVwap.atr_dic['ATR'].iloc[-1]
                if 1:
                    if self.direction == -2:
                        negative_atr = atr_1x
                        positive_atr = 0
                    elif self.direction == 2:
                        negative_atr = 0
                        positive_atr = atr_1x
                    elif self.direction == -1:
                        negative_atr = atr_1x / 2
                        positive_atr = 0
                    elif self.direction == 1:
                        negative_atr = 0
                        negative_atr = atr_1x / 2
                    else:
                        negative_atr = 0
                        positive_atr = 0
                else:
                    negative_atr = 0
                    positive_atr = 0

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
                    
                if self.asset_normal==1 and self.upl_short_open==1 and sig_short != None and sig_short.action:
                    #开空
                    try:  
                        if amount_long == 0 and 1 :
                            self.usdt_total=self.get_usdt_total()
                            model='buyshort'
                            if amount_short == 0:
                                price0=max(sig_short.price + positive_atr, self.coin_data['high'].tail(20).max()) 
                            else:  #一开仓按照最新的最低价挂单，因为之前触及后，sl会移动，导致后续虽然成本更差，但是不一定还能触及sl等价格。
                                price0=min(sig_short.price, self.coin_data['high'].tail(20).max())
                            price = price0
                            fv=self.fv[self.symbol]
                            amount=sig_short.amount  #max(float(round(self.usdt_total/self.asset_coe/price0/fv)), 1) #* get_martingale_coefficient(len(record_buy_total_short))
                            
                            symbol=self.symbol
                            place_order=self.create_order1(symbol,price,amount,model) 
                            print(place_order,symbol,model)
                            try:
                                # plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, True)
                                logging.info((self.user,symbol,'sig_short.price',sig_short.price,'price0',price0,time11,model))
                            except:
                                pass
                            
                    except Exception as e:
                          print('buyshort erro1',e)
                    self.upl_open_condition()

                
                if self.asset_normal==1 and self.upl_long_open==1 and sig_long != None and sig_long.action:
                    #开多
                    try:  
                        if 1 and amount_short == 0:
                            self.usdt_total=self.get_usdt_total()
                            model='buylong'
                            if amount_long == 0:
                                price0=min(sig_long.price - positive_atr, self.coin_data['low'].tail(20).min()) #if is_long_un_opend else cur_low  #首次开仓立刻成交。limit挂单需要配合撤单 
                            else:
                                price0=min(sig_long.price, self.coin_data['low'].tail(20).min()) 
                            price=price0
                            fv=self.fv[self.symbol]
                            amount=  sig_long.amount #max(float(round(self.usdt_total/self.asset_coe/price0/fv)), 1) #* get_martingale_coefficient(len(record_buy_total_long))
                            symbol=self.symbol
                            place_order=self.create_order1(symbol,price,amount,model)
                            try:
                                # plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, True)
                                logging.info((self.user,symbol,'sig_long.price',sig_long.price,'price0',price0,time11,model))
                            except:
                                pass
                            print('place_order:', place_order,symbol,model)
                           

                    except Exception as e:
                          print('buylong erro1',e) 
                    self.upl_open_condition()
                #平仓
                if 1:
                    # lever_dic = self.accountAPI.get_leverage(self.symbol, 'cross')
                    fee_require_profit = 0.0004 + 0.0001
                    # if len(lever_dic) > 0:
                    #     lever_dic = lever_dic['data'][0]
                    #     fee_require_profit = 0.0004 * float(lever_dic['lever'])
                    force_exit_profit = 0.02  #机构的一半强制止盈要求（伦敦、纽约，因为它的4倍是5%、波动亏损5%人是比较无感的。）
                    exit_required_profit=max(fee_require_profit,atr_1x/cur_close * 2)  
                    
                    ChineseTradeTime = False
                    center_tp_poc = self.multiFrameVwap.SFrame_vwap_poc.iloc[-1]  #(self.multiFrameVwap.SFrame_vwap_up_getin.iloc[-1] - self.multiFrameVwap.SFrame_vwap_down_getin.iloc[-1])/2 + self.multiFrameVwap.SFrame_vwap_down_getin.iloc[-1]
                    NoneTradeTimeProfit = 0.0025
                    # if not ChineseTradeTime:
                    #     exit_required_profit = min(exit_required_profit, 0.0025)  #20倍杠杆的4%，夜间盘看不到就当做马丁止盈吧。 后续加一下时间段来区分。
                    
                    if amount_short>0:
                        def close_short(close_amount, price0, instantly = False):
                            if close_amount > 0:
                                model='sellshort'
                                price=price0 + negative_atr if instantly else price0 - positive_atr
                                amount=max(close_amount, 1)
                                symbol=self.symbol
                                place_order=self.create_order1(symbol,price,amount,model)
                                
                                print(place_order,symbol,model)
                                plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, False)

                        short_profit = upl_short / notionalUsd_short
                        # if short_profit <= self.max_history_short_profit / 2 and self.max_history_short_profit > 0.004:
                        #     close_short(amount_short, cur_close, instantly=True)
                        #     if self.no_holdin_orders("short"):
                        #         self.max_history_short_profit = 0
                        if short_profit >= force_exit_profit:
                            close_short(amount_short, cur_close, instantly=True)
                            if self.no_holdin_orders("short"):
                                self.max_history_short_profit = 0
                        else:
                            self.max_history_short_profit = max(self.max_history_short_profit, short_profit)

                            # 1) 严格止损：最近 3 根 K 线里 >=2 根  cur_HFrame_vwap_down_sl 向下突破，直接全部平
                            consecutive_belowsl = (self.multiFrameVwap.HFrame_vwap_down_sl.iloc[-3:].to_numpy() > close.iloc[-3:].to_numpy()).sum() >= 2
                            # 2) 中心线止盈
                            consecutive_cross_center = (self.multiFrameVwap.SFrame_vwap_poc.iloc[-6:].to_numpy()  > close.iloc[-6:].to_numpy() ).sum() >= 4  and self.multiFrameVwap.SFrame_vwap_poc.iloc[-1] < self.multiFrameVwap.HFrame_vwap_poc.iloc[-1]
                            cross_center_and_tp = consecutive_cross_center \
                                                and short_profit > exit_required_profit and self.coin_data['vol'].iloc[-1] > self.multiFrameVwap.vol_df['sma_scaled'].iloc[-1]
                            # 3) 下轨止盈
                            cross_down_and_tp = (cur_close <= cur_SFrame_vwap_down_poc) \
                                                and (short_profit > fee_require_profit)
                            # 4) 常规止损：跌破  cur_HFrame_vwap_down_sl（不论盈亏都平）
                            below_sl = (cur_close <=  cur_HFrame_vwap_down_sl)

                            # ---- 执行顺序：①连续突破 ②中心止盈 ③下轨止盈 ④常规止损
                            close_amount = 0
                            price0       = cur_close

                            if consecutive_belowsl:
                                close_amount = amount_short
                                price0 = cur_HFrame_vwap_down_sl
                            elif below_sl:
                                close_amount = min(3, amount_short)
                                price0       = cur_close
                            elif cross_down_and_tp:
                                close_amount = min(2, amount_short)
                                price0       = cur_SFrame_vwap_down_poc
                            elif cross_center_and_tp:
                                close_amount = 1 #amount_long
                                price0 = cur_SFrame_vwap_down_poc

                            close_short(close_amount, price0)
                    if amount_long>0:
                        def close_long(close_amount, price0, instantly = False):
                            if close_amount > 0:
                                model='selllong'
                                price=price0 - positive_atr if instantly else price0 + positive_atr
                                amount=max(close_amount, 1)
                                symbol=self.symbol
                                place_order=self.create_order1(symbol,price,amount,model)
                                
                                print(place_order,symbol,model)
                                plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, False)

                        long_profit = upl_long / notionalUsd_long

                        # if long_profit <= self.max_history_long_profit / 2 and self.max_history_long_profit > 0.004:
                        #     close_long(amount_long, cur_close, instantly = True)
                        #     if self.no_holdin_orders("short"):
                        #         self.max_history_long_profit = 0
                        if long_profit >= force_exit_profit:
                            close_long(amount_long, cur_close, instantly = True)
                            if self.no_holdin_orders("short"):
                                self.max_history_long_profit = 0
                        else:
                            self.max_history_long_profit = max(self.max_history_long_profit, long_profit)

                            # 1) 最近 3 根 K 线里 ≥2 根 cur_HFrame_vwap_up_sl 被突破 → 直接全平
                            consecutive_upponsl = (self.multiFrameVwap.HFrame_vwap_up_sl.iloc[-3:].to_numpy()  < close.iloc[-3:].to_numpy() ).sum() >= 2  
                            # 2) 中心线止盈：收盘在中心线上方，且利润达 exit_required_profit
                            consecutive_cross_center = (self.multiFrameVwap.SFrame_vwap_poc.iloc[-6:].to_numpy()   < close.iloc[-6:].to_numpy() ).sum() >= 4 and self.multiFrameVwap.SFrame_vwap_poc.iloc[-1] > self.multiFrameVwap.HFrame_vwap_poc.iloc[-1]
                            cross_center_and_tp = consecutive_cross_center and (long_profit > exit_required_profit)
                            # 3) SFrame 上轨止盈：收盘突破 SFrame 上轨，且利润达 fee_require_profit
                            cross_up_and_tp = (cur_close >= cur_SFrame_vwap_up_poc) \
                                            and (long_profit > fee_require_profit) and  exit_required_profit and self.coin_data['vol'].iloc[-1] > self.multiFrameVwap.vol_df['sma_scaled'].iloc[-1]
                            # 4) 常规止损（突破 HFrame 上轨），不论盈亏都平
                            upon_sl = cur_close >= cur_HFrame_vwap_up_sl

                            # 默认不平仓
                            close_amount = 0
                            price0       = cur_close

                            if consecutive_upponsl:
                                # ① 连续突破 → 全部平
                                close_amount = amount_long
                                price0 = cur_HFrame_vwap_up_sl
                            elif upon_sl:
                                # ④ 常规止损 → 最多平 3
                                close_amount = min(3, amount_long)
                                price0       = cur_close
                            elif cross_up_and_tp:
                                # ③ SFrame 上轨止盈 → 最多平 2
                                close_amount = min(2, amount_long)
                                price0       = cur_SFrame_vwap_up_poc

                            #oi为多头的时候，多头可以拿远一点。  有消息面的时候，以中轨为止盈，避免趋势出来无法反转到上方去止盈。
                            elif cross_center_and_tp:
                                # ② 中心止盈 → 平 1
                                close_amount = 1 #amount_long
                                # price0 = cur_high
                                price0 = cur_SFrame_vwap_up_poc
                        
                            close_long(close_amount, price0)

                    
                    if ((upl_long + upl_short)/self.usdt_total<-0.30) or self.asset_normal==0 :
                        try:  
                            if self.asset_normal==0:
                               logging.info((self.user,'当日回撤过大所有仓位止损'))
                            if amount_short>0 :
                                model='sellshort'
                                price=price_short*(1.00 + 0.0003)
                                amount=amount_short
                                print('--1',symbol,model)
                                place_order=self.create_order1(symbol,price,amount,model)
                                
                                plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, False)

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
                                print(place_order,symbol,model)

                                plot_all_multiftfpoc_vars( self.multiFrameVwap, self.symbol, False)

                                logging.info(('多单止损',symbol,'整体亏损比例',(upl_long+ upl_short)/(notionalUsd_long+notionalUsd_short)))
                        except Exception as e:
                              logging.info(('多单止损e',symbol,e))
                              print('selllong erro1',e)
                
                # #     # from vpvr_gmm_splited import VPVRAnalyzer
                # #     # analyzer = VPVRAnalyzer(self.coin_data, n_bins=60, n_components=3)
                # #     # result = analyzer.run(self.symbol)
                # #     # print(result['regions'])
        except Exception as e:
            time.sleep(0.2)
            traceback.print_exc()
            print('运行出错',e)

    def upl_open_condition(self):
        notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict,upl_long_dict,upl_short_dict=self.get_entryprice_okx()
        if (sum(upl_long_dict.values())/self.usdt_total<-0.002 and sum(upl_long_dict.values())<sum(upl_short_dict.values())) :
            self.upl_long_open=0
        else:
            self.upl_long_open=1

        if sum(upl_short_dict.values())/self.usdt_total<-0.002 and sum(upl_short_dict.values())<sum(upl_long_dict.values()):
            self.upl_short_open=0
        else:
            self.upl_short_open=1  
    def no_holdin_orders(self, direction:str):
        notionalUsd_dict, markprice_dict, positionAmount_dict, upl_dict, upl_long_dict, upl_short_dict = \
            self.get_entryprice_okx()
        if direction == "short":
            return positionAmount_dict.get(f"{self.symbol}-SHORT", 0) > 0
        else:
            return positionAmount_dict.get(f"{self.symbol}-LONG", 0) > 0

    def mv_direction(self):
        kline = self.get_1h_kline()
        ma_5 = kline['close'].ewm(span=5, adjust=False).mean()
        ma_14 = kline['close'].ewm(span=14, adjust=False).mean()
        ma_30 = kline['close'].ewm(span=30, adjust=False).mean()

        def ma_signal(ma_5, ma_14, ma_30,
              th_extreme=0.0):
            """
            参数
            ----
            ma_5, ma_14, ma_30 : float
                三条均线的当前值
            th_extreme : float
                用来区分“极端”与“普通”的阈值，
                默认为 0.0（即只看正负，不分幅度）。
                如果你想“极端看多”要求 d1, d2 均 > 某个正数，
                就把它设大一些，比如设为 0.01*ma_14。

            返回
            ----
            sig : int
                2   极端看多  
                1   看多  
                0   中性/多空交错  
            -1   看空  
            -2   极端看空
            """
            d1 = ma_5  - ma_14
            d2 = ma_14 - ma_30

            # 两者同向且幅度都超过阈值 → 极端
            if d1 > th_extreme and d2 > th_extreme:
                return 2
            if d1 < -th_extreme and d2 < -th_extreme:
                return -2

            # 两者同向（>0 或 <0）→ 普通多/空
            if d1 > 0 and d2 > 0:
                return 1
            if d1 < 0 and d2 < 0:
                return -1

            # 其余情况（多空不一致或有一条差值为 0）当中性
            return 0


        return ma_signal(
            ma_5.iloc[-1],
            ma_14.iloc[-1],
            ma_30.iloc[-1],
            th_extreme=0.0
        )
        
    def get_1h_kline(self):
        a11 = 1
        while(1):
            try:
                f_kline =  self.marketAPI.get_candlesticks(self.symbol,'','','1D', limit=600) #最新100个K
                klen = len(f_kline['data'])
                assert klen>2
                break
            except Exception as e:
                a11+=1
                time.sleep(0.5)
                print('\t 获取f_kline数据超时', e)
                print('\t a11=%s'%a11)
                if a11<20:
                    continue
                else:
                    break
        """
        如果返回的是 6 列（一般是现货 Spot）
        [
        0: "ts" // 毫秒时间戳
        1: "open" // 开盘价
        2: "high" // 最高价
        3: "low" // 最低价
        4: "close" // 收盘价
        5: "volume" // 成交量（基准币种）
        ]

        如果返回的是 9 列（合约、永续等）
        [
        0: "ts" // 毫秒时间戳
        1: "open" // 开盘价
        2: "high" // 最高价
        3: "low" // 最低价
        4: "close" // 收盘价
        5: "vol" // 成交量（合约张数或基准币种量）
        6: "volCcy" // 成交量（计价币种量）
        7: "turnover" // 成交额（基准币*价格≈计价币）
        8: "tradeCount" // 成交笔数
        ]

        """
        df = pd.DataFrame(f_kline['data'], columns=['timestamp','open','high','low','close','volume', 'volCcy', 'turnover', 'tradeCount'])

        # 3. 类型转换 & 索引
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df
    
    def get_kline(self, init=False):
        """
        init: 是否首次抓取
        self.coin_data: 历史拼接过的数据
        self.last_timestamp: 记录最后K线的时间戳
        """

        # 1. 从 SQLite 读最新 2000 条
        try:
            # 先拿最新 2000 条（倒序）
            df = self.client.read_df(limit=LIMIT_K_N if init else LIMIT_K_N_APPEND, order_by="ts DESC")
            df["ts"] = df["ts"].astype(int)
            df = df.drop_duplicates("ts").sort_values("ts")
            df["datetime"] = pd.to_datetime(df["ts"], unit="s")
            df = df.set_index("ts", drop=True)
           
            # 6) 保证数据是连续、升序的
            df = df.sort_index()
            # print(df.tail)
            # print(df.head)
            
        except Exception as e:
            print(f"读取数据库错误：{e}", '&&&'*10)
            return None

        return df
    
    def close_all_positions(self):
        """
        一键平掉所有多/空仓。
        返回一个 dict，包含每个子操作的执行结果。
        """
        results = {"short": None, "long": None, "emergency": None}

        # 1. 拉最新资金和仓位信息
        self.usdt_total, markprice_dict = self.get_usdt_total(), None
        notionalUsd_dict, markprice_dict, positionAmount_dict, upl_dict, upl_long_dict, upl_short_dict = \
            self.get_entryprice_okx()

        # 2. 关闭空仓
        amount_short = positionAmount_dict.get(f"{self.symbol}-SHORT", 0)
        price_short  = markprice_dict.get(f"{self.symbol}-SHORT", 0)
        if amount_short > 0:
            try:
                model = "sellshort"
                # 参考现价略打折
                px = price_short * (1 + 0.0005)
                order = self.create_order1(self.symbol, px, amount_short, model)
                
                results["short"] = {
                    "order": order,
                    "filled_amount": amount_short,
                    "price": px
                }
                # 清空本地记录文件
                csv_path = os.path.join(root, f"record_buy_total_short_{self.symbol}_{self.user}.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                # 取消残留挂单
                self.cancel_order()
            except Exception as e:
                results["short"] = {"error": str(e)}
                print('***close_position_err***'* 5, e)

        # 3. 关闭多仓
        amount_long = positionAmount_dict.get(f"{self.symbol}-LONG", 0)
        price_long  = markprice_dict.get(f"{self.symbol}-LONG", 0)
        if amount_long > 0:
            try:
                model = "selllong"
                px = price_long * (1 - 0.0005)
                order = self.create_order1(self.symbol, px, amount_long, model)
                
                results["long"] = {
                    "order": order,
                    "filled_amount": amount_long,
                    "price": px
                }
                csv_path = os.path.join(root, f"record_buy_total_long_{self.symbol}_{self.user}.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                self.long_append_counter = 1

                # 取消残留挂单
                self.cancel_order()
            except Exception as e:
                results["long"] = {"error": str(e)}
                print('***close_position_err***'* 5, e)

        return results
    
    def create_order1(self,symbol,price,amount,model,tag="520ccb3f7df2SUDE"):#bef23d76c2f8SUDE model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
        self.create_order_with_stop_loss(symbol, price, amount, model, tag, stop_loss_pct=1)
        return
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
    
    def create_order_with_stop_loss(self, symbol, price, amount, model, tag="520ccb3f7df2SUDE", stop_loss_pct=0.01):
        # 计算止损价格
        if model == 'buylong':
            stop_loss_price = price * (1 - stop_loss_pct)  # 买入时止损价格
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'buy', 'limit', str(int(float(amount))), '', '', tag, 'long', str(price), '')
            self.tradeAPI.place_algo_order(symbol, 'cross', 'sell', 'conditional', str(int(float(amount))), '', 'long', '', str(stop_loss_price), str(stop_loss_price))
        
        elif model == 'buyshort':
            stop_loss_price = price * (1 + stop_loss_pct)  # 卖出短仓时止损价格
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'sell', 'limit', str(int(float(amount))), '', '', tag, 'short', str(price), '')
            self.tradeAPI.place_algo_order(symbol, 'cross', 'buy', 'conditional', str(int(float(amount))), '', 'short', '', str(stop_loss_price), str(stop_loss_price))
        
        elif model == 'selllong':
            stop_loss_price = price * (1 + stop_loss_pct)  # 平仓时止损价格
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'sell', 'limit', str(int(float(amount))), '', '', tag, 'long', str(price), '')
            self.tradeAPI.place_algo_order(symbol, 'cross', 'buy', 'conditional', str(int(float(amount))), '', 'long', '', str(stop_loss_price), str(stop_loss_price))
        
        elif model == 'sellshort':
            stop_loss_price = price * (1 - stop_loss_pct)  # 平仓时止损价格
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'buy', 'limit', str(int(float(amount))), '', '', tag, 'short', str(price), '')
            self.tradeAPI.place_algo_order(symbol, 'cross', 'sell', 'conditional', str(int(float(amount))), '', 'short', '', str(stop_loss_price), str(stop_loss_price))
        
        elif model == 'buycash':
            # 现金买入时不设置止损
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'buy', 'limit', str(amount), '', '', tag, '', str(price), '')
        
        elif model == 'sellcash':
            # 现金卖出时不设置止损
            place_order = self.tradeAPI.place_order(symbol, 'cross', 'sell', 'limit', str(amount), '', '', tag, '', str(price), '')

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
                time.sleep(0.5)
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
                    
                except:
                    
                    pass
                lock.release()
                totalEq=float(account_info['data'][0]['totalEq'])
                
                break
            except Exception as e:
                print('\t 获取总资金失败',e)
                self.setsystem_time()
                
                continue
         return totalEq 
   
    def cancel_order(self, side = None): #side保留为后续设计，long或者short
        try:
            lock.acquire()
            unfill=self.tradeAPI.get_order_list(instId='')['data']
            time.sleep(0.1)
            print('查询订单请求')
        except:
            time.sleep(0.1)
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
                 time.sleep(0.5)
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
                 time.sleep(0.5)
                 continue
         time.sleep(0.1 )         
         while(1):
              try:
                  self.tradeAPI =trade.TradeAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                  break
              except:
                  print('\t 交易账户登陆超时')
                  time.sleep(0.5)
                  continue
              time.sleep(0.1 ) 
         while(1):
             try:
                 self.fundingAPI = funding.FundingAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                 break
             except:
                 print('\t 资金账户登陆超时')
                 time.sleep(0.5)
                 continue
         time.sleep(0.1 ) 
         while(1):
             try:
                 self.marketAPI = market.MarketAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                 break
             except:
                 print('\t 市场账户登陆超时')
                 time.sleep(0.5)
                 continue
         time.sleep(0.1)
         while(1):
             try:
                 self.publicAPI = public.PublicAPI(self.api_key, self.seceret_key, self.passphrase, False, flag)
                 break
             except:
                 print('\t 公共账户登陆超时')
                 time.sleep(0.5)
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

    def import_account(self):
         filepath = ABSPATH+'Account.json'
         f = open(filepath).read()
         accountlist = json.loads(f)
         return accountlist 
    
if __name__=='__main__':
    aa1=trade_coin('ETH-USDT-SWAP','yyyyy2_okx',1500)
    # aa2=trade_coin('XRP-USDT-SWAP','yyyyy2_okx',1500)
    # aa3=trade_coin('SOL-USDT-SWAP','yyyyy2_okx',1500)
    # aa4=trade_coin('DOGE-USDT-SWAP','yyyyy2_okx',1500)
    # aa5=trade_coin('TRUMP-USDT-SWAP','yyyyy2_okx',1500)
    # aa6=trade_coin('BTC-USDT-SWAP','yyyyy2_okx',1500)
    while(1):
        try:
            time1=datetime.datetime.now()
            time11=time1.hour*100+time1.minute
            week_day=time1.isoweekday()
            threads=[]
            t1 = threading.Thread(target = aa1.trade1) 
            threads.append(t1)
            # t2 = threading.Thread(target = aa2.trade1) 
            # threads.append(t2)
            # t3 = threading.Thread(target = aa3.trade1) 
            # threads.append(t3)
            # t4 = threading.Thread(target = aa4.trade1) 
            # threads.append(t4)
            # t5 = threading.Thread(target = aa5.trade1) 
            # threads.append(t5)
            # t6 = threading.Thread(target = aa6.trade1) 
            # threads.append(t6)
            i=0
            for t in threads:
                i+=1
                t.start()
                time.sleep(7)
            for t in threads:
                t.join()
        except:
            time.sleep(5)
            print('启动报错')
   