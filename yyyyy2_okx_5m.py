
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

import numpy as np  
import pandas as pd  

class MultiTFvpPOC:  
    def __init__(self,  
                 lambd=0.05,  
                 window_LFrame=30,  
                 window_HFrame=720,  
                 std_window_LFrame=30):  
        self.lambd = lambd  
        self.window_LFrame = window_LFrame  
        self.window_HFrame = window_HFrame  
        self.std_window_LFrame = std_window_LFrame  

        # 预定义所有结果属性为None  
        self.LFrame_vpPOC_series = None  
        self.LFrame_ohlc5_series = None  
        self.LFrame_rolling_std = None  
        self.LFrame_std_2_upper = None  
        self.LFrame_std_2_lower = None  
        self.LFrame_std_3_upper = None  
        self.LFrame_std_3_lower = None  

        self.HFrame_vpPOC = None  
        self.HFrame_ohlc5_series = None  
        self.HFrame_price_std = None  

        # HFrame标准差边界会动态赋值，例如self.HFrame_std_1_0_bounds等  

    @staticmethod  
    def calculate_ohlc5(coin_date: pd.DataFrame) -> pd.Series:  
        open_ = coin_date.iloc[:, 1]  
        high = coin_date.iloc[:, 2]  
        low = coin_date.iloc[:, 3]  
        close = coin_date.iloc[:, 4]  

        weights_l, weights_c, weights_h, weights_o = 1.5, 2.0, 1.5, 0.5  
        weight_sum = weights_l + weights_c + weights_h + weights_o  

        ohlc5 = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
        return ohlc5  

    def twpoc_calc_with_lambda(self, coin_date_df):  
        """  
        基于输入的DataFrame计算滑动窗口的时间加权成交量加权价格 (twpoc) 序列，  
        并返回带时间索引的pd.Series，长度与coin_date_df相同。  
        
        说明：  
        - 对coin_date_df每个时间点向前窗口长度内计算twpoc  
        - 时间衰减以 e^(-lambda * i) 递减，i表示距离当前时间的步数  
        """  

        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  
        volume = coin_date_df.iloc[:, 5]  

        decay = np.exp(-self.lambd)  
        n = len(coin_date_df)  
        window = self.window_LFrame  

        # 预分配结果数组  
        twpoc_values = np.full(n, np.nan)  

        # 遍历每个时间点计算带时间衰减的vpPOC  
        for idx in range(n):  
            start_idx = max(0, idx - window + 1)  
            sub_open = open_.iloc[start_idx:idx + 1]  
            sub_high = high.iloc[start_idx:idx + 1]  
            sub_low = low.iloc[start_idx:idx + 1]  
            sub_close = close.iloc[start_idx:idx + 1]  
            sub_volume = volume.iloc[start_idx:idx + 1]  

            # 计算origin_LFrame_vpPOC，用于确定权重，此处取子窗口最近数据  
            origin_LFrame_vpPOC = np.average(sub_close, weights=sub_volume) if sub_volume.sum() > 0 else np.nan  

            if np.isnan(origin_LFrame_vpPOC):  
                twpoc_values[idx] = np.nan  
                continue  

            # 根据origin_LFrame_vpPOC和当前收盘价最后一条决定权重  
            if sub_close.iloc[-1] - origin_LFrame_vpPOC <= 0:  
                weights_l, weights_c, weights_h, weights_o = 1.75, 2, 1.25, 0.5  
            else:  
                weights_l, weights_c, weights_h, weights_o = 1.25, 2, 1.75, 0.5  
            weight_sum = weights_l + weights_c + weights_h + weights_o  

            # 计算ohlc5加权价格序列，长度等于子窗口长度  
            ohlc5 = (sub_low * weights_l + sub_close * weights_c + sub_high * weights_h + sub_open * weights_o) / weight_sum  

            twpoc_num = 0.0  
            twpoc_den = 0.0  
            length = len(sub_close)  

            # 计算滑动窗口内的加权和，权重衰减从当前时间点开始递增(即最近时间权重最大)  
            for i in range(length):  
                w = decay ** i  
                price = ohlc5.iloc[length - 1 - i]  # 从当前时间点逆序取价格  
                vol = sub_volume.iloc[length - 1 - i]  
                twpoc_num += price * vol * w  
                twpoc_den += vol * w  

            if twpoc_den == 0:  
                twpoc_values[idx] = np.nan  
            else:  
                twpoc_values[idx] = twpoc_num / twpoc_den  

        # 构造返回Series，索引保持与输入DataFrame一致  
        return pd.Series(twpoc_values, index=coin_date_df.index) 
    
    def calculate_HFrame_vpPOC_and_std(self, coin_date_df):  
        """  
        计算LFrame和HFrame vpPOC及相关标准差上下轨，所有Series索引与输入df对齐。  
        """  

        # 计算LFrame vpPOC序列（滑动窗口）  
        self.LFrame_vpPOC_series = self.twpoc_calc_with_lambda(coin_date_df)  

        # 计算加权ohlc5价格序列  
        open_ = coin_date_df.iloc[:, 1]  
        high = coin_date_df.iloc[:, 2]  
        low = coin_date_df.iloc[:, 3]  
        close = coin_date_df.iloc[:, 4]  

        weights_l, weights_c, weights_h, weights_o = 1.5, 2, 1.5, 0.5  
        weight_sum = weights_l + weights_c + weights_h + weights_o  
        ohlc5_values = (low * weights_l + close * weights_c + high * weights_h + open_ * weights_o) / weight_sum  
        self.LFrame_ohlc5_series = pd.Series(ohlc5_values.values, index=coin_date_df.index)  

        # LFrame滚动标准差（用于上下轨）  
        self.LFrame_rolling_std = self.LFrame_ohlc5_series.rolling(window=self.std_window_LFrame, min_periods=1).std()  
        self.LFrame_rolling_std.index = coin_date_df.index  

        # 计算LFrame上下轨（基于vpPOC，加减std）  
        self.LFrame_std_2_upper = self.LFrame_vpPOC_series + 2 * self.LFrame_rolling_std  
        self.LFrame_std_2_lower = self.LFrame_vpPOC_series - 2 * self.LFrame_rolling_std  
        self.LFrame_std_3_upper = self.LFrame_vpPOC_series + 3 * self.LFrame_rolling_std  
        self.LFrame_std_3_lower = self.LFrame_vpPOC_series - 3 * self.LFrame_rolling_std  

        # HFrame vpPOC，指数移动平均平滑，索引对齐  
        self.HFrame_vpPOC = self.LFrame_vpPOC_series.ewm(span=self.window_HFrame, adjust=False).mean()  
        self.HFrame_vpPOC.index = coin_date_df.index  

        # HFrame对应ohlc5同LFrame一致  
        self.HFrame_ohlc5_series = self.LFrame_ohlc5_series  

        # HFrame价格标准差（滚动）  
        self.HFrame_price_std = self.HFrame_ohlc5_series.rolling(window=self.window_HFrame, min_periods=1).std()  
        self.HFrame_price_std.index = coin_date_df.index  

        # 计算HFrame多个倍数标准差上下轨（vpPOC加减std）  
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]  
        for m in multipliers:  
            upper = self.HFrame_vpPOC + m * self.HFrame_price_std  
            lower = self.HFrame_vpPOC - m * self.HFrame_price_std  
            key = f"HFrame_std_{str(m).replace('.', '_')}_bounds"  
            setattr(self, key, (upper, lower))  

import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt  
import matplotlib.pyplot as plt  
import time  
import pandas as pd  


def plot_all_multiftfpoc_vars(multFramevpPOC, symbol=''):  
    fig, ax = plt.subplots(figsize=(15, 8))  

    colors = {  
        'LFrame_vpPOC_series': 'yellow',  
        'LFrame_ohlc5_series': 'green',  
        'LFrame_std_2_upper': 'cyan',  
        'LFrame_std_2_lower': 'cyan',  
        'LFrame_std_3_upper': 'lightblue',  
        'LFrame_std_3_lower': 'lightblue',  
        'HFrame_vpPOC': 'purple',  
        'HFrame_ohlc5_series': 'orange',  
    }  

    all_y_values = []  

    for var in [  
        'LFrame_ohlc5_series',  
        'LFrame_std_2_upper', 'LFrame_std_2_lower',  
        'LFrame_std_3_upper', 'LFrame_std_3_lower',  
        'HFrame_ohlc5_series',  
    ]:  
        val = getattr(multFramevpPOC, var, None)  
        if isinstance(val, pd.Series):  
            ax.plot(val.index, val.values, label=var, color=colors.get(var, 'black'), linewidth=1)  
            all_y_values.extend(val.values)  

    lframe_vp = getattr(multFramevpPOC, 'LFrame_vpPOC_series', None)  
    if isinstance(lframe_vp, pd.Series):  
        ax.plot(lframe_vp.index, lframe_vp.values, label='LFrame vpPOC', color='yellow', linewidth=3)  
        all_y_values.extend(lframe_vp.values)  

    hframe_vp = getattr(multFramevpPOC, 'HFrame_vpPOC', None)  
    if isinstance(hframe_vp, pd.Series):  
        ax.plot(hframe_vp.index, hframe_vp.values, label='HFrame vpPOC', color='purple', linewidth=3)  
        all_y_values.extend(hframe_vp.values)  

    fill_colors = {  
        1.5: '#4A90E2',  
        2.0: '#50E3C2',  
        3.0: '#007AFF',  
        3.5: '#32CD32',  
    }  

    std_segments = [  
        (3.5, 3.0),  
        (3.0, 2.0),  
        (2.0, 1.5),  
    ]  

    for upper_mult, lower_mult in std_segments:  
        upper_outer, lower_outer = getattr(multFramevpPOC, f'HFrame_std_{str(upper_mult).replace(".", "_")}_bounds', (None, None))  
        upper_inner, lower_inner = getattr(multFramevpPOC, f'HFrame_std_{str(lower_mult).replace(".", "_")}_bounds', (None, None))  
        if any(v is None for v in [upper_outer, lower_outer, upper_inner, lower_inner]):  
            continue  
        ax.fill_between(upper_outer.index, lower_outer.values, lower_inner.values, color=fill_colors[upper_mult], alpha=0.5)  
        ax.fill_between(upper_outer.index, upper_inner.values, upper_outer.values, color=fill_colors[upper_mult], alpha=0.5)  
        all_y_values.extend(upper_outer.values)  
        all_y_values.extend(lower_outer.values)  
        all_y_values.extend(upper_inner.values)  
        all_y_values.extend(lower_inner.values)  

    if all_y_values:  
        ymin = min(all_y_values) *0.99  
        ymax = max(all_y_values) *1.01  
        ax.set_ylim(ymin, ymax)  

    ax.set_title(f"Combined vpPOC and Std Lines - {symbol}")  
    ax.set_xlabel("Time")  
    ax.set_ylabel("Price/Value")  
    ax.legend(loc='upper left', fontsize='small')  
    ax.grid(True)  

    fig.autofmt_xdate()  
    plt.tight_layout()  

    save_dir = "plots"  
    os.makedirs(save_dir, exist_ok=True)  

    timestamp = int(time.time())  
    prefix = f"{symbol}_" if symbol else ""  
    filename = os.path.join(save_dir, f"{prefix}multFramevpPOC_combined_plot_{timestamp}.png")  
    fig.savefig(filename)  
    plt.close(fig)  
    print(f"Plot saved to file: {filename}")  

# 调用示例：  
# multFramevpPOC = MultiTFvpPOC(...)  
# multFramevpPOC.calculate_HFrame_vpPOC_and_std(coin_date_df)  
# plot_all_multiftfpoc_vars(multFramevpPOC)  

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
         if 'ETH' in self.symbol:
             self.asset_coe=5
         else:
             self.asset_coe=5  #资金分配系数，5/2000
         try:
              self.accountAPI.get_position_mode('long_short_mode')
              account_config=self.accountAPI.get_account_config()
              self.autoLoan=account_config['data'][0]['autoLoan']#true为自动借币
              self.margin_model=account_config['data'][0]['acctLv']#'3'3为跨币种保证金模式
              self.fv={}
              self.accountAPI.set_leverage('2', 'cross', symbol,'','long')
              self.accountAPI.set_leverage('2', 'cross', symbol,'','short')
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
                # print('001'*1000,self.symbol,self.asset_record)
            if len(self.asset_record)>=0:
                try:
                   usdt_total 
                except:
                   usdt_total=self.get_usdt_total()
                max_draw=usdt_total/max(self.asset_record)
            else:
                max_draw=1
            if max_draw <0.96:
                self.asset_normal=0
                print('当日资金回撤百分比超过%s'%(1-max_draw)*100,'当日停止开仓'*100)
            else:
                self.asset_normal=1
                print('当日资金回撤百分比为%s'%(1-max_draw)*10,usdt_total,max(self.asset_record))
            sleep_time = random.randint(1,2)
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
                print('003'*100,e)
            lock.release()
            
            try:
                lock.acquire()
                _pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'rb')
                buy_total_short = pickle.load(_pk1)
                _pk1.close()
                time.sleep(0.1)
            except Exception as e:
                time.sleep(0.1)
                print('002'*100,e)  
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
            window_tau_1h = window_tau_1m * 12
            multFramevpPOC = MultiTFvpPOC(window_LFrame=window_tau_1m, window_HFrame=window_tau_1h)
            multFramevpPOC.calculate_HFrame_vpPOC_and_std(self.coin_date)
            LFrame_vpPOCs = multFramevpPOC.LFrame_vpPOC_series
            HFrame_vpPOCs = multFramevpPOC.HFrame_vpPOC

            plot_all_multiftfpoc_vars( multFramevpPOC, self.symbol)

            if 0:
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

                 #boll
                Band=pd.Series(0.000,index=range(len(ohcl5)))
                std5=pd.Series(0.000,index=range(len(ohcl5)))
                std10=pd.Series(0.000,index=range(len(ohcl5)))
                std20=pd.Series(0.000,index=range(len(ohcl5)))
                std30=pd.Series(0.000,index=range(len(ohcl5)))
                std60=pd.Series(0.000,index=range(len(ohcl5)))
                std90=pd.Series(0.000,index=range(len(ohcl5)))
                std120=pd.Series(0.000,index=range(len(ohcl5)))
                Offset = 2
                Offset1 = 2.1
                for i in range(21,len(ohcl5)):
                     Band[i]=round(np.std(ohcl5[i-20+1:i+1],ddof = 0),4)
                     
                MidLine = ohcl5.rolling(20).mean()
                MidLine_degree=np.arctan((MidLine/(MidLine.shift(1))-1)*100)*180/np.pi
                UpLine=MidLine+ Offset*Band
                UpLine_degree=np.arctan((UpLine/(UpLine.shift(1))-1)*100)*180/np.pi
                DownLine=MidLine - Offset*Band
                DownLine_degree=np.arctan((DownLine/(DownLine.shift(1))-1)*100)*180/np.pi
                print('ohcl5[-1]',ohcl5.iloc[-1],'UpLine[-1]',UpLine.iloc[-1],'DownLine[-1]',DownLine.iloc[-1])
                if 'SOL'in self.symbol and LFrame_vpPOCs.ilc[-1]<MidLine.iloc[-1]:
                    LFrame_vpPOC_short=1
                elif 'SOL'in self.symbol and LFrame_vpPOCs.iloc[-1]>=MidLine.iloc[-1]:
                    LFrame_vpPOC_short=0
                elif 'SOL' not in self.symbol:
                    LFrame_vpPOC_short=1
                if 'SOL'in self.symbol and LFrame_vpPOCs.iloc[-1]>MidLine.iloc[-1]:
                     LFrame_vpPOC_long=1
                elif 'SOL'in self.symbol and LFrame_vpPOCs.iloc[-1]<=MidLine.iloc[-1]:
                     LFrame_vpPOC_long=0
                elif 'SOL' not in self.symbol:
                     LFrame_vpPOC_long=1
                if self.asset_normal==1 and LFrame_vpPOC_short==1 and self.upl_short_open==1 and ohcl5.iloc[-1]/min(ohcl5.iloc[-23:])<1.06 \
                        and\
                            ((len(buy_total_short)<2 and ohcl5.iloc[-1]>UpLine.iloc[-1] ) \
                              or \
                                (len(buy_total_short)>=2 and len(buy_total_short)<=3 and time.time()-buy_total_short.iloc[-1,5]>3600 \
                                 and ohcl5.iloc[-1]>UpLine.iloc[-1] and ohcl5.iloc[-1]/buy_total_short.iloc[-1,2]>1.03)):
                    #开空
                    try:  
                        if 1:
                          usdt_total=self.get_usdt_total()
                          model='buyshort'
                          price0=ohcl5.iloc[-1]
                          price=price0*0.998
                          fv=self.fv[self.symbol]
                          amount=max(int(round(usdt_total/self.asset_coe/price0/fv)),1)
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
                # if 'DOGE' in self.symbol:
                #     print(self.asset_normal==1 , LFrame_vpPOC_long==1 ,ohcl5.iloc[-1]/max(ohcl5.iloc[-23:])>0.94 ,(len(buy_total_long)<2 and ohcl5.iloc[-1]<DownLine.iloc[-1] ) , len(buy_total_long)>=2 , len(buy_total_long)<=3 ,time.time()-buy_total_long.iloc[-1,5]>3600 , ohcl5.iloc[-1]<DownLine.iloc[-1] , ohcl5.iloc[-1]/buy_total_long.iloc[-1,2]<0.97,ohcl5.iloc[-1],buy_total_long.iloc[-1,2],buy_total_long.iloc[-2,2])
                #     time.sleep(100)
                print('self.upl_long_open==1',self.upl_long_open==1,'self.upl_short_open==1',self.upl_short_open==1,'003'*500)
                if self.asset_normal==1 and LFrame_vpPOC_long==1 and self.upl_long_open==1 and ohcl5.iloc[-1]/max(ohcl5.iloc[-23:])>0.94 and ((len(buy_total_long)<2 and ohcl5.iloc[-1]<DownLine.iloc[-1] ) or (len(buy_total_long)>=2 and len(buy_total_long)<=3 and time.time()-buy_total_long.iloc[-1,5]>3600 and ohcl5.iloc[-1]<DownLine.iloc[-1] and ohcl5.iloc[-1]/buy_total_long.iloc[-1,2]<0.97)):
                    #开多
                    try:  
                        if 1:
                          usdt_total=self.get_usdt_total()
                          model='buylong'
                          price0=ohcl5.iloc[-1]
                          price=price0*1.002
                          fv=self.fv[self.symbol]
                          amount=max(int(round(usdt_total/self.asset_coe/price0/fv)),1)
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
                if len(buy_total_long)>=2 or len(buy_total_short)>=2 or time.time()-self.position_time>10:
                    self.position_time=time.time()
                    usdt_total=self.get_usdt_total()
                    notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict,upl_long_dict,upl_short_dict=self.get_entryprice_okx()
                    print(notionalUsd_dict, markprice_dict , positionAmount_dict_sub, upl_dict)
                    print('002'*200,upl_long_dict,upl_short_dict,sum(upl_long_dict.values()),sum(upl_short_dict.values()))

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
                         stop_profit=0.005
                    else:
                        stop_profit=0.005
                    if amount_short>0 and upl_short/notionalUsd_short>stop_profit :
                        model='sellshort'
                        price0=ohcl5.iloc[-1]
                        price=price0*1.002
                        amount=amount_short
                        symbol=self.symbol
                        place_order=self.create_order1(symbol,price,amount,model)
                        print(place_order,symbol,model)
                        buy_total = pd.DataFrame(np.zeros([1,6])) 
                        buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                        pk1 = open(root+'/buy_total_short_%s_%s.spydata'%(self.symbol,self.user),'wb')
                        pickle.dump(buy_total,pk1)
                        pk1.close()
                    if amount_long>0 and upl_long/notionalUsd_long>stop_profit :
                        model='selllong'
                        price0=ohcl5.iloc[-1]
                        price=price0*0.998
                        amount=amount_long
                        symbol=self.symbol
                        place_order=self.create_order1(symbol,price,amount,model)
                        print(place_order,symbol,model)
                        buy_total = pd.DataFrame(np.zeros([1,6])) 
                        buy_total.iloc[0,:] = '下单时间', '买入方式', '开仓价格', '开仓数量','开仓金额','时间戳'
                        pk1 = open(root+'/buy_total_long_%s_%s.spydata'%(self.symbol,self.user),'wb')
                        pickle.dump(buy_total,pk1)
                        pk1.close()
                
                    if ((upl_long+ upl_short)/usdt_total<-0.03) or self.asset_normal==0 :
                        try:  
                            if self.asset_normal==0:
                               logging.info((self.user,'当日回撤过大所有仓位止损'))
                            if amount_short>0 :
                              model='sellshort'
                              price=price_short*1.003
                              amount=amount_short
                              print('001',symbol,model)
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
                              print('001',symbol,model)
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
    
    def create_order1(self,symbol,price,amount,model,tag="7ec66e652c1bBCDE"):#bef23d76c2f8SUDE model:buylong ,buyshort ,selllong ,sellshort ,buycash ,sellcash
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
                time.sleep(1)
                continue
         return totalEq 
   
    def cancel_order(self):
        try:
            lock.acquire()
            unfill=self.tradeAPI.get_order_list(instId='')['data']
            time.sleep(0.5)
            print('订单请求')
        except:
            time.sleep(0.5)
        lock.release()
        if len(unfill)>0:
            for i in range(len(unfill)):
                instid=unfill[i]['instId']
                orderid=unfill[i]['ordId']
                print('撤单开始')
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
    aa4=trade_coin('DOGE-USDT-SWAP','yyyyy2_okx',1500)
    aa5=trade_coin('TRUMP-USDT-SWAP','yyyyy2_okx',1500)
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
            t4 = threading.Thread(target = aa4.trade1) 
            threads.append(t4)
            t5 = threading.Thread(target = aa5.trade1) 
            threads.append(t5)
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
   