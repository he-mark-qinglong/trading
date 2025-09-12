import ccxt  
import pandas as pd  
from typing import Dict, List  
import time  
from datetime import datetime , timedelta
import os  


class __HistoricalDataLoader:  
    def __init__(self, exchange_id: str = 'okx'):  
        """  
        初始化数据加载器  
        :param exchange_id: 交易所ID  
        """  
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({  
            'enableRateLimit': True,  
            'options': {  
                'defaultType': 'swap',  
                'adjustForTimeDifference': True,  
            },  
            'proxies': {  
                'http': 'http://127.0.0.1:7890',  # clash 默认 HTTP 代理端口  
                'https': 'http://127.0.0.1:7890'  # clash 默认 HTTPS 代理端口  
            }  
        })  

        self.timeframes = {  
            '1m': {'limit': 1000, 'duration': 60},  
            '3m': {'limit': 1000, 'duration': 180},  
            '5m': {'limit': 1000, 'duration': 300},  
            '15m': {'limit': 1000, 'duration': 900},  
            '30m': {'limit': 1000, 'duration': 1800},  
            '1h': {'limit': 1000, 'duration': 3600},  
            '4h': {'limit': 1000, 'duration': 14400},  
            '1d': {'limit': 1000, 'duration': 86400},  
        }  

    

import time
import math
from datetime import datetime, timedelta
import pandas as pd
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class HistoricalDataLoader:
    """
    功能：
      - 读取本地数据（返回 DataFrame，及最早/最晚时间）
      - 向前补历史（从 given_earliest 向更久远抓取指定数量）
      - 向后补最新（从 given_latest 向现在抓取直到最近或达到数量）
      - 合并并保存
    依赖：
      - self.exchange.fetch_ohlcv(symbol, timeframe, since=ms, limit=n)
      - data_manager.load_data(exchange_id, symbol, timeframe) -> pd.DataFrame or None
      - data_manager.save_data(exchange_id, symbol, timeframe, df)
    """

    # timeframe -> seconds 映射
    TIMEFRAME_SECONDS = {
        '1m': 60,
        '3m': 3 * 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1h': 60 * 60,
        '4h': 4 * 60 * 60,
        '1d': 24 * 60 * 60,
    }
    def __init__(self, exchange_id: str = 'okx', rate_limit_sleep_factor: float = 1.0):  
        """  
        初始化数据加载器  
        :param exchange_id: 交易所ID  
        """  
        self.exchange_id = exchange_id
        self.rate_limit_sleep_factor = rate_limit_sleep_factor

        self.exchange = getattr(ccxt, exchange_id)({  
            'enableRateLimit': True,  
            'options': {  
                'defaultType': 'swap',  
                'adjustForTimeDifference': True,  
            },  
            'proxies': {  
                'http': 'http://127.0.0.1:7890',  # clash 默认 HTTP 代理端口  
                'https': 'http://127.0.0.1:7890'  # clash 默认 HTTPS 代理端口  
            }  
        })  

        self.timeframes = {  
            '1m': {'limit': 1000, 'duration': 60},  
            '3m': {'limit': 1000, 'duration': 180},  
            '5m': {'limit': 1000, 'duration': 300},  
            '15m': {'limit': 1000, 'duration': 900},  
            '30m': {'limit': 1000, 'duration': 1800},  
            '1h': {'limit': 1000, 'duration': 3600},  
            '4h': {'limit': 1000, 'duration': 14400},  
            '1d': {'limit': 1000, 'duration': 86400},  
        }  

    def format_symbol(self, symbol: str) -> str:  
        """格式化交易对符号以适应OKX永续合约市场"""  
        if self.exchange_id == 'binance':
            # 移除 '/', '-'，并将 'USDT' 放在最后
            formatted_symbol = symbol.replace('/', '').replace('-', '').replace('SWAP', '').replace('USDT', '') + 'USDT'
            return formatted_symbol.upper()  # 转换为大写 (可选)
        if '-SWAP' not in symbol:  
            base = symbol.replace('USDT', '').replace('/', '')  
            return f"{base}-USDT-SWAP"  
        return symbol  

    # -------------------------
    # 读取本地数据并返回最早/最晚时间
    # -------------------------
    def load_local_data(self, symbol: str, timeframe: str, data_manager) -> (pd.DataFrame, int, int):
        """
        从 data_manager 加载数据，返回 (df, earliest_ts, latest_ts)
        - df: pd.DataFrame，index 为 datetime（升序）；若无数据返回空 DataFrame
        - earliest_ts, latest_ts: unix 秒（int），若没有数据返回 (None, None)
        """
        df = data_manager.load_data(self.exchange_id, symbol, timeframe)
        if df is None or df.empty:
            return pd.DataFrame(), None, None

        # 确保 index 为 DatetimeIndex 并升序
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df['timestamp'], unit='ms')
                df.drop(columns=['timestamp'], inplace=True, errors='ignore')
            except Exception:
                # 假设存储的 index 就是时间索引；若不行则返回空
                logger.warning("Loaded data index is not datetime and 'timestamp' column not found/convertible.")
                return pd.DataFrame(), None, None

        df = df.sort_index()
        earliest_ts = int(df.index[0].timestamp())
        latest_ts = int(df.index[-1].timestamp())
        return df, earliest_ts, latest_ts

    # -------------------------
    # 通用 fetch 单次请求（包含重试、退避）
    # -------------------------
    def _fetch_with_retries(self, formatted_symbol: str, timeframe: str, since_seconds: int, limit_per_call: int,
                            max_retries: int = 5, backoff_base: float = 0.5):
        """
        fetch_ohlcv with retries and exponential backoff.
        - since_seconds: unix seconds (int) or None
        - returns list of ohlcv or [] on unrecoverable failure
        """
        attempt = 0
        while attempt <= max_retries:
            try:
                since_ms = None if since_seconds is None else int(since_seconds * 1000)
                ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, timeframe=timeframe, since=since_ms, limit=limit_per_call)
                # ensure list
                return ohlcv or []
            except Exception as e:
                attempt += 1
                wait = min(10, backoff_base * (2 ** (attempt - 1)))  # cap backoff to 10s
                extra = random.uniform(0, 0.1 * wait)
                logger.warning(f"fetch_ohlcv error{e} (attempt {attempt}/{max_retries}) for {formatted_symbol} {timeframe} since={since_seconds}: {e}. Backoff {wait+extra:.2f}s")
                time.sleep((wait + extra) * self.rate_limit_sleep_factor)
        logger.error(f"Failed fetch_ohlcv after {max_retries} attempts for {formatted_symbol} {timeframe} since={since_seconds}")
        return []

    # -------------------------
    # 向后补最新数据（从 latest_ts 向现在抓取直到现在或达到 limit）
    # -------------------------
    def fetch_forward_to_now(self, symbol: str, timeframe: str, data_manager, limit: int = 2000,
                             limit_per_call: int = 500, max_retries: int = 5, local_only: bool = False) -> pd.DataFrame:
        """
        从本地 latest_ts 向后抓取，直到接近当前时间或累计到 limit 条。
        返回合并后的数据并保存本地。
        - local_only: True 时仅返回本地数据，不联网
        """
        formatted_symbol = self.format_symbol(symbol)
        local_df, earliest_ts, latest_ts = self.load_local_data(symbol, timeframe, data_manager)
        if local_only:
            return local_df.iloc[-min(len(local_df), limit):] if not local_df.empty else pd.DataFrame()

        tf_seconds = self.TIMEFRAME_SECONDS.get(timeframe)
        if tf_seconds is None:
            raise ValueError(f"Unsupported timeframe {timeframe}")

        # 如果没有本地数据，从 limit 条之前的时间点开始抓取
        end_ts_now = int(time.time())
        if latest_ts is None:
            # start from now - limit * timeframe
            since = max(0, end_ts_now - limit * tf_seconds)
        else:
            # start from the next candle after latest_ts
            since = latest_ts + tf_seconds

        all_ohlcv = []
        while since * 1000 <= end_ts_now * 1000 and len(all_ohlcv) < limit:
            # fetch
            ohlcv = self._fetch_with_retries(formatted_symbol, timeframe, since, min(limit_per_call, limit - len(all_ohlcv)), max_retries=max_retries)
            if not ohlcv:
                # nothing fetched => break (or retry handled inside)
                break
            # ccxt 返回升序，[oldest ... newest]
            all_ohlcv.extend(ohlcv)
            # advance since to last returned timestamp + 1 timeframe
            last_ts_ms = ohlcv[-1][0]
            since = int(last_ts_ms / 1000) + tf_seconds

            # 遵守 rate limit
            if hasattr(self.exchange, 'rateLimit') and self.exchange.rateLimit:
                # ccxt rateLimit 通常是毫秒
                time.sleep(max(0.001, (self.exchange.rateLimit / 1000.0) * 0.1) * self.rate_limit_sleep_factor)

        # build DataFrame
        if all_ohlcv:
            new_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)
            new_df = new_df[~new_df.index.duplicated(keep='last')]
            new_df.sort_index(inplace=True)

            # merge: existing_data first, new_data second -> keep last (new data overwrites any duplicates)
            if not local_df.empty:
                combined = pd.concat([local_df, new_df])
            else:
                combined = new_df

            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            # save
            data_manager.save_data(self.exchange_id, symbol, timeframe, combined)
            return combined.iloc[-min(len(combined), limit):]
        else:
            # nothing new fetched, return local
            return local_df.iloc[-min(len(local_df), limit):] if not local_df.empty else pd.DataFrame()

    # -------------------------
    # 向前补历史数据（从 earliest_ts 向过去抓取指定数量）
    # -------------------------
    def fetch_backward_history(  
        self,  
        symbol: str,  
        timeframe: str,  
        data_manager,  
        limit=365*24*12,
        local_only = False
    ) -> pd.DataFrame:  
        """  
        自动加载并补充历史数据到最新，检查并补全缺失数据。  
        :param symbol: 交易对  
        :param timeframe: 时间框架  
        :param data_manager: 数据管理器实例  
        :return: 补充后的完整数据  
        """  
        try:  
            formatted_symbol = self.format_symbol(symbol)  
            print('after format', formatted_symbol)
            # 加载本地数据  
            existing_data = data_manager.load_data(self.exchange_id, symbol, timeframe)  
            if local_only:
                return existing_data
            
            if existing_data is None:
                exist_len = 0
            else:
                exist_len =  len(existing_data) 
                
            # print(f'symbol:{symbol} timeframe:{timeframe} existing_data:{existing_data.head}')
            # 确定拉取起点  
            if existing_data is not None and not existing_data.empty:  
                first_timestamp = int(existing_data.index[0].timestamp()) + 1  
            else:  
                first_timestamp = None  

            # 当前时间作为终点  
            end_timestamp = int(time.time())  

            # 增量拉取数据  
            all_data = []  
            from_timestamp = first_timestamp   
            
            def get_timestamp_before_minutes(current_time, num_of_min: int) -> int:  
                """获取当前时间往后推移指定数量的5分钟的时间戳（毫秒）。  
                
                :param num_of_five_min: 要推移的5分钟的数量  
                :return: 推移后的时间戳（毫秒）  
                """  
                # 计算推移后的时间  
                after_time = current_time - timedelta(minutes=num_of_min)  
                # 转换为时间戳（毫秒）  
                timestamp_after_time = int(after_time.timestamp())  
                return timestamp_after_time  
            
            current_date = datetime.now()
            # current_date = datetime.fromtimestamp(current_time/1000)
            print(f'now date:{current_date}')

            # from_timestamp = get_timestamp_before_minutes(current_date, limit*(5 if timeframe == '5m' else 15)) 
            if from_timestamp == None:
                from_timestamp = get_timestamp_before_minutes(current_date, 100) 
            
            retry_time = 0

            while from_timestamp is None or from_timestamp < end_timestamp:  
                current_date = datetime.fromtimestamp(from_timestamp)
                to_date = datetime.fromtimestamp(end_timestamp)
                # print(f'going to get from date:{current_date} to {to_date}')
                try:
                    ohlcv = self.exchange.fetch_ohlcv(  
                        formatted_symbol,  
                        timeframe=timeframe,  
                        since=from_timestamp * 1000,  
                        limit=100  
                    )  

                    if not ohlcv:
                        if retry_time < 100:
                            retry_time += 1  
                            print(f'retrying time:{retry_time}')
                            continue
                        else:
                            print(f'retry times consumed {retry_time}')
                            break
                except Exception as e:
                    if retry_time < 100:
                        retry_time += 1  
                        continue
                    else:
                        print(f'retry timeout {retry_time}')
                        break
                if len(all_data) > 0:
                    #print(f'timeframe {timeframe}:{all_data[-1][0]/(60 * 1000)} < {ohlcv[0][0]/(60 * 1000)}')
                    a = datetime.fromtimestamp(all_data[0][0]/1000)
                    b = datetime.fromtimestamp(ohlcv[-1][0]/1000)
                    c = datetime.fromtimestamp(ohlcv[0][0]/1000)
                    print(f'datetime {timeframe}:{c} < {b} < {a}, got count:{len(ohlcv)} pct:{round((len(all_data) + exist_len)/limit, 2)} limit_all={limit}')
                    
                all_data = ohlcv + all_data    
                # |-----all_data-----|-----ohlcv 1500 len-----|---------|current
                multipliers = {'1m':1, '3m':3, '5m':5, '15m':15, '1h': 60, '4h':240, '1d':1440}
                # len_of_minute = multipliers[timeframe] * (limit-len(all_data))
                # from_timestamp = get_timestamp_before_minutes(current_date, len_of_minute)
                old_date = datetime.fromtimestamp(from_timestamp)
                from_timestamp = get_timestamp_before_minutes(old_date, 100)
                
                if len(all_data) + exist_len >= limit:
                    print('required data length matched')
                    break 
                # 避免触发频率限制  
                time.sleep(self.exchange.rateLimit / 5000)  

            # 如果有新数据，创建 DataFrame  
            if all_data:  
                new_data = pd.DataFrame(  
                    all_data,  
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']  
                )  
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')  
                new_data.set_index('timestamp', inplace=True)  

                # 合并新数据和本地数据  
                if existing_data is not None and not existing_data.empty:  
                    combined_data = pd.concat([new_data, existing_data])  
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  
                    combined_data.sort_index(inplace=True)  
                else:  
                    combined_data = new_data  
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  
                    combined_data.sort_index(inplace=True)  

                
                # 保存到本地文件  
                data_manager.save_data(self.exchange_id, symbol, timeframe, combined_data)  
                print(f"Updated data saved for {symbol} {timeframe}, total rows: {len(combined_data)}")  
                return combined_data.iloc[-min(len(combined_data), limit):]  

            # 如果没有新数据，返回现有数据  
            #print(f"No new data fetched for {symbol} {timeframe}. Returning existing data.")  
            return existing_data.iloc[-min(exist_len, limit):] if existing_data is not None else pd.DataFrame()  

        except Exception as e:  
            print(f"Error fetching data for {symbol} {timeframe}: {str(e)}")  
            return pd.DataFrame()  

class DataManager:  
    """数据管理类，用于处理和存储历史数据"""  
    def __init__(self, base_path: str = './data'):  
        self.base_path = base_path  
        os.makedirs(base_path, exist_ok=True)  

    def save_data(self, exchange_id, symbol, timeframe, df):  
        safe_symbol = symbol.replace('/', '_')  
        directory = f"{self.base_path}/{exchange_id}_{safe_symbol}"  
        os.makedirs(directory, exist_ok=True)  

        filename = f"{directory}/{timeframe}.parquet"  
        df.to_parquet(filename)  
        print(f"Data saved to {filename}")  

    def load_data(self, exchange_id, symbol, timeframe):  
        safe_symbol = symbol.replace('/', '_')  
        filename = f"{self.base_path}/{exchange_id}_{safe_symbol}/{timeframe}.parquet"  
        print('load file:', filename)
        if os.path.exists(filename):  
            return pd.read_parquet(filename)  
        return None  


def read_and_sort_df(client=None, LIMIT_K_N=None):
    data_manager = DataManager('./data')  
    
    loader = HistoricalDataLoader('binance')  
    # loader = HistoricalDataLoader('okx')  

    symbol = "BTC-USDT-SWAP"
    symbol = "ETH-USDT-SWAP"
    # symbol = "XRP-USDT-SWAP"
    # symbol = "SOL-USDT-SWAP"
    timeframe = '5m'
    # timeframe = '1h'
    df = None
    for i in range(10):
        # df = loader.fetch_backward_history(symbol, timeframe, data_manager, 5 * 700_000,
        #                             #    local_only=True
        #                                )  
        
        df = loader.fetch_forward_to_now(symbol, timeframe, data_manager, 5 * 700_000)  
        if df is None or df.empty:
            time.sleep(10)
            continue
        
        df['vol'] = df['volume']
        df['datetime'] = df.index
        break
    # df = df.iloc[-120_000:]
    # df = df.iloc[-480_000:-120_000]
    # df = df.iloc[-480_000:]
    print(f"Fetched and updated {timeframe} data for {symbol}, total rows: {len(df)}")  
    print(df.index[0], df.index[-1])
    print(df.tail(20))
    return  df

if __name__ == "__main__":  
    # main()
    read_and_sort_df()
