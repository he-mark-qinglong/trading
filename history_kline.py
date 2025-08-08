import ccxt  
import pandas as pd  
from typing import Dict, List  
import time  
from datetime import datetime , timedelta
import os  


class HistoricalDataLoader:  
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

    def fetch_historical_data(  
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

            # 加载本地数据  
            existing_data = data_manager.load_data(self.exchange_id, symbol, timeframe)  
            if local_only:
                return existing_data
            
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
                            continue
                        else:
                            break
                except Exception as e:
                    if retry_time < 10:
                        retry_time += 1  
                        continue
                    else:
                        break
                if len(all_data) > 0:
                    #print(f'timeframe {timeframe}:{all_data[-1][0]/(60 * 1000)} < {ohlcv[0][0]/(60 * 1000)}')
                    a = datetime.fromtimestamp(all_data[0][0]/1000)
                    b = datetime.fromtimestamp(ohlcv[-1][0]/1000)
                    c = datetime.fromtimestamp(ohlcv[0][0]/1000)
                    print(f'datetime {timeframe}:{c} < {b} < {a}, got count:{len(ohlcv)} pct:{round((len(all_data) + len(existing_data))/limit, 2)}')
                    print()
                all_data = ohlcv + all_data    
                # |-----all_data-----|-----ohlcv 1500 len-----|---------|current
                multipliers = {'1m':1, '3m':3, '5m':5, '15m':15, '1h': 60, '4h':240, '1d':1440}
                # len_of_minute = multipliers[timeframe] * (limit-len(all_data))
                # from_timestamp = get_timestamp_before_minutes(current_date, len_of_minute)
                old_date = datetime.fromtimestamp(from_timestamp)
                from_timestamp = get_timestamp_before_minutes(old_date, 100)
                
                if len(all_data) + len(existing_data) >= limit:
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

                # 检查缺失数据  
                missing_data = find_missing_data(combined_data, timeframe)  
                if not missing_data.empty:  
                    print("Filling missing data...")  
                    for _, row in missing_data.iterrows():  
                        start = int(row['start'].timestamp() * 1000)  
                        end = int(row['end'].timestamp() * 1000)  

                        while start < end:  
                            ohlcv = self.exchange.fetch_ohlcv(  
                                formatted_symbol,  
                                timeframe=timeframe,  
                                since=start,  
                                limit=300 #limit  
                            )  
                            if not ohlcv:  
                                break  
                            all_data.extend(ohlcv)  
                            start = ohlcv[-1][0] + 1  
                            time.sleep(self.exchange.rateLimit / 1000)  
                    else:
                        print("No missing data found.")  
                # 保存到本地文件  
                data_manager.save_data(self.exchange_id, symbol, timeframe, combined_data)  
                print(f"Updated data saved for {symbol} {timeframe}, total rows: {len(combined_data)}")  
                return combined_data.iloc[-min(len(combined_data), limit):]  

            # 如果没有新数据，返回现有数据  
            #print(f"No new data fetched for {symbol} {timeframe}. Returning existing data.")  
            return existing_data.iloc[-min(len(existing_data), limit):] if existing_data is not None else pd.DataFrame()  

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


def find_missing_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:  
    """  
    检查数据中是否存在缺失的时间段。  
    :param df: 包含历史数据的 DataFrame，索引为时间戳  
    :param timeframe: 时间框架（如 '1m'、'5m'）  
    :return: 缺失时间段的 DataFrame  
    """  
    if df is None or df.empty:  
        print("DataFrame is empty. No data to check.")  
        return pd.DataFrame()  

    # 时间间隔（秒）  
    timeframe_duration = {  
        '1m': 60,  
        '3m': 180,  
        '5m': 300,  
        '15m': 900,  
        '30m': 1800,  
        '1h': 3600,  
        '4h': 14400,  
        '1d': 86400,  
    }  
    if timeframe not in timeframe_duration:  
        raise ValueError(f"Unsupported timeframe: {timeframe}")  

    # 计算预期的时间间隔  
    expected_interval = pd.Timedelta(seconds=timeframe_duration[timeframe])  

    # 找出实际时间间隔  
    actual_intervals = df.index.to_series().diff()  

    # 找出缺失的时间点（实际间隔大于预期间隔）  
    missing_intervals = actual_intervals[actual_intervals > expected_interval]  

    if missing_intervals.empty:  
        return pd.DataFrame()  

    # 构造缺失时间段的 DataFrame  
    missing_data = pd.DataFrame({  
        'start': missing_intervals.index - missing_intervals,  
        'end': missing_intervals.index  
    })  
    missing_data['duration'] = (missing_data['end'] - missing_data['start']).astype('timedelta64[s]')  

    print(f"Found {len(missing_data)} missing intervals:")  
    print(missing_data)  

    return missing_data  


def main():  
    # 初始化数据加载器  
    data_manager = DataManager('../data')  
    loader = HistoricalDataLoader('binance')  

    # 定义要获取的交易对和时间框架    
    symbols = ['SOL-USDT-SWAP', 'BTC-USDT-SWAP', 'XRP-USDT-SWAP',  ]  
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']  

    symbols = ["BTC-USDT-SWAP"]
    timeframes = ['1m']
    # 获取并保存数据  
    for symbol in symbols:  
        print(f"\nFetching data for {symbol}")  
        for timeframe in timeframes:  
            df = loader.fetch_historical_data(symbol, timeframe, data_manager, 500000)  
            if not df.empty:  
                print(f"Fetched and updated {timeframe} data for {symbol}, total rows: {len(df)}")  

                print(df.head)

                # 定义时间区间  
                start_date = '2024-06-01'  
                end_date = '2024-06-24' 
                df = df.loc[start_date:end_date]  
                rolled = df['volume'].rolling(24)

                mean = rolled.mean()
                std = rolled.std()
                vol_zscore = (df['volume'] - mean) / std

                # 截取该时间区间的数据  
                df_interval = vol_zscore

                for i in range(len(df_interval)):
                    if abs(df_interval.iloc[-i]) > 1.5:
                        print(df_interval.index[-i], df_interval.iloc[-i])

                    if i > 200:
                        break
            else:  
                print(f"No new data for {symbol} {timeframe}")  


def read_and_sort_df(client=None, LIMIT_K_N=None):
    data_manager = DataManager('../data')  
    
    loader = HistoricalDataLoader('binance')  
    loader = HistoricalDataLoader('okx')  

    symbol = "BTC-USDT-SWAP"
    symbol = "ETH-USDT-SWAP"
    # symbol = "XAUT-USDT-SWAP"
    timeframe = '5m'
    # timeframe = '1h'
    df = None
    for i in range(10):
        df = loader.fetch_historical_data(symbol, timeframe, data_manager, 700_000,
                                       local_only=True
                                       )  
        if df.empty:
            time.sleep(10)
            continue
        
        df['vol'] = df['volume']
        df['datetime'] = df.index
        break
    df = df.iloc[-30_000:]
    print(f"Fetched and updated {timeframe} data for {symbol}, total rows: {len(df)}")  
    # print(df.index[0], df.index[-1])
    return  df

if __name__ == "__main__":  
    # main()
    read_and_sort_df()
