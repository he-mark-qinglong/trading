import pandas as pd
import os
class TradeLogManager:
    """交易记录管理类，用于处理和存储策略交易日志"""

    def __init__(self, base_path: str = './trade_logs'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_trade_log(self, exchange_id: str, symbol: str, timeframe: str, trade_log: list):
        """
        保存交易日志列表（字典列表）为 parquet 文件
        :param exchange_id: 交易所标识，如 'binance'
        :param symbol: 交易对，如 'ETH/USDT'
        :param timeframe: 时间周期，如 '5min'
        :param trade_log: 交易日志列表，列表内元素为字典
        """
        safe_symbol = symbol.replace('/', '_')
        directory = f"{self.base_path}/{exchange_id}_{safe_symbol}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{timeframe}_trade_log.parquet"

        df = pd.DataFrame(trade_log)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        df.to_parquet(filename)
        print(f"Trade log saved to {filename}")

    def load_trade_log(self, exchange_id: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        读取交易日志 parquet 文件，返回 DataFrame 或 None
        """
        safe_symbol = symbol.replace('/', '_')
        filename = f"{self.base_path}/{exchange_id}_{safe_symbol}/{timeframe}_trade_log.parquet"
        print(f"Loading trade log from {filename}")
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            return df
        else:
            print("Trade log file not found.")
            return None