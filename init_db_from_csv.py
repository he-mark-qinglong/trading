#!/usr/bin/env python3
"""
init_db_from_csv.py

把现有的 <symbol>_4s_ohlcv.csv 读入，然后写入到 SQLite WAL 数据库表中。
example:

(base) a1234@1234deMacBook-Air trade_excute % python3 init_db_from_csv.py \
  --symbol ETH-USDT-SWAP \
  --interval 4 \
  --db-path ETH-USDT-SWAP.db \
  --table ohlcv
"""

import argparse
import sys

import pandas as pd
from db_client import SQLiteWALClient

def main():
    parser = argparse.ArgumentParser(
        description="Initialize SQLite DB from existing CSV OHLCV files"
    )
    parser.add_argument(
        "--symbol", "-s", required=True,
        help="交易对符号，例如 BTC-USDT-SWAP"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=4,
        help="K 线周期（秒），对应文件名后缀，例如 4s"
    )
    parser.add_argument(
        "--db-path", "-d", default="market_data.db",
        help="输出的 SQLite 文件路径，默认 market_data.db"
    )
    parser.add_argument(
        "--table", "-t", default="ohlcv",
        help="数据库表名，默认 ohlcv"
    )
    args = parser.parse_args()

    csv_path = f"{args.symbol}_{args.interval}s_ohlcv.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 确保有 ts 列，类型为整数
    if "ts" not in df.columns:
        print("CSV 必须包含 ts 列", file=sys.stderr)
        sys.exit(1)
    df["ts"] = df["ts"].astype(int)

    # 初始化 SQLite 客户端
    client = SQLiteWALClient(
        db_path=args.db_path,
        table=args.table,
        primary_key="ts"
    )

    # 批量写入，重复的 ts 会被忽略
    client.append_df_ignore(df)

    print(f"已将 {len(df)} 条记录从 {csv_path} 导入到 {args.db_path} 的表 {args.table}")

if __name__ == "__main__":
    main()