# aggregate_4x.py

import pandas as pd
import sqlite3
from db_client import SQLiteWALClient

# —— 配置区 ——  
DB_PATH     = "ETH-USDT-SWAP.db"   # 或者你的 DB 路径
SRC_TABLE   = "ohlcv"             # 1×interval 表名
DST_TABLE   = "ohlcv_4x"          # 要写入的 4×interval 表名
BASE_SEC    = 4                   # 1× 的秒数
FACTOR      = 4                   # 聚合倍数
# —— end 配置 ——  

def aggregate_4x(
    db_path: str = DB_PATH,
    src_table: str = SRC_TABLE,
    dst_table: str = DST_TABLE,
    base_sec: int = BASE_SEC,
    factor: int = FACTOR
):
    # 1) 读源表
    client = SQLiteWALClient(db_path=db_path, table=src_table)
    df = client.read_df(order_by="ts ASC")
    if df.empty:
        print("⚠️ 没有可聚合的数据。")
        return

    # 2) 准备 DataFrame
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("datetime")

    # 3) 重采样
    period = f"{base_sec * factor}S"  # e.g. "16S"
    df4 = df.resample(period, label="left", closed="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "vol":   "sum"
    }).dropna()

    # 4) 恢复 ts 列
    df4 = df4.reset_index()
    df4["ts"] = df4["datetime"].astype("int64") // 10**9
    df4 = df4[["ts", "open", "high", "low", "close", "vol"]]

    # 5) 覆盖写入目标表
    conn = sqlite3.connect(db_path)
    df4.to_sql(dst_table, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ 已写入 {len(df4)} 条到 `{dst_table}`。")

if __name__ == "__main__":
    aggregate_4x()