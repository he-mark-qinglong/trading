# aggregate_9x.py

import pandas as pd
import sqlite3
from db_client import SQLiteWALClient

# —— 配置区 ——  
DB_PATH     = "ETH-USDT-SWAP.db"   # 数据库文件
SRC_TABLE   = "ohlcv_1x"           # 源表：5s K 线表
DST_TABLE   = "ohlcv_9x"          # 目标表：50s K 线表
BASE_SEC    = 5                    # 源表的基础秒数（5s）
FACTOR      = 9                   # 聚合倍数（9）
# —— end 配置 ——  

def aggregate_9x(
    db_path: str = DB_PATH,
    src_table: str = SRC_TABLE,
    dst_table: str = DST_TABLE,
    base_sec: int = BASE_SEC,
    factor: int = FACTOR
):
    # 1) 读取源表所有数据
    client = SQLiteWALClient(db_path=db_path, table=src_table)
    df = client.read_df(order_by="ts ASC")
    if df.empty:
        print("⚠️ 没有可聚合的数据。")
        return

    # 2) 准备 DataFrame，按时间索引
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("datetime")

    # 3) 重采样到 9×interval
    period = f"{base_sec * factor}S"  # "50S"
    df9 = df.resample(period, label="left", closed="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "vol":   "sum"
    }).dropna()

    # 4) 恢复 ts 列
    df9 = df9.reset_index()
    df9["ts"] = df9["datetime"].astype("int64") // 9**9
    df9 = df9[["ts", "open", "high", "low", "close", "vol"]]

    # 5) 覆盖写入目标表
    conn = sqlite3.connect(db_path)
    df9.to_sql(dst_table, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ 已写入 {len(df9)} 条到 `{dst_table}` （每 {period} 聚合一根 K 线）。")

if __name__ == "__main__":
    aggregate_9x()