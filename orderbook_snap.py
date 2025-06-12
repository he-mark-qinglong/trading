# collect_depth.py

import asyncio
import json
import time

import pandas as pd
import websockets

from db_client import OrderbookWALClient, drop_orderbook_snap

# —— 配置区 —— 
symbol       = "ETH-USDT-SWAP"
WS_URL       = "wss://ws.okx.com:8443/ws/v5/public"
INTERVAL     = 1         # 采样间隔：每 1 秒
TOP_N        = 10         # 前 5 档
MAX_ROWS     = 10000     # 表中最多保留行数
DB_PATH      = f"{symbol}.db"
TABLE        = "orderbook_snap"

# drop_orderbook_snap(DB_PATH)
client_book = OrderbookWALClient(db_path=DB_PATH, table=TABLE, top_n=TOP_N)


# 打印 CREATE TABLE 语句
conn = client_book._connect()
with conn:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (TABLE,)
    ).fetchone()
    print("=== CREATE TABLE ===")
    print(row[0], "\n")

    # 打印每一列的详细信息
    print("=== PRAGMA table_info ===")
    cols = conn.execute(f"PRAGMA table_info({TABLE});").fetchall()
    # PRAGMA table_info 返回 (cid, name, type, notnull, dflt_value, pk)
    for cid, name, col_type, notnull, dflt, pk in cols:
        print(f"{cid:>2} | {name:<10} | {col_type:<6} | notnull={notnull} | pk={pk}")
conn.close()

import numpy as np
def compute_obpi(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    给定一个包含 bid1..bidN, ask1..askN 的 df，
    计算 sum_bid, sum_ask 和 OBPI=(sum_bid - sum_ask)/(sum_bid+sum_ask)。
    """
    bid_cols = [f"bid{i}" for i in range(1, top_n+1)]
    ask_cols = [f"ask{i}" for i in range(1, top_n+1)]
    # 累加买卖量
    df["sum_bid"] = df[bid_cols].sum(axis=1)
    df["sum_ask"] = df[ask_cols].sum(axis=1)
    # 防止除零
    denom = df["sum_bid"] + df["sum_ask"]
    df["obpi"] = np.where(denom>0,
                          (df["sum_bid"] - df["sum_ask"]) / denom,
                          0.0)
    return df

# 2) 本地盘口状态
bids_map = {}  # price(float) -> size(float)
asks_map = {}

async def _collect_books_once():
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        # 订阅 books5
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel":  "books",
                "instType": "SWAP",
                "instId":   symbol
            }]
        }))
        print(f"[{symbol}] subscribed to books")

        last_ts = 0
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("event") in ("subscribe","cancel","error"):
                continue
            if msg.get("arg",{}).get("channel")!="books":
                continue

            rec    = msg["data"][0]
            action = msg.get("action")   # snapshot | update
            # 3) 全量加载 or 增量更新
            if action == "snapshot":
                bids_map.clear()
                asks_map.clear()
                for p, s in rec["bids"]:
                    bids_map[float(p)] = float(s)
                for p, s in rec["asks"]:
                    asks_map[float(p)] = float(s)

            elif action == "update":
                # bids
                for p, s in rec["bids"]:
                    price = float(p); size = float(s)
                    if size == 0:
                        bids_map.pop(price, None)
                    else:
                        bids_map[price] = size
                # asks
                for p, s in rec["asks"]:
                    price = float(p); size = float(s)
                    if size == 0:
                        asks_map.pop(price, None)
                    else:
                        asks_map[price] = size

            # 4) 到了采样时刻再去算 TopN + OBPI
            now = int(time.time())
            if now - last_ts < INTERVAL:
                continue
            last_ts = now

            # sort 并切 TopN
            top_bids = sorted(bids_map.items(), key=lambda x: x[0], reverse=True)[:TOP_N]
            top_asks = sorted(asks_map.items(), key=lambda x: x[0])[:TOP_N]

            sum_b = sum(sz for _, sz in top_bids)
            sum_a = sum(sz for _, sz in top_asks)
            obpi  = (sum_b - sum_a) / (sum_b + sum_a) if (sum_b+sum_a)>0 else 0.0

            # 构造写库的 row
            row = {"ts": now}
            for i, (_, sz) in enumerate(top_bids,  start=1):
                row[f"bid{i}"] = sz
            for i, (_, sz) in enumerate(top_asks,  start=1):
                row[f"ask{i}"] = sz
            row["sum_bid"] = round(sum_b, 4)
            row["sum_ask"] = round(sum_a, 4)
            row["obpi"]    = round(obpi,   4)

            # 落库 & 裁剪
            client_book.append_df_ignore(pd.DataFrame([row]))
            # … 同你原来那段裁剪逻辑 …

            print(f"[Books @ {now}] Top{TOP_N} sum_bid={sum_b:.0f} sum_ask={sum_a:.0f} obpi={obpi:.4f}")

async def collect_depth():
    while True:
        try:
            await _collect_books_once()
        except (websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError):
            print("Websocket closed, retry in 2s…")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unexpected error: {e!r}, retry in 5s…")
            await asyncio.sleep(5)

if __name__ == "__main__":
    print("Start collecting orderbook (books5)…")
    asyncio.run(collect_depth())