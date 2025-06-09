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
TOP_N        = 5         # 前 5 档
MAX_ROWS     = 10000     # 表中最多保留行数
DB_PATH      = f"{symbol}.db"
TABLE        = "orderbook_snap"

# drop_orderbook_snap(DB_PATH)
client_book = OrderbookWALClient(db_path=DB_PATH, table=TABLE)


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

async def _collect_books_once():
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        # 订阅 books5
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel":  "books5",
                "instType": "SWAP",
                "instId":   symbol
            }]
        }))
        print(f"[{symbol}] subscribed to books5")

        last_ts = 0
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("No depth data for 30s, reconnecting…")
                break

            data = json.loads(raw)

            # 过滤订阅确认、心跳包
            if data.get("event") in ("subscribe", "cancel", "error"):
                continue
            if data.get("arg", {}).get("channel") != "books5":
                continue

            rec = data["data"][0]
            now = int(time.time())
            if now - last_ts < INTERVAL:
                # 没到采样间隔就不落库
                continue

            # 拿前 TOP_N 档
            bids = rec.get("bids", [])[:TOP_N]
            asks = rec.get("asks", [])[:TOP_N]
            sum_b = sum(float(x[1]) for x in bids)
            sum_a = sum(float(x[1]) for x in asks)
            obpi  = (sum_b - sum_a) / (sum_b + sum_a) if (sum_b + sum_a) > 0 else 0.0

            row = {
                "ts": now,
                **{f"bid{i+1}": float(bids[i][1]) for i in range(len(bids))},
                **{f"ask{i+1}": float(asks[i][1]) for i in range(len(asks))},
                "sum_bid": round(sum_b, 4),
                "sum_ask": round(sum_a, 4),
                "obpi":    round(obpi, 4)
            }
            df = pd.DataFrame([row])
            client_book.append_df_ignore(df)
            print(f"[Books5 @ {now}] sum_bid={sum_b:.2f}, sum_ask={sum_a:.2f}, obpi={obpi:.4f}")

            # 裁剪旧数据，只保留最新 MAX_ROWS 行
            conn = client_book._connect()
            with conn:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
                if cnt > MAX_ROWS:
                    to_delete = cnt - MAX_ROWS
                    conn.execute(f"""
                        DELETE FROM {TABLE}
                         WHERE ts IN (
                           SELECT ts FROM {TABLE}
                           ORDER BY ts ASC
                           LIMIT ?
                         )
                    """, (to_delete,))
            conn.close()

            last_ts = now

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