
import asyncio
import json
import time

import pandas as pd
import websockets

from db_client import OrderbookWALClient

# —— 配置区 —— 
symbol       = "ETH-USDT-SWAP"
WS_URL       = "wss://ws.okx.com:8443/ws/v5/public"
INTERVAL     = 5         # 采样间隔：每 5 秒
TOP_N        = 10        # 前 N 档
DB_PATH      = f"{symbol}.db"
TABLE        = "orderbook_snap"

client_book = OrderbookWALClient(db_path=DB_PATH, table=TABLE, top_n=TOP_N)

# DataFrame 用于保存 VWAP 指标
df_vwap_metrics = pd.DataFrame(columns=[
    "ts", "bid_vwap", "ask_vwap", "depth_mid_vwap"
])

# 本地盘口状态
bids_map = {}  # price(float) -> size(float)
asks_map = {}

def apply_snapshot(rec):
    bids_map.clear()
    asks_map.clear()
    for p, s, *_ in rec["bids"]:
        bids_map[float(p)] = float(s)
    for p, s, *_ in rec["asks"]:
        asks_map[float(p)] = float(s)

# （可保留或移除，按需）
last_ts = time.time()
bid_eaten = ask_eaten = 0.0
WINDOW = 5.0  # 秒

def apply_update(rec):
    global bid_eaten, ask_eaten, last_ts
    for p, s, *_ in rec["bids"]:
        price, size = float(p), float(s)
        old = bids_map.get(price, 0.0)
        if size < old:
            bid_eaten += (old - size)
        if size == 0:
            bids_map.pop(price, None)
        else:
            bids_map[price] = size
    for p, s, *_ in rec["asks"]:
        price, size = float(p), float(s)
        old = asks_map.get(price, 0.0)
        if size < old:
            ask_eaten += (old - size)
        if size == 0:
            asks_map.pop(price, None)
        else:
            asks_map[price] = size

    now = time.time()
    if now - last_ts >= WINDOW:
        bid_rate = bid_eaten / WINDOW
        ask_rate = ask_eaten / WINDOW
        print(f"[{WINDOW}s] bid_eat_rate={bid_rate:.4f}, ask_eat_rate={ask_rate:.4f} is_bid_eat_stronger={bid_rate > ask_rate}")
        bid_eaten = ask_eaten = 0.0
        last_ts = now

async def _collect_books_once():
    global df_vwap_metrics

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel":  "books",
                "instType": "SWAP",
                "instId":   symbol
            }]
        }))
        print(f"[{symbol}] subscribed to books")

        last_sample_ts = time.time()
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("event") in ("subscribe", "cancel", "error"):
                continue
            if msg.get("arg", {}).get("channel") != "books":
                continue

            rec    = msg["data"][0]
            action = msg.get("action")

            if action == "snapshot":
                apply_snapshot(rec)
            elif action == "update":
                apply_update(rec)

            # 到采样时刻再取 TopN + VWAP
            now = int(time.time())
            if now - last_sample_ts < INTERVAL:
                continue
            last_sample_ts = now

            top_bids = sorted(bids_map.items(), key=lambda x: x[0], reverse=True)[:TOP_N]
            top_asks = sorted(asks_map.items(), key=lambda x: x[0])[:TOP_N]

            sum_b = sum(sz for _, sz in top_bids)
            sum_a = sum(sz for _, sz in top_asks)
            obpi  = (sum_b - sum_a) / (sum_b + sum_a) if (sum_b + sum_a) > 0 else 0.0

            # 计算 Depth‐VWAP
            bid_vwap = (sum(price * size for price, size in top_bids) / sum_b) if sum_b > 0 else 0.0
            ask_vwap = (sum(price * size for price, size in top_asks) / sum_a) if sum_a > 0 else 0.0
            depth_mid_vwap = (bid_vwap + ask_vwap) / 2

            # 打印
            print(f"[Books @ {now}] sum_bid={sum_b:.0f} sum_ask={sum_a:.0f} obpi={obpi:.4f}")
            print(f"VWAPs @ {now}: bid_vwap={bid_vwap:.6f}, ask_vwap={ask_vwap:.6f}, mid_vwap={depth_mid_vwap:.6f}")

            # 用 DataFrame.loc 添加新行，避免 concat/append 警告
            df_vwap_metrics.loc[len(df_vwap_metrics)] = {
                "ts": now,
                "bid_vwap": round(bid_vwap, 6),
                "ask_vwap": round(ask_vwap, 6),
                "depth_mid_vwap": round(depth_mid_vwap, 6)
            }
            print(df_vwap_metrics.tail())

            # 仅写入原字段到数据库
            row = {"ts": now}
            for i, (_, sz) in enumerate(top_bids, start=1):
                row[f"bid{i}"] = sz
            for i, (_, sz) in enumerate(top_asks, start=1):
                row[f"ask{i}"] = sz
            row["sum_bid"] = round(sum_b, 4)
            row["sum_ask"] = round(sum_a, 4)
            row["obpi"]    = round(obpi,   4)

            client_book.append_df_ignore(pd.DataFrame([row]))

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
    print("Start collecting orderbook (books)…")
    asyncio.run(collect_depth())
