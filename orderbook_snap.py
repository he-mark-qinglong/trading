
import asyncio
import json
import time
from collections import defaultdict
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
# 保存 VWAP 与成交侧数据
df_metrics = pd.DataFrame(columns=[
    "ts",
    "bid_vwap", "ask_vwap", "depth_mid_vwap",
    "trade_buy_vol", "trade_buy_vwap",
    "trade_sell_vol", "trade_sell_vwap"
])

# 本地盘口状态
bids_map = {}  # price(float) -> size(float)
asks_map = {}
# 用于记录成交明细
trades = []

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
WINDOW = INTERVAL  # 秒

def apply_update(rec):
    global bid_eaten, ask_eaten
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

async def _collect_books_once():
    global df_vwap_metrics
    global bid_eaten, ask_eaten, last_ts

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel":  "books",
                "instType": "SWAP",
                "instId":   symbol
            }, 
            {
                "channel": "trades",
                "instType": "SWAP",
                "instId": symbol
            }]
        }))
        print(f"[{symbol}] subscribed to books")

        last_sample_ts = time.time()
        while True:
            time.sleep(1)
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("event") in ("subscribe", "cancel", "error"):
                continue

            if msg.get("arg", {}).get("channel") == "trades":
                for rec in msg["data"]:
                    trades.append({
                        "trade_id": rec["tradeId"],
                        "price":    float(rec["px"]),
                        "size":     float(rec["sz"]),
                        "side":     rec["side"],
                        "ts":       int(rec["ts"])
                    })

                # 到采样时刻再取 TopN + VWAP
                now = int(time.time())
                ts_int = int(now)
                if now - last_sample_ts < INTERVAL:  #last_sample_ts由盘口的部分进行更新，假设两个要么一起断开要么一起都能连接。
                    continue
                last_sample_ts = now   # ← 这里一定要更新！

                # 1) 聚合同价同向成交（可选展示）
                agg = defaultdict(float)
                for tr in trades:
                    agg[(tr["price"], tr["side"])] += tr["size"]
                    
                # print(f"\n[Trades @ {ts_int}] 合并后成交：")
                # for (price, side), size in sorted(agg.items()):
                #     print(f"  {side:<4} 量={size:.4f} 价={price:.2f}")

                # 2) 分离计算买卖方向成交量价
                buy_trades  = [tr for tr in trades if tr["side"] == "buy"]
                sell_trades = [tr for tr in trades if tr["side"] == "sell"]

                buy_vol = sum(tr["size"] for tr in buy_trades)
                sell_vol= sum(tr["size"] for tr in sell_trades)
                buy_vwap = sum(tr["price"]*tr["size"] for tr in buy_trades) / buy_vol  if buy_vol else 0.0
                sell_vwap= sum(tr["price"]*tr["size"] for tr in sell_trades)/ sell_vol if sell_vol else 0.0

                print(f"[TradesMetrics @ {ts_int}] buy_vol={buy_vol:.4f}, buy_vwap={buy_vwap:.6f}")
                print(f"[TradesMetrics @ {ts_int}] sell_vol={sell_vol:.4f}, sell_vwap={sell_vwap:.6f}")

                trades.clear()



                # 3) 计算盘口 VWAP（TopN）
                top_bids = sorted(bids_map.items(), key=lambda x: x[0], reverse=True)[:TOP_N]
                top_asks = sorted(asks_map.items(), key=lambda x: x[0])[:TOP_N]
                sum_b = sum(sz for _, sz in top_bids)
                sum_a = sum(sz for _, sz in top_asks)
                bid_vwap = sum(p*sz for p,sz in top_bids)/sum_b if sum_b else 0.0
                ask_vwap = sum(p*sz for p,sz in top_asks)/sum_a if sum_a else 0.0
                mid_vwap = (bid_vwap + ask_vwap) / 2

                print(f"[VWAP @ {ts_int}] bid={bid_vwap:.6f}, ask={ask_vwap:.6f}, mid={mid_vwap:.6f}")

                # 4) 存入 DataFrame
                df_metrics.loc[len(df_metrics)] = {
                    "ts": ts_int,
                    "bid_vwap":      round(bid_vwap, 6),
                    "ask_vwap":      round(ask_vwap, 6),
                    "depth_mid_vwap":round(mid_vwap, 6),
                    "trade_buy_vol": round(buy_vol, 6),
                    "trade_buy_vwap":round(buy_vwap, 6),
                    "trade_sell_vol":round(sell_vol, 6),
                    "trade_sell_vwap":round(sell_vwap, 6),
                }

                # 5) 写盘口快照到数据库
                row = {"ts": ts_int}
                for i,(_,sz) in enumerate(top_bids, start=1):
                    row[f"bid{i}"] = sz
                for i,(_,sz) in enumerate(top_asks, start=1):
                    row[f"ask{i}"] = sz
                row["sum_bid"] = round(sum_b,4)
                row["sum_ask"] = round(sum_a,4)
                row["obpi"]    = round((sum_b - sum_a)/(sum_b + sum_a) if (sum_b+sum_a) else 0,4)
                client_book.append_df_ignore(pd.DataFrame([row]))

                print(f'obpi={row["obpi"]}')
                # 清空已处理成交
                trades.clear()

                now = time.time()
                if now - last_ts >= WINDOW:
                    bid_rate = bid_eaten / WINDOW
                    ask_rate = ask_eaten / WINDOW
                    print(f"[{WINDOW}s] bid_eat_rate={bid_rate:.4f}, ask_eat_rate={ask_rate:.4f} is_bid_eat_stronger={bid_rate > ask_rate}")
                    bid_eaten = ask_eaten = 0.0
                    last_ts = now

                continue
            if msg.get("arg", {}).get("channel") == "books":
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
