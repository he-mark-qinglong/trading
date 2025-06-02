# collect_ohlcv.py

import asyncio
import json
import time

import pandas as pd
import websockets

from db_client import SQLiteWALClient

symbol="ETH-USDT-SWAP"
WS_URL   = "wss://ws.okx.com:8443/ws/v5/public"
MAX_ROWS = 100000   # 主表最大行数，超过后修剪最老数据

# 把 CSV 路径换成 db
DB_PATH  = f"{symbol}.db"
client   = SQLiteWALClient(db_path=DB_PATH, table="ohlcv")

async def _collect_trades_and_save_once(
    symbol="BTC-USDT-SWAP",
    interval=10,
):
    # 订阅 websocket
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": symbol}]
        }))
        cache = []
        interval_start = int(time.time())
        print(f"[{symbol}] subscribed, aggregating OHLCV every {interval}s…")

        while True:
            # 1. 接收消息
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("No data for 30s, reconnecting…")
                break

            data = json.loads(msg)
            if data.get("arg", {}).get("channel") == "trades" and "data" in data:
                for t in data["data"]:
                    ts = int(int(t["ts"]) / 1000)
                    cache.append({
                        "ts": ts,
                        "price": float(t["px"]),
                        "size":  float(t["sz"])
                    })

            # 2. 达到一个时间段，计算 OHLCV
            now = int(time.time())
            if now - interval_start >= interval and cache:
                df = pd.DataFrame(cache)
                open_  = df.iloc[0]["price"]
                close_ = df.iloc[-1]["price"]
                high_  = df["price"].max()
                low_   = df["price"].min()
                vol    = round(df["size"].sum(), 2)

                new_row = pd.DataFrame([{
                    "ts":   interval_start,
                    "open": open_, "high": high_,
                    "low":  low_,   "close": close_,
                    "vol":  vol
                }])

                # 3. 写入 DB（主键冲突时跳过）
                client.append_df_ignore(new_row)
                print(f"Saved OHLCV @ ts={interval_start}: "
                      f"{open_},{high_},{low_},{close_},{vol}")

                # 4. 修剪最老数据：保留最新 MAX_ROWS 条
                conn = client._connect()
                with conn:
                    # 4.1 计算总行数
                    total = conn.execute(f"SELECT COUNT(*) FROM {client.table}").fetchone()[0]
                    if total > MAX_ROWS:
                        excess = total - MAX_ROWS
                        # 4.2 删除最老的 excess 条
                        conn.execute(f"""
                            DELETE FROM {client.table}
                              WHERE ts IN (
                                SELECT ts FROM {client.table}
                                ORDER BY ts ASC
                                LIMIT ?
                              )
                        """, (excess,))
                conn.close()

                # 重置缓存和起始时间
                cache = []
                interval_start = now

async def collect_trades_and_save(symbol="BTC-USDT-SWAP", interval=10):
    while True:
        try:
            await _collect_trades_and_save_once(symbol, interval)
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK):
            print("Websocket closed, retrying in 2s…")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unknown error: {e}, retry in 15s…")
            await asyncio.sleep(15)

if __name__ == "__main__":
    # 启动示例，用 10s K 线
    asyncio.run(collect_trades_and_save(symbol=symbol, interval=4))