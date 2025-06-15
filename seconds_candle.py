# collect_ohlcv.py

import asyncio
import json
import time

import pandas as pd
import websockets

from db_client import SQLiteWALClient, drop_orderbook_snap

symbol    = "ETH-USDT-SWAP"
WS_URL    = "wss://ws.okx.com:8443/ws/v5/public"
MAX_ROWS  = 100000      # 每张表保留的最大行数
INTERVAL  = 5           # 基础聚合周期：5s

DB_PATH    = f"{symbol}.db"
# 主表：1×interval
client1x   = SQLiteWALClient(db_path=DB_PATH, table="ohlcv_1x")
# 聚合表：9×interval
client9x  = SQLiteWALClient(db_path=DB_PATH, table="ohlcv_9x")

# drop_orderbook_snap(DB_PATH, "ohlcv_9x")

async def _collect_trades_and_save_once(symbol="BTC-USDT-SWAP", interval=INTERVAL):
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": symbol}]
        }))
        cache      = []    # 缓存这 interval 内的 trades
        interval_start = int(time.time())
        agg_buffer = []    # 缓存 1×interval K 线，用于 9×interval 聚合

        print(f"[{symbol}] subscribed, aggregating OHLCV every {interval}s…")

        while True:
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
                        "ts":    ts,
                        "price": float(t["px"]),
                        "size":  float(t["sz"])
                    })
            else:
                print(data)

            # 达到一个 interval 则产出 1× interval K 线
            now = int(time.time())
            if now - interval_start >= interval and cache:
                df     = pd.DataFrame(cache)
                open_  = df.iloc[0]["price"]
                close_ = df.iloc[-1]["price"]
                high_  = df["price"].max()
                low_   = df["price"].min()
                vol    = round(df["size"].sum(), 2)

                new_row = pd.DataFrame([{
                    "ts":    interval_start,
                    "open":  open_,
                    "high":  high_,
                    "low":   low_,
                    "close": close_,
                    "vol":   vol
                }])
                client1x.append_df_ignore(new_row)
                print(f"1× saved @ ts={interval_start}: "
                      f"{open_},{high_},{low_},{close_},{vol}")

                # 推入 buffer，等待 9 条 1×bar
                agg_buffer.append({
                    "ts":    interval_start,
                    "open":  open_,
                    "high":  high_,
                    "low":   low_,
                    "close": close_,
                    "vol":   vol
                })
                if len(agg_buffer) == 9:
                    ob = agg_buffer[0]
                    cb = agg_buffer[-1]
                    hh = max(r["high"] for r in agg_buffer)
                    ll = min(r["low"]  for r in agg_buffer)
                    vv = round(sum(r["vol"] for r in agg_buffer), 2)

                    row9x = pd.DataFrame([{
                        "ts":    ob["ts"],
                        "open":  ob["open"],
                        "high":  hh,
                        "low":   ll,
                        "close": cb["close"],
                        "vol":   vv
                    }])
                    client9x.append_df_ignore(row9x)
                    print(f"9× saved @ ts={ob['ts']}: "
                          f"{ob['open']},{hh},{ll},{cb['close']},{vv}")

                    # 清空 buffer，开启下一个 9-bar 聚合
                    agg_buffer.clear()

                # 修剪最老数据：保留最新 MAX_ROWS 条
                for cli in (client1x, client9x):
                    conn = cli._connect()
                    with conn:
                        total = conn.execute(
                            f"SELECT COUNT(*) FROM {cli.table}"
                        ).fetchone()[0]
                        if total > MAX_ROWS:
                            excess = total - MAX_ROWS
                            conn.execute(f"""
                                DELETE FROM {cli.table}
                                  WHERE ts IN (
                                    SELECT ts FROM {cli.table}
                                    ORDER BY ts ASC
                                    LIMIT ?
                                  )
                            """, (excess,))
                    conn.close()

                # 重置缓存和起始时间
                cache = []
                interval_start = now

async def collect_trades_and_save(symbol="BTC-USDT-SWAP", interval=INTERVAL):
    while True:
        try:
            await _collect_trades_and_save_once(symbol, interval)
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK):
            print("Websocket closed, retrying in 2s…")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unknown error: {e}, retry in 5s…")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(collect_trades_and_save(symbol, INTERVAL))