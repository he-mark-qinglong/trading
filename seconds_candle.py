import asyncio
import websockets
import json
import pandas as pd
import time
from pathlib import Path

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

async def _collect_trades_and_save_once(symbol="BTC-USDT-SWAP", interval=10, save_path="btc_10s_ohlcv.csv"):
    if Path(save_path).exists():
        ohlcv_df = pd.read_csv(save_path, index_col="ts")
    else:
        ohlcv_df = pd.DataFrame(columns=["open", "high", "low", "close", "vol"])

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        params = {
            "op": "subscribe",
            "args": [
                {"channel": "trades", "instId": symbol}
            ]
        }
        await ws.send(json.dumps(params))

        cache = []
        interval_start = int(time.time())

        print(f"Subscribed and saving OHLCV every {interval}s... Press Ctrl+C to stop")

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)  # 防止死等
            except asyncio.TimeoutError:
                print("No data for 30s, sending ping or reconnecting...")
                break  # 跳出让外层重连
            data = json.loads(msg)
            if "data" in data and data.get("arg", {}).get("channel") == "trades":
                for t in data["data"]:
                    price = float(t["px"])
                    size = float(t["sz"])
                    ts = int(int(t["ts"]) / 1000)  # 秒级时间戳
                    cache.append({"ts": ts, "price": price, "size": size})

            now = int(time.time())
            if now - interval_start >= interval:
                if cache:
                    df = pd.DataFrame(cache)
                    if not df.empty:
                        open_ = df.iloc[0]["price"]
                        close_ = df.iloc[-1]["price"]
                        high_ = df["price"].max()
                        low_ = df["price"].min()
                        vol = df["size"].sum()
                        ohlcv_row = pd.DataFrame(
                            [[open_, high_, low_, close_, vol]],
                            columns=["open", "high", "low", "close", "vol"],
                            index=[interval_start]
                        )
                        ohlcv_df = pd.concat([ohlcv_df, ohlcv_row])
                        ohlcv_df = ohlcv_df[~ohlcv_df.index.duplicated(keep="last")]
                        ohlcv_df.index.name = "ts"
                        ohlcv_df.to_csv(save_path)
                        print(ohlcv_row)
                cache = []
                interval_start = now

async def collect_trades_and_save(symbol="BTC-USDT-SWAP", interval=10, save_path="btc_10s_ohlcv.csv"):
    while True:
        try:
            await _collect_trades_and_save_once(symbol, interval, save_path)
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
            print(f"Websocket connection closed: {e}. Retrying in 2s...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unknown error: {e}. Retrying in 15s...")
            await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(collect_trades_and_save())