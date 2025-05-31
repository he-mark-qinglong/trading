import asyncio
import websockets
import json
import pandas as pd
import time
from pathlib import Path

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
MAX_ROWS = 2000

async def _collect_trades_and_save_once(
    symbol="BTC-USDT-SWAP",
    interval=10,
    save_path="btc_10s_ohlcv.csv",
    archive_path=None
):
    # 自动派生 archive_path
    if archive_path is None:
        p = Path(save_path)
        archive_path = str(p.with_name(p.stem + "_archive" + p.suffix))

    # 1. 读主文件
    if Path(save_path).exists():
        ohlcv_df = pd.read_csv(save_path, index_col="ts")
        ohlcv_df.index = ohlcv_df.index.astype(int)
    else:
        ohlcv_df = pd.DataFrame(columns=["open", "high", "low", "close", "vol"])

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": symbol}]
        }))
        cache = []
        interval_start = int(time.time())
        print(f"Subscribed and saving OHLCV every {interval}s...")

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("No data for 30s, reconnecting...")
                break

            data = json.loads(msg)
            if data.get("arg", {}).get("channel") == "trades" and "data" in data:
                for t in data["data"]:
                    ts = int(int(t["ts"]) / 1000)
                    cache.append({
                        "ts": ts,
                        "price": float(t["px"]),
                        "size": float(t["sz"])
                    })

            now = int(time.time())
            if now - interval_start >= interval:
                if cache:
                    df = pd.DataFrame(cache)
                    open_  = df.iloc[0]["price"]
                    close_ = df.iloc[-1]["price"]
                    high_  = df["price"].max()
                    low_   = df["price"].min()
                    vol    = round(df["size"].sum(), 2)

                    new_row = pd.DataFrame(
                        [[open_, high_, low_, close_, vol]],
                        index=[interval_start],
                        columns=["open", "high", "low", "close", "vol"]
                    )
                    new_row.index.name = "ts"

                    # 2. 合并到主表
                    ohlcv_df = pd.concat([ohlcv_df, new_row])
                    # 去重、强制 int、排序
                    ohlcv_df = ohlcv_df[~ohlcv_df.index.duplicated(keep="last")]
                    ohlcv_df.index = ohlcv_df.index.astype(int)
                    ohlcv_df = ohlcv_df.sort_index()

                    # 3. 超长就切分归档
                    if len(ohlcv_df) > MAX_ROWS:
                        n_over = len(ohlcv_df) - MAX_ROWS
                        to_archive = ohlcv_df.iloc[:n_over]
                        print(f"[DEBUG] archiving {n_over} rows → {archive_path}")
                        # 追加到归档
                        if Path(archive_path).exists():
                            to_archive.to_csv(archive_path, mode="a", header=False)
                        else:
                            to_archive.to_csv(archive_path, mode="w")
                        # 保留最新 MAX_ROWS 行
                        ohlcv_df = ohlcv_df.iloc[-MAX_ROWS:]

                    # 4. 写回主文件
                    ohlcv_df.to_csv(save_path)
                    print(f"Saved new OHLCV @ ts={interval_start}: {open_},{high_},{low_},{close_},{vol}")

                # 重置
                cache = []
                interval_start = now

async def collect_trades_and_save(symbol="BTC-USDT-SWAP", interval=4):
    while True:
        try:
            save_path = symbol+'_4s_ohlcv.csv'
            await _collect_trades_and_save_once(symbol, interval, save_path)
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
            print(f"Websocket connection closed: {e}. Retrying in 2s...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unknown error: {e}. Retrying in 15s...")
            await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(collect_trades_and_save(symbol="ETH-USDT-SWAP"))