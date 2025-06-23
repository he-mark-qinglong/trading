# collect_and_merge.py

import asyncio
import json
import time
import pandas as pd
import websockets

from db_client import SQLiteWALClient, drop_orderbook_snap

# 配置
SYMBOL_OKX      = "ETH-USDT-SWAP"
SYMBOL_BINANCE  = "ethusdt"
INTERVAL        = 150        # 秒
DB_PATH         = f"{SYMBOL_OKX}.db"
OKX_WS          = "wss://ws.okx.com:8443/ws/v5/public"
BINANCE_WS      = f"wss://fstream.binance.com/ws/{SYMBOL_BINANCE}@aggTrade"
MAX_ROWS        = 100_000


okx_30x_cli      = SQLiteWALClient(DB_PATH, table="ohlcv_30x")
binance_30x_cli  = SQLiteWALClient(DB_PATH, table="binance_30x")
cmb_1x_cli      = SQLiteWALClient(DB_PATH, table="combined_1x")
cmb_30x_cli      = SQLiteWALClient(DB_PATH, table="combined_30x")
# drop_orderbook_snap(DB_PATH, "combined_30x")
async def collect_okx(queue: asyncio.Queue):
    """订阅 OKX trades，产出 1×、30× K 线，推到 queue"""
    while True:
        try:
            async with websockets.connect(OKX_WS, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{"channel": "trades", "instId": SYMBOL_OKX}]
                }))
                cache, agg_buf = [], []
                ts0 = int(time.time())
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        print("No data for 30s, reconnecting…")
                        break

                    data = json.loads(msg)
                    if data.get("arg", {}).get("channel") == "trades" and "data" in data:
                        for t in data["data"]:
                            s = int(int(t["ts"])//1000)
                            cache.append({"ts":s, "price":float(t["px"]), "vol":float(t["sz"])})
                    now = int(time.time())
                    if now - ts0 >= INTERVAL and cache:
                        df   = pd.DataFrame(cache)
                        o,h,l,c = df.iloc[0].price, df.price.max(), df.price.min(), df.iloc[-1].price
                        v    = round(df.vol.sum(),2)
                        row = {"ts":ts0,"open":o,"high":h,"low":l,"close":c,"vol":v}
                        
                        # 写 OKX 表
                        okx_30x_cli.append_df_ignore(pd.DataFrame([row]))
                        # print('okx 30x ', pd.DataFrame([row]))
                        # 推到队列
                        await queue.put(("30x", "okx", row))


                        cache, ts0 = [], now
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                asyncio.TimeoutError,
                OSError) as e:
            print(f"[OKX] 连接断开，准备重连：{type(e).__name__} {e!r}")
            await asyncio.sleep(INTERVAL)
        except Exception as e:
            print(f"[OKX] 未知异常，重连中：{e!r}")
            await asyncio.sleep(INTERVAL)

async def collect_binance(queue: asyncio.Queue):
    """订阅 Binance aggTrade，产出 1×、30× K 线，推到 queue"""
    while True:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=20, ping_timeout=20) as ws:
                cache = []
                ts0 = int(time.time())
                print("[Binance] subscribed")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    s = data["E"]//1000
                    cache.append({"ts":s, "price":float(data["p"]), "vol":float(data["q"])})
                    now = int(time.time())
                    if now - ts0 >= INTERVAL and cache:
                        df   = pd.DataFrame(cache)
                        o,h,l,c = df.iloc[0].price, df.price.max(), df.price.min(), df.iloc[-1].price
                        v    = round(df.vol.sum(),6)
                        row = {"ts":ts0,"open":o,"high":h,"low":l,"close":c,"vol":v}

                        binance_30x_cli.append_df_ignore(pd.DataFrame([row]))
                        # print('bnr 30x', pd.DataFrame([row]))
                        await queue.put(("30x", "binance", row))


                        cache, ts0 = [], now
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                asyncio.TimeoutError,
                OSError) as e:
            print(f"[Binance] 连接断开，准备重连：{type(e).__name__} {e!r}")
            await asyncio.sleep(INTERVAL)
        except Exception as e:
            print(f"[Binance] 未知异常，重连中：{e!r}")
            await asyncio.sleep(INTERVAL)
MAX_WAIT = 15  # 超时秒数

async def merger(queue: asyncio.Queue):
    """
    从 queue 里读 (level, source, row):
      - 缓存在 storage[level][ts][source] = row
      - 第一条到达时注册超时，MAX_WAIT 秒后若仍不完整，则在 _on_timeout 里处理
    """
    storage     = {"1x": {}, "30x": {}}
    first_arrive= {"1x": {}, "30x": {}}
    loop = asyncio.get_running_loop()

    def _on_timeout(level, ts):
        # 超时回调：如果还在 storage，说明没有正常合并
        lvl_st = storage[level]
        if ts not in lvl_st:
            return  # 已被正常合并并清理
        rowdict = lvl_st.pop(ts)
        first_arrive[level].pop(ts, None)

        okr = rowdict.get("okx")
        bnr = rowdict.get("binance")
        # 如果两端其实都到了，正常合并已做，这里直接返回
        if okr and bnr:
            return

        # 来一条“不完整”合并记录（另一端字段填 None 或 0）
        o = okr["open"]    if okr else bnr["open"]
        h = max(okr["high"], bnr["high"]) if okr and bnr else (okr or bnr)["high"]
        l = min(okr["low"],  bnr["low"])  if okr and bnr else (okr or bnr)["low"]
        c = okr["close"]   if okr else bnr["close"]
        v = (okr["vol"] if okr else 0) + (bnr["vol"] if not okr else 0)

        cmb = {"ts": ts, "open": o, "high": h, "low": l, "close": c, "vol": v}
        df  = pd.DataFrame([cmb])
        target = cmb_30x_cli
        target.append_df_ignore(df)
        print(f"[MERGED-timeout] {level} @ {ts}  okx={'Y' if okr else 'N'}  bin={'Y' if bnr else 'N'}")
        print(df)
    while True:
        level, src, row = await queue.get()
        ts = row["ts"]
        lvl_st = storage[level].setdefault(ts, {})

        # 第一次看到这个 (level, ts)，注册超时回调
        if ts not in first_arrive[level]:
            first_arrive[level][ts] = time.time()
            loop.call_later(MAX_WAIT, _on_timeout, level, ts)

        lvl_st[src] = row

        # 如果 OKX 和 Binance 都来了，就立即正常合并
        if "okx" in lvl_st and "binance" in lvl_st:
            okr = lvl_st["okx"]
            bnr = lvl_st["binance"]
            cmb = {
                "ts":    ts,
                "open":  min(okr["open"],  bnr["open"]),
                "high":  max(okr["high"],  bnr["high"]),
                "low":   min(okr["low"],   bnr["low"]),
                "close": max(okr["close"], bnr["close"]),
                "vol":   okr["vol"] + bnr["vol"],
            }
            df = pd.DataFrame([cmb])
        
            print("30x combined\t", df)
            cmb_30x_cli.append_df_ignore(df)
            print(f"[MERGED] {level} @ {ts}  OKX:{okr['vol']}  BIN:{bnr['vol']}")

            # 清理：后续即便 _on_timeout 触发也不会重复写
            storage[level].pop(ts, None)
            first_arrive[level].pop(ts, None)

        queue.task_done()

async def main():
    q = asyncio.Queue()
    await asyncio.gather(
        collect_okx(q),
        collect_binance(q),
        merger(q),
    )

if __name__=="__main__":
    asyncio.run(main())