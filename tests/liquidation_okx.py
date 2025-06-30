import asyncio, json, websockets

OKX_WS = "wss://ws.okx.com:8443/ws/v5/public"

async def watch_liquidations():
    async with websockets.connect(OKX_WS) as ws:
        # 订阅 BTC-USDT 永续合约 的爆仓推送
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel":    "liquidation_orders",
                "instType":   "SWAP",               # SWAP：永续；FUTURES：交割合约；OPTION：期权
                "instId":     "BTC-USDT-SWAP"       # 可选，不填则订阅该 instType 下所有合约
            }]
        }))
        print("subscribed to liquidation_orders")
        async for msg in ws:
            data = json.loads(msg)
            # channel == liquidation_orders 时，data["data"] 是一个列表
            for liq in data.get("data", []):
                # 常见字段：instId(合约), px(爆仓价), sz(爆仓量), side(买/卖), ts(时间戳)
                print(liq)
                
asyncio.run(watch_liquidations())
