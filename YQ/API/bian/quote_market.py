import asyncio
import time
import websocket
import json
import threading
import ccxt
from decimal import Decimal
import winsound
import web3

def on_error(ws, error):
    global reconnect_count
    print(type(error))
    print(error)
    print("正在尝试第%d次重连"%reconnect_count)
    with open('errors.log', 'w+') as f:
        f.write(str(error))
    reconnect_count += 1
    if reconnect_count < 1000:
        connection_tmp(ws)
        time.sleep(3)

def on_close(ws):
    global reconnect_count
    print("### closed ###")
    print('restarting')
    reconnect_count += 1
    if reconnect_count < 1000:
        connection_tmp(ws)
        time.sleep(3)

def connection_tmp(ws):
    websocket.enableTrace(True)
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
    except:
        ws.close()


def on_message(ws, message):
    # print(message)
    global bid0, bidV0, ask0, askV0,  bid1, bidV1, ask1, askV1, bid2, bidV2, ask2, askV2
    _ = json.loads(message)
    # print(_)
    if _['data']['s'] == symbol0:
        bid0, bidV0, ask0, askV0 = float(_['data']['b']), float(_['data']['B']), float(_['data']['a']), float(_['data']['A'])
    elif _['data']['s'] == symbol1:
        bid1, bidV1, ask1, askV1 = float(_['data']['b']), float(_['data']['B']), float(_['data']['a']), float(_['data']['A'])
    elif _['data']['s'] == symbol2:
        bid2, bidV2, ask2, askV2 = float(_['data']['b']), float(_['data']['B']), float(_['data']['a']), float(_['data']['A'])


def on_open(ws):
    print('starting')



def main():
    while True:
        print('*' * 100)
        if not all((bid0, ask0, bid1, ask1, bid2, ask2)):
            print('等待获取全部行情')
            time.sleep(1)
            continue
        syntheticBid0 = bid1 / ask2
        syntheticAsk0 = ask1 / bid2
        print('通过{}与{}合成的{}买价为：{}， 卖价为：{}'.format(symbol1, symbol2, symbol0, syntheticBid0, syntheticAsk0))
        print('当前{}盘口的买价为：{}, 卖价为：{}'.format(symbol0, bid0, ask0))
        print(ask0/syntheticAsk0)
        print(syntheticBid0/bid0)
        if ask0/syntheticAsk0 > 1.003:
            print('卖出{}，卖出{}，买入{}'.format(symbol0, symbol2, symbol1))

            print(askV0)
        if syntheticBid0/bid0 > 1.003 and bidV0 < 5000000:
            print(bidV0)
            print('买入{}，卖出{}，买入{}'.format(symbol0, symbol1, symbol2))
        time.sleep(0.001)



if __name__ == '__main__':
    ex = ccxt.binance()
    ex.apiKey = ''
    ex.secret = ''
    symbol0 = 'ETHBTC'
    symbol1 = 'ETHUSDT'
    symbol2 = 'BTCUSDT'
    reconnect_count = 0
    bid0, bidV0, ask0,askV0,  bid1, bidV1, ask1, askV1, bid2, bidV2, ask2, askV2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ws = websocket.WebSocketApp('wss://stream.yshyqxx.com/stream?streams=%s@bookTicker/%s@bookTicker/%s@bookTicker' % (symbol0.lower(), symbol1.lower(), symbol2.lower()),
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)

    threads = []
    threads.append(threading.Thread(target=ws.run_forever,))
    threads.append(threading.Thread(target=main))
    for t in threads:
        t.start()