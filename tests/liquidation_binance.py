
import json
import websocket

def on_message(ws, message):
    """
    Called when a new message is received.
    We parse and pretty-print the JSON payload.
    """
    data = json.loads(message)
    print("Received forceOrder event:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    # Optionally, close after first message
    # ws.close()

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)

def on_open(ws):
    print("WebSocket connection opened.")

if __name__ == "__main__":
    # Replace 'ETHUSDT' with any symbol you care about
    symbol = "ETHUSDT"
    stream = f"{symbol.lower()}@forceOrder"
    url = f"wss://fstream.binance.com/ws/{stream}"
    
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    # Run forever, printing each liquidation (forceOrder) payload
    ws.run_forever()
