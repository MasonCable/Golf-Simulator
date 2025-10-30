# listen_data_stream.py
import asyncio
import websockets
import json
from typing import AsyncGenerator, Dict, Any

async def shot_stream() -> AsyncGenerator[Dict[Any, Any], None]:
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print("Connected to simulator")
                async for msg in ws:
                    try:
                        shot = json.loads(msg)
                        yield shot
                    except json.JSONDecodeError:
                        print(f"Bad JSON: {msg}")
        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"Connection lost: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)