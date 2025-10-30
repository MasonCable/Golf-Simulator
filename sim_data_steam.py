#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Square Golf Data Stream Simulator – returns live object

Features:
- WebSocket server (ws://localhost:8765)
- UDP broadcast
- Non-blocking start
- sim.stop() to shut down

Usage:
    sim = run_simulate_data_stream('real_shots.json', delay=2.0)
    # → server is live
    # later: sim.stop()
"""

import json
import time
import asyncio
import threading
import socket
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
WEBSOCKET_PORT = 8765
DEFAULT_UDP_PORT = 9001
DEFAULT_UDP_BROADCAST_ADDR = "255.255.255.255"


# ----------------------------------------------------------------------
# Core simulator
# ----------------------------------------------------------------------
class SquareSimulator:
    def __init__(self, shots: List[Dict[Any, Any]], delay: float = 3.0):
        self.shots = shots
        self.delay = delay

        self._stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._udp_thread: threading.Thread | None = None

    # --------------------------------------------------------------
    # WebSocket server (runs in its own thread with asyncio.run)
    # --------------------------------------------------------------
    async def _ws_handler(self, websocket, path):
        for shot in self.shots:
            if self._stop_event.is_set():
                break
            await websocket.send(json.dumps(shot))
            await asyncio.sleep(self.delay)

    def _run_websocket_server(self):
        async def server():
            from websockets.server import serve
            async with serve(self._ws_handler, "localhost", WEBSOCKET_PORT) as server:
                print(f"WebSocket server RUNNING → ws://localhost:{WEBSOCKET_PORT}")
                await asyncio.Future()  # run forever

        try:
            asyncio.run(server())
        except asyncio.CancelledError:
            pass

    # --------------------------------------------------------------
    # UDP broadcaster
    # --------------------------------------------------------------
    def _udp_loop(self, ip: str, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        print(f"UDP broadcaster → {ip}:{port}")

        for shot in self.shots:
            if self._stop_event.is_set():
                break
            msg = json.dumps(shot).encode("utf-8")
            sock.sendto(msg, (ip, port))
            time.sleep(self.delay)

        sock.close()

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def start(self, ws: bool = True, udp: bool = True,
              udp_ip: str = DEFAULT_UDP_BROADCAST_ADDR,
              udp_port: int = DEFAULT_UDP_PORT):
        """Start transports in background threads."""
        if ws:
            self._ws_thread = threading.Thread(target=self._run_websocket_server, daemon=True)
            self._ws_thread.start()

        if udp:
            self._udp_thread = threading.Thread(
                target=self._udp_loop, args=(udp_ip, udp_port), daemon=True
            )
            self._udp_thread.start()

        print(f"Sending {len(self.shots)} shots every {self.delay:.1f}s …")

    def stop(self):
        """Stop everything."""
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=1.0)
        if self._udp_thread and self._udp_thread.is_alive():
            self._udp_thread.join(timeout=1.0)
        print("Simulator stopped.")


# ----------------------------------------------------------------------
# Helper – load + return live object
# ----------------------------------------------------------------------
def run_simulate_data_stream(json_file: str, delay: float = 10.0):
    """
    Load shots, start WebSocket + UDP, and return the live simulator.
    """
    # ---- Load & validate ------------------------------------------------
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON root must be an array")
        required = {"BallSpeed", "VLA", "HLA", "BackSpin", "SideSpin"}
        for i, s in enumerate(data):
            if not required.issubset(s.keys()):
                raise ValueError(f"Shot {i} missing required keys")
    except Exception as e:
        print(f"Error loading shots: {e}")
        return None

    # ---- Start simulator -------------------------------------------------
    sim = SquareSimulator(data, delay=delay)
    sim.start(ws=True, udp=True)
    return sim
