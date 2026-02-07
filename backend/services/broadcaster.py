# WebSocket Connection Manager and Broadcaster

import asyncio
import json
from typing import List, Dict, Any, Set
from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts data."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected clients."""
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        
        # Copy set to avoid modification during iteration
        async with self._lock:
            connections = list(self.active_connections)
        
        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)
    
    @property
    def client_count(self) -> int:
        return len(self.active_connections)


def create_broadcast_payload(
    frame_id: int,
    video_time: float,
    fps: float,
    board_box: List[int] | None,
    hands_info: Dict[str, Any],
    counters: Dict[str, float],
    alerts: List[str] = None,
    image: str = None
) -> Dict[str, Any]:
    """
    Create the JSON payload to broadcast to clients.
    
    Args:
        frame_id: Current frame number
        video_time: Timestamp in video
        fps: Current processing FPS
        board_box: [x1, y1, x2, y2] or None
        hands_info: Dict with hand data
        counters: Accumulated time counters
        alerts: List of alert messages
    
    Returns:
        JSON-serializable dictionary
    """
    payload = {
        "frame_id": frame_id,
        "video_time": round(video_time, 3),
        "fps": round(fps, 1),
        "board_zone": None,
        "hands": {},
        "counters": {k: round(v, 2) for k, v in counters.items()},
        "alerts": alerts or [],
        "image": image
    }
    
    if board_box:
        payload["board_zone"] = {
            "x1": board_box[0],
            "y1": board_box[1],
            "x2": board_box[2],
            "y2": board_box[3]
        }
    
    for hand_label in ["Left", "Right"]:
        if hand_label in hands_info:
            info = hands_info[hand_label]
            payload["hands"][hand_label] = {
                "visible": True,
                "state": info.get("state", "Transport"),
                "velocity": round(info.get("velocity", 0), 1),
                "fingers_in_zone": info.get("fingers", 0),
                "center": list(info.get("pos", [0, 0])),
                "landmarks": info.get("landmarks", [])
            }
        else:
            payload["hands"][hand_label] = {
                "visible": False,
                "state": "Transport",
                "velocity": 0,
                "fingers_in_zone": 0,
                "center": [0, 0],
                "landmarks": []
            }
    
    return payload
