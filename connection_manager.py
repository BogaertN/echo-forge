#!/usr/bin/env python3
"""
Connection manager for EchoForge WebSocket connections.
Handles real-time communication between clients and the server.
"""

import logging
from typing import Dict, List
from fastapi import WebSocket
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for EchoForge."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        logger.info("Connection manager initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send a message to a specific session."""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                await websocket.send_text(json.dumps(message))
                logger.debug(f"Message sent to {session_id}: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                # Remove broken connection
                self.disconnect(session_id)
        else:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active connections to broadcast to")
            return
        
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)
        
        logger.info(f"Message broadcast to {len(self.active_connections)} connections")
    
    async def disconnect_all(self):
        """Disconnect all active connections."""
        session_ids = list(self.active_connections.keys())
        
        for session_id in session_ids:
            try:
                websocket = self.active_connections[session_id]
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {session_id}: {e}")
            finally:
                self.disconnect(session_id)
        
        logger.info("All WebSocket connections closed")
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_connections.keys())
    
    def is_connected(self, session_id: str) -> bool:
        """Check if a session is connected."""
        return session_id in self.active_connections
