"""
connection_manager.py - WebSocket connection management for EchoForge
"""
import logging
import json
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("Connection manager initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_metadata[session_id] = {
            "connected_at": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown"),
            "message_count": 0,
            "last_activity": None
        }
        logger.info(f"WebSocket connected: {session_id}")
        
        # Send welcome message
        await self.send_to_session(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to EchoForge",
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
        })
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific session"""
        if session_id in self.active_connections:
            try:
                # Update session metadata
                if session_id in self.session_metadata:
                    self.session_metadata[session_id]["message_count"] += 1
                    self.session_metadata[session_id]["last_activity"] = str(
                        logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown"
                    )
                
                # Send the message
                await self.active_connections[session_id].send_json(message)
                logger.debug(f"Message sent to {session_id}: {message.get('type', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {str(e)}")
                # Remove the connection if it's broken
                self.disconnect(session_id)
        else:
            logger.warning(f"Session {session_id} not found for message delivery")
    
    async def send_typing_indicator(self, session_id: str, agent_name: str, is_typing: bool = True):
        """Send typing indicator to show agent is working"""
        await self.send_to_session(session_id, {
            "type": "typing_indicator",
            "agent": agent_name,
            "is_typing": is_typing,
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
        })
    
    async def broadcast(self, message: Dict[str, Any], exclude_session: str = None):
        """Send a message to all connected sessions"""
        disconnected = []
        sent_count = 0
        
        for session_id, websocket in self.active_connections.items():
            if exclude_session and session_id == exclude_session:
                continue
                
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {session_id}: {str(e)}")
                disconnected.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected:
            self.disconnect(session_id)
        
        logger.info(f"Broadcast sent to {sent_count} sessions")
    
    async def send_system_message(self, session_id: str, message: str, message_type: str = "info"):
        """Send a system message to a session"""
        await self.send_to_session(session_id, {
            "type": "system_message",
            "message_type": message_type,
            "content": message,
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
        })
    
    async def send_error(self, session_id: str, error_message: str, error_code: str = None):
        """Send an error message to a session"""
        await self.send_to_session(session_id, {
            "type": "error",
            "content": error_message,
            "error_code": error_code,
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
        })
    
    async def send_progress_update(self, session_id: str, stage: str, progress: int, total: int, message: str = None):
        """Send progress update for long-running operations"""
        await self.send_to_session(session_id, {
            "type": "progress_update",
            "stage": stage,
            "progress": progress,
            "total": total,
            "percentage": round((progress / total) * 100, 1) if total > 0 else 0,
            "message": message,
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
        })
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get metadata for a specific session"""
        return self.session_metadata.get(session_id, {})
    
    def get_all_session_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all sessions"""
        return self.session_metadata.copy()
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is currently active"""
        return session_id in self.active_connections
    
    async def ping_all_connections(self):
        """Send ping to all connections to check health"""
        disconnected = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.ping()
            except Exception as e:
                logger.warning(f"Ping failed for session {session_id}: {str(e)}")
                disconnected.append(session_id)
        
        # Clean up unresponsive connections
        for session_id in disconnected:
            self.disconnect(session_id)
        
        if disconnected:
            logger.info(f"Cleaned up {len(disconnected)} unresponsive connections")
    
    async def close_all_connections(self):
        """Close all WebSocket connections gracefully"""
        sessions_to_close = list(self.active_connections.keys())
        
        for session_id in sessions_to_close:
            try:
                websocket = self.active_connections[session_id]
                
                # Send goodbye message
                await websocket.send_json({
                    "type": "server_shutdown",
                    "message": "Server is shutting down",
                    "timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown")
                })
                
                # Close the connection
                await websocket.close()
                
            except Exception as e:
                logger.error(f"Error closing connection {session_id}: {str(e)}")
            finally:
                self.disconnect(session_id)
        
        logger.info("All WebSocket connections closed")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about connections"""
        total_messages = sum(
            metadata.get("message_count", 0) 
            for metadata in self.session_metadata.values()
        )
        
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": total_messages,
            "sessions": list(self.active_connections.keys())
        }
