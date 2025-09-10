import asyncio
import json
import logging
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import weakref

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    """Message types for WebSocket communication"""
    # System messages
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    STATUS = "status"
    
    # Clarification messages
    CLARIFICATION_STARTED = "clarification_started"
    CLARIFICATION_QUESTION = "clarification_question"
    CLARIFICATION_RESPONSE = "clarification_response"
    CLARIFICATION_COMPLETE = "clarification_complete"
    
    # Debate messages
    DEBATE_STARTED = "debate_started"
    DEBATE_ROUND_STARTED = "round_started"
    AGENT_RESPONSE = "agent_response"
    SPECIALIST_INPUT = "specialist_input"
    SYNTHESIS_GENERATED = "synthesis_generated"
    DEBATE_COMPLETE = "debate_complete"
    
    # Quality assurance
    GHOST_LOOP_DETECTED = "ghost_loop_detected"
    AUDIT_COMPLETE = "audit_complete"
    
    # Journal messages
    JOURNAL_ENTRY_PREPARED = "journal_entry_prepared"
    JOURNAL_SAVED = "journal_saved"
    
    # Resonance map
    RESONANCE_MAP_DATA = "resonance_map_data"
    
    # Preferences
    PREFERENCES_UPDATED = "preferences_updated"

@dataclass
class Message:
    """WebSocket message structure"""
    type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for JSON serialization"""
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class Connection:
    """WebSocket connection information"""
    websocket: WebSocket
    session_id: str
    client_ip: str
    user_agent: str
    connected_at: datetime
    last_activity: datetime
    state: ConnectionState
    message_count: int = 0
    error_count: int = 0
    last_ping: Optional[datetime] = None
    last_pong: Optional[datetime] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def increment_message_count(self):
        """Increment message counter"""
        self.message_count += 1
        self.update_activity()
    
    def increment_error_count(self):
        """Increment error counter"""
        self.error_count += 1
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        if self.state != ConnectionState.CONNECTED:
            return False
        
        # Check if websocket is still open
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return False
        
        # Check for recent activity (within last 5 minutes)
        if datetime.now() - self.last_activity > timedelta(minutes=5):
            return False
        
        # Check ping/pong health if available
        if self.last_ping and not self.last_pong:
            if datetime.now() - self.last_ping > timedelta(seconds=30):
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        uptime = datetime.now() - self.connected_at
        return {
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "connected_at": self.connected_at.isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "state": self.state.value,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_activity": self.last_activity.isoformat(),
            "is_healthy": self.is_healthy()
        }

class MessageQueue:
    """Message queue for handling offline clients and message buffering"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queues: Dict[str, List[Message]] = {}
        self.message_counts: Dict[str, int] = {}
    
    def add_message(self, session_id: str, message: Message):
        """Add message to session queue"""
        if session_id not in self.queues:
            self.queues[session_id] = []
            self.message_counts[session_id] = 0
        
        self.queues[session_id].append(message)
        self.message_counts[session_id] += 1
        
        # Trim queue if too large
        if len(self.queues[session_id]) > self.max_size:
            removed = self.queues[session_id].pop(0)
            logger.debug(f"Removed message from queue for session {session_id}: {removed.type}")
    
    def get_messages(self, session_id: str) -> List[Message]:
        """Get all queued messages for session"""
        return self.queues.get(session_id, [])
    
    def clear_queue(self, session_id: str):
        """Clear message queue for session"""
        if session_id in self.queues:
            del self.queues[session_id]
        if session_id in self.message_counts:
            del self.message_counts[session_id]
    
    def get_queue_size(self, session_id: str) -> int:
        """Get queue size for session"""
        return len(self.queues.get(session_id, []))
    
    def cleanup_old_messages(self, max_age_hours: int = 24):
        """Clean up old messages from queues"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for session_id in list(self.queues.keys()):
            original_count = len(self.queues[session_id])
            self.queues[session_id] = [
                msg for msg in self.queues[session_id]
                if msg.timestamp > cutoff_time
            ]
            removed_count = original_count - len(self.queues[session_id])
            
            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} old messages for session {session_id}")

class RateLimiter:
    """Rate limiter for WebSocket messages to prevent spam"""
    
    def __init__(self, max_messages: int = 100, window_minutes: int = 1):
        self.max_messages = max_messages
        self.window_minutes = window_minutes
        self.message_times: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, session_id: str) -> bool:
        """Check if message is allowed under rate limit"""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=self.window_minutes)
        
        # Initialize or clean up old messages
        if session_id not in self.message_times:
            self.message_times[session_id] = []
        
        self.message_times[session_id] = [
            timestamp for timestamp in self.message_times[session_id]
            if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.message_times[session_id]) >= self.max_messages:
            return False
        
        # Add current message time
        self.message_times[session_id].append(now)
        return True
    
    def get_remaining_quota(self, session_id: str) -> int:
        """Get remaining message quota for session"""
        current_count = len(self.message_times.get(session_id, []))
        return max(0, self.max_messages - current_count)
    
    def cleanup_old_entries(self):
        """Clean up old rate limit entries"""
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes * 2)
        
        for session_id in list(self.message_times.keys()):
            self.message_times[session_id] = [
                timestamp for timestamp in self.message_times[session_id]
                if timestamp > cutoff_time
            ]
            
            if not self.message_times[session_id]:
                del self.message_times[session_id]

class ConnectionManager:
    """
    Manages WebSocket connections for EchoForge real-time communication.
    
    Provides session-based connection management, message routing, health monitoring,
    and automatic cleanup of stale connections.
    """
    
    def __init__(self):
        # Active connections by session ID
        self.connections: Dict[str, Connection] = {}
        
        # Message queue for offline clients
        self.message_queue = MessageQueue(max_size=200)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_messages=100, window_minutes=1)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_errors": 0,
            "connection_errors": 0,
            "message_errors": 0,
            "peak_concurrent_connections": 0
        }
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        # self._start_background_tasks()  # Delay until server starts
        
        logger.info("ConnectionManager initialized")
    
    def _start_background_tasks(self):
        """Start background tasks for health monitoring and cleanup"""
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Background task for monitoring connection health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_connection_health()
                await self._cleanup_stale_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
    
    async def _check_connection_health(self):
        """Check health of all connections and clean up unhealthy ones"""
        unhealthy_sessions = []
        
        for session_id, connection in self.connections.items():
            if not connection.is_healthy():
                unhealthy_sessions.append(session_id)
                logger.warning(f"Unhealthy connection detected for session {session_id}")
        
        # Clean up unhealthy connections
        for session_id in unhealthy_sessions:
            await self.disconnect(session_id, reason="health_check_failed")
    
    async def _cleanup_stale_data(self):
        """Clean up stale data from queues and rate limiters"""
        self.message_queue.cleanup_old_messages(max_age_hours=24)
        self.rate_limiter.cleanup_old_entries()
    
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept and manage a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            session_id: Unique session identifier
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Accept the WebSocket connection
            await websocket.accept()
            
            # Get client information
            client_ip = getattr(websocket.client, 'host', 'unknown') if websocket.client else 'unknown'
            user_agent = websocket.headers.get('user-agent', 'unknown')
            
            # Check if session already has a connection
            if session_id in self.connections:
                logger.warning(f"Session {session_id} already has a connection, closing old one")
                await self.disconnect(session_id, reason="duplicate_connection")
            
            # Create connection object
            connection = Connection(
                websocket=websocket,
                session_id=session_id,
                client_ip=client_ip,
                user_agent=user_agent,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                state=ConnectionState.CONNECTED
            )
            
            # Store connection
            self.connections[session_id] = connection
            
            # Update metrics
            self.metrics["total_connections"] += 1
            current_connections = len(self.connections)
            if current_connections > self.metrics["peak_concurrent_connections"]:
                self.metrics["peak_concurrent_connections"] = current_connections
            
            # Send any queued messages
            await self._send_queued_messages(session_id)
            
            # Send connection confirmation
            await self.send_message(session_id, {
                "type": MessageType.STATUS.value,
                "payload": {
                    "status": "connected",
                    "session_id": session_id,
                    "server_time": datetime.now().isoformat(),
                    "queue_size": self.message_queue.get_queue_size(session_id)
                }
            })
            
            # Trigger connection event
            await self._trigger_event("connection_established", {
                "session_id": session_id,
                "client_ip": client_ip,
                "user_agent": user_agent
            })
            
            logger.info(f"WebSocket connection established for session {session_id} from {client_ip}")
            return True
            
        except Exception as e:
            logger.error(f"Error establishing WebSocket connection for session {session_id}: {e}")
            self.metrics["connection_errors"] += 1
            return False
    
    async def disconnect(self, session_id: str, reason: str = "client_disconnected") -> bool:
        """
        Disconnect a WebSocket connection.
        
        Args:
            session_id: Session identifier
            reason: Reason for disconnection
            
        Returns:
            True if disconnection successful, False if session not found
        """
        if session_id not in self.connections:
            logger.warning(f"Attempted to disconnect non-existent session: {session_id}")
            return False
        
        connection = self.connections[session_id]
        
        try:
            # Update connection state
            connection.state = ConnectionState.DISCONNECTING
            
            # Close WebSocket if still open
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close()
            
            # Remove from active connections
            del self.connections[session_id]
            
            # Trigger disconnection event
            await self._trigger_event("connection_closed", {
                "session_id": session_id,
                "reason": reason,
                "stats": connection.get_stats()
            })
            
            logger.info(f"WebSocket connection closed for session {session_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting session {session_id}: {e}")
            # Force remove from connections
            if session_id in self.connections:
                del self.connections[session_id]
            return False
    
    async def send_message(self, session_id: str, message_data: Dict[str, Any]) -> bool:
        """
        Send message to a specific session.
        
        Args:
            session_id: Target session identifier
            message_data: Message content
            
        Returns:
            True if message sent successfully, False otherwise
        """
        # Create message object
        message = Message(
            type=message_data.get("type", "unknown"),
            payload=message_data.get("payload", {})
        )
        
        # Check rate limiting
        if not self.rate_limiter.is_allowed(session_id):
            logger.warning(f"Rate limit exceeded for session {session_id}")
            await self.send_error(session_id, "Rate limit exceeded", error_code="RATE_LIMIT")
            return False
        
        # Check if connection exists and is healthy
        if session_id not in self.connections:
            logger.debug(f"Session {session_id} not connected, queueing message")
            self.message_queue.add_message(session_id, message)
            return False
        
        connection = self.connections[session_id]
        
        if not connection.is_healthy():
            logger.warning(f"Unhealthy connection for session {session_id}, queueing message")
            self.message_queue.add_message(session_id, message)
            await self.disconnect(session_id, reason="unhealthy_connection")
            return False
        
        try:
            # Send message
            await connection.websocket.send_text(message.to_json())
            
            # Update connection stats
            connection.increment_message_count()
            self.metrics["total_messages_sent"] += 1
            
            logger.debug(f"Message sent to session {session_id}: {message.type}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
            self.metrics["message_errors"] += 1
            connection.increment_error_count()
            
            # Queue message for retry
            self.message_queue.add_message(session_id, message)
            
            # Disconnect unhealthy connection
            await self.disconnect(session_id, reason="send_error")
            return False
    
    async def send_error(self, session_id: str, error_message: str, error_code: str = "GENERAL_ERROR") -> bool:
        """
        Send error message to a specific session.
        
        Args:
            session_id: Target session identifier
            error_message: Error description
            error_code: Error code for categorization
            
        Returns:
            True if error sent successfully, False otherwise
        """
        error_data = {
            "type": MessageType.ERROR.value,
            "payload": {
                "message": error_message,
                "code": error_code,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.metrics["total_errors"] += 1
        return await self.send_message(session_id, error_data)
    
    async def broadcast_message(self, message_data: Dict[str, Any], exclude_sessions: Set[str] = None) -> int:
        """
        Broadcast message to all connected sessions.
        
        Args:
            message_data: Message content
            exclude_sessions: Set of session IDs to exclude from broadcast
            
        Returns:
            Number of sessions the message was sent to
        """
        if exclude_sessions is None:
            exclude_sessions = set()
        
        sent_count = 0
        
        for session_id in list(self.connections.keys()):
            if session_id not in exclude_sessions:
                if await self.send_message(session_id, message_data):
                    sent_count += 1
        
        logger.info(f"Broadcast message sent to {sent_count} sessions")
        return sent_count
    
    async def _send_queued_messages(self, session_id: str):
        """Send queued messages to a newly connected session"""
        queued_messages = self.message_queue.get_messages(session_id)
        
        if not queued_messages:
            return
        
        logger.info(f"Sending {len(queued_messages)} queued messages to session {session_id}")
        
        for message in queued_messages:
            try:
                await self.send_message(session_id, message.to_dict())
            except Exception as e:
                logger.error(f"Error sending queued message to session {session_id}: {e}")
                break
        
        # Clear queue after sending
        self.message_queue.clear_queue(session_id)
    
    async def ping_session(self, session_id: str) -> bool:
        """
        Send ping to specific session to check connection health.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if ping sent successfully, False otherwise
        """
        if session_id not in self.connections:
            return False
        
        connection = self.connections[session_id]
        connection.last_ping = datetime.now()
        
        ping_data = {
            "type": MessageType.PING.value,
            "payload": {
                "timestamp": connection.last_ping.isoformat()
            }
        }
        
        return await self.send_message(session_id, ping_data)
    
    async def handle_pong(self, session_id: str, pong_data: Dict):
        """
        Handle pong response from client.
        
        Args:
            session_id: Session identifier
            pong_data: Pong message data
        """
        if session_id in self.connections:
            connection = self.connections[session_id]
            connection.last_pong = datetime.now()
            
            # Calculate round-trip time if ping timestamp available
            ping_timestamp = pong_data.get("payload", {}).get("ping_timestamp")
            if ping_timestamp and connection.last_ping:
                try:
                    ping_time = datetime.fromisoformat(ping_timestamp.replace('Z', '+00:00'))
                    rtt = (connection.last_pong - ping_time).total_seconds() * 1000  # milliseconds
                    logger.debug(f"Session {session_id} RTT: {rtt:.2f}ms")
                except Exception as e:
                    logger.warning(f"Error calculating RTT for session {session_id}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """
        Add event handler for connection events.
        
        Args:
            event_type: Type of event (connection_established, connection_closed, etc.)
            handler: Async function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Added event handler for {event_type}")
    
    async def _trigger_event(self, event_type: str, event_data: Dict):
        """Trigger event handlers for specified event type"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def get_connection_info(self, session_id: str) -> Optional[Dict]:
        """
        Get connection information for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Connection information dictionary or None if not found
        """
        if session_id not in self.connections:
            return None
        
        return self.connections[session_id].get_stats()
    
    def get_all_connections(self) -> Dict[str, Dict]:
        """Get information about all active connections"""
        return {
            session_id: connection.get_stats()
            for session_id, connection in self.connections.items()
        }
    
    def get_metrics(self) -> Dict:
        """Get connection manager performance metrics"""
        return {
            **self.metrics,
            "active_connections": len(self.connections),
            "queued_sessions": len(self.message_queue.queues),
            "total_queued_messages": sum(len(queue) for queue in self.message_queue.queues.values()),
            "rate_limit_sessions": len(self.rate_limiter.message_times)
        }
    
    async def disconnect_all(self):
        """Disconnect all active connections (used during shutdown)"""
        logger.info(f"Disconnecting all {len(self.connections)} active connections")
        
        disconnect_tasks = []
        for session_id in list(self.connections.keys()):
            task = asyncio.create_task(
                self.disconnect(session_id, reason="server_shutdown")
            )
            disconnect_tasks.append(task)
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        # Cancel health monitor task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("All connections disconnected")
    
    def is_connected(self, session_id: str) -> bool:
        """Check if a session is currently connected"""
        return (session_id in self.connections and 
                self.connections[session_id].is_healthy())
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.connections)
    
    async def send_status_update(self, session_id: str, status: str, details: Dict = None) -> bool:
        """
        Send status update to a session.
        
        Args:
            session_id: Target session identifier
            status: Status message
            details: Additional status details
            
        Returns:
            True if status sent successfully, False otherwise
        """
        status_data = {
            "type": MessageType.STATUS.value,
            "payload": {
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return await self.send_message(session_id, status_data)
