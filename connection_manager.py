import logging
import json
import asyncio
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import uuid

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates during EchoForge sessions.
    Provides session-based routing and broadcast capabilities.
    """
    
    def __init__(self):
        # Active WebSocket connections by session_id
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Connection metadata for tracking
        self.connection_metadata: Dict[str, Dict] = {}
        
        # Message queue for offline sessions (optional feature)
        self.message_queue: Dict[str, List[Dict]] = {}
        
        # Statistics tracking
        self.stats = {
            'total_connections': 0,
            'current_connections': 0,
            'messages_sent': 0,
            'connection_errors': 0
        }
        
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[session_id] = websocket
            
            # Store metadata
            self.connection_metadata[session_id] = {
                'connected_at': datetime.now().isoformat(),
                'session_id': session_id,
                'client_info': self._extract_client_info(websocket),
                'message_count': 0,
                'last_activity': datetime.now().isoformat()
            }
            
            # Update statistics
            self.stats['total_connections'] += 1
            self.stats['current_connections'] += 1
            
            logger.info(f"WebSocket connected for session {session_id}")
            
            # Send connection confirmation
            await self.send_to_session(session_id, {
                'type': 'connection_established',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'server_info': {
                    'version': '1.0.0',
                    'capabilities': ['real_time_debates', 'voice_transcription', 'journal_updates']
                }
            })
            
            # Send any queued messages
            await self._send_queued_messages(session_id)
            
        except Exception as e:
            logger.error(f"Connection failed for session {session_id}: {str(e)}")
            self.stats['connection_errors'] += 1
            raise
    
    def disconnect(self, session_id: str):
        """Disconnect and cleanup a WebSocket session"""
        if session_id in self.active_connections:
            try:
                # Update metadata
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]['disconnected_at'] = datetime.now().isoformat()
                
                # Remove active connection
                del self.active_connections[session_id]
                
                # Update statistics
                self.stats['current_connections'] -= 1
                
                logger.info(f"WebSocket disconnected for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error during disconnect for session {session_id}: {str(e)}")
                self.stats['connection_errors'] += 1
    
    async def send_to_session(self, session_id: str, message: Dict):
        """Send a message to a specific session"""
        if session_id in self.active_connections:
            try:
                # Add message metadata
                enriched_message = {
                    **message,
                    'message_id': str(uuid.uuid4()),
                    'timestamp': message.get('timestamp', datetime.now().isoformat()),
                    'session_id': session_id
                }
                
                # Send message
                await self.active_connections[session_id].send_text(
                    json.dumps(enriched_message)
                )
                
                # Update metadata and stats
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]['message_count'] += 1
                    self.connection_metadata[session_id]['last_activity'] = datetime.now().isoformat()
                
                self.stats['messages_sent'] += 1
                
                logger.debug(f"Message sent to session {session_id}: {message.get('type', 'unknown')}")
                
            except WebSocketDisconnect:
                logger.warning(f"WebSocket disconnected during send for session {session_id}")
                self.disconnect(session_id)
                
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {str(e)}")
                self.stats['connection_errors'] += 1
                
                # Try to disconnect cleanly
                self.disconnect(session_id)
        else:
            # Queue message for offline session (optional feature)
            await self._queue_message(session_id, message)
    
    async def broadcast_to_all(self, message: Dict, exclude_sessions: Set[str] = None):
        """Broadcast message to all active sessions"""
        exclude_sessions = exclude_sessions or set()
        
        broadcast_tasks = []
        for session_id in list(self.active_connections.keys()):
            if session_id not in exclude_sessions:
                broadcast_tasks.append(
                    self.send_to_session(session_id, {
                        **message,
                        'broadcast': True
                    })
                )
        
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            logger.info(f"Broadcast sent to {len(broadcast_tasks)} sessions")
    
    async def send_to_multiple_sessions(self, session_ids: List[str], message: Dict):
        """Send message to multiple specific sessions"""
        send_tasks = []
        for session_id in session_ids:
            if session_id in self.active_connections:
                send_tasks.append(self.send_to_session(session_id, message))
        
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
            logger.info(f"Message sent to {len(send_tasks)} sessions")
    
    async def _queue_message(self, session_id: str, message: Dict):
        """Queue message for offline session (optional feature)"""
        if session_id not in self.message_queue:
            self.message_queue[session_id] = []
        
        # Add timestamp if not present
        message['queued_at'] = datetime.now().isoformat()
        
        self.message_queue[session_id].append(message)
        
        # Limit queue size to prevent memory issues
        max_queue_size = 50
        if len(self.message_queue[session_id]) > max_queue_size:
            self.message_queue[session_id] = self.message_queue[session_id][-max_queue_size:]
        
        logger.debug(f"Message queued for offline session {session_id}")
    
    async def _send_queued_messages(self, session_id: str):
        """Send any queued messages when session reconnects"""
        if session_id in self.message_queue and self.message_queue[session_id]:
            queued_messages = self.message_queue[session_id]
            del self.message_queue[session_id]
            
            # Send queued messages
            for message in queued_messages:
                message['type'] = f"queued_{message.get('type', 'message')}"
                await self.send_to_session(session_id, message)
            
            logger.info(f"Sent {len(queued_messages)} queued messages to session {session_id}")
    
    def _extract_client_info(self, websocket: WebSocket) -> Dict:
        """Extract client information from WebSocket headers"""
        try:
            headers = dict(websocket.headers)
            return {
                'user_agent': headers.get('user-agent', 'Unknown'),
                'host': headers.get('host', 'Unknown'),
                'origin': headers.get('origin', 'Unknown'),
                'client_ip': getattr(websocket.client, 'host', 'Unknown') if websocket.client else 'Unknown'
            }
        except Exception as e:
            logger.warning(f"Failed to extract client info: {str(e)}")
            return {'error': 'Failed to extract client info'}
    
    def get_active_sessions(self) -> List[str]:
        """Get list of currently active session IDs"""
        return list(self.active_connections.keys())
    
    def get_active_count(self) -> int:
        """Get count of active connections"""
        return len(self.active_connections)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is currently active"""
        return session_id in self.active_connections
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get metadata for a specific session"""
        return self.connection_metadata.get(session_id)
    
    def get_connection_stats(self) -> Dict:
        """Get comprehensive connection statistics"""
        active_sessions = list(self.active_connections.keys())
        
        return {
            **self.stats,
            'active_sessions': active_sessions,
            'active_session_count': len(active_sessions),
            'queued_message_sessions': list(self.message_queue.keys()),
            'total_queued_messages': sum(len(queue) for queue in self.message_queue.values()),
            'uptime_info': {
                'manager_started': getattr(self, '_start_time', datetime.now().isoformat()),
                'current_time': datetime.now().isoformat()
            }
        }
    
    async def ping_all_connections(self):
        """Send ping to all connections to check health"""
        ping_message = {
            'type': 'ping',
            'timestamp': datetime.now().isoformat()
        }
        
        # Track failed pings for cleanup
        failed_sessions = []
        
        for session_id in list(self.active_connections.keys()):
            try:
                await self.send_to_session(session_id, ping_message)
            except Exception as e:
                logger.warning(f"Ping failed for session {session_id}: {str(e)}")
                failed_sessions.append(session_id)
        
        # Cleanup failed connections
        for session_id in failed_sessions:
            self.disconnect(session_id)
        
        if failed_sessions:
            logger.info(f"Cleaned up {len(failed_sessions)} stale connections")
    
    async def disconnect_all(self):
        """Disconnect all active connections (used during shutdown)"""
        disconnect_tasks = []
        
        for session_id, websocket in list(self.active_connections.items()):
            try:
                # Send shutdown notification
                await websocket.send_text(json.dumps({
                    'type': 'server_shutdown',
                    'message': 'Server is shutting down',
                    'timestamp': datetime.now().isoformat()
                }))
                
                # Close connection
                await websocket.close()
                
            except Exception as e:
                logger.warning(f"Error during shutdown disconnect for {session_id}: {str(e)}")
            finally:
                self.disconnect(session_id)
        
        logger.info("All WebSocket connections disconnected")
    
    async def send_system_notification(self, message: str, notification_type: str = "info", 
                                     target_sessions: Optional[List[str]] = None):
        """Send system-wide notification to all or specific sessions"""
        notification = {
            'type': 'system_notification',
            'notification_type': notification_type,  # info, warning, error, success
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'system_generated': True
        }
        
        if target_sessions:
            await self.send_to_multiple_sessions(target_sessions, notification)
        else:
            await self.broadcast_to_all(notification)
        
        logger.info(f"System notification sent: {notification_type} - {message}")
    
    async def handle_client_message(self, session_id: str, message: Dict):
        """Handle incoming message from client"""
        try:
            message_type = message.get('type', 'unknown')
            
            # Update last activity
            if session_id in self.connection_metadata:
                self.connection_metadata[session_id]['last_activity'] = datetime.now().isoformat()
            
            # Handle different message types
            if message_type == 'ping':
                await self.send_to_session(session_id, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif message_type == 'keep_alive':
                # Just update activity timestamp
                pass
                
            elif message_type == 'client_info':
                # Update client metadata
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]['client_info'].update(
                        message.get('data', {})
                    )
                
            elif message_type == 'request_status':
                # Send session status
                await self.send_to_session(session_id, {
                    'type': 'session_status',
                    'data': self.get_session_metadata(session_id),
                    'active_sessions': self.get_active_count()
                })
                
            else:
                # Echo unknown messages back (for debugging)
                await self.send_to_session(session_id, {
                    'type': 'echo',
                    'original_message': message,
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.debug(f"Handled client message: {message_type} from session {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling client message from {session_id}: {str(e)}")
            await self.send_to_session(session_id, {
                'type': 'error',
                'message': 'Failed to process message',
                'timestamp': datetime.now().isoformat()
            })

class DebateUpdateManager:
    """
    Specialized manager for debate-related WebSocket updates.
    Works with ConnectionManager to provide debate-specific functionality.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_debates: Dict[str, Dict] = {}
        
    async def start_debate_updates(self, session_id: str, debate_id: str, config: Dict):
        """Initialize debate update tracking"""
        self.active_debates[debate_id] = {
            'session_id': session_id,
            'config': config,
            'started_at': datetime.now().isoformat(),
            'round_count': 0,
            'last_update': datetime.now().isoformat()
        }
        
        await self.connection_manager.send_to_session(session_id, {
            'type': 'debate_updates_started',
            'debate_id': debate_id,
            'config': config
        })
    
    async def send_round_update(self, debate_id: str, round_num: int, round_data: Dict):
        """Send real-time update for debate round completion"""
        if debate_id in self.active_debates:
            session_id = self.active_debates[debate_id]['session_id']
            
            # Update tracking
            self.active_debates[debate_id]['round_count'] = round_num
            self.active_debates[debate_id]['last_update'] = datetime.now().isoformat()
            
            await self.connection_manager.send_to_session(session_id, {
                'type': 'debate_round_update',
                'debate_id': debate_id,
                'round': round_num,
                'data': round_data,
                'progress': {
                    'current_round': round_num,
                    'total_rounds': self.active_debates[debate_id]['config'].get('rounds', 3)
                }
            })
    
    async def send_agent_thinking(self, debate_id: str, agent_role: str, status: str):
        """Send agent thinking status (for UX feedback)"""
        if debate_id in self.active_debates:
            session_id = self.active_debates[debate_id]['session_id']
            
            await self.connection_manager.send_to_session(session_id, {
                'type': 'agent_thinking',
                'debate_id': debate_id,
                'agent': agent_role,
                'status': status,  # 'thinking', 'generating', 'complete'
                'timestamp': datetime.now().isoformat()
            })
    
    async def complete_debate_updates(self, debate_id: str, final_data: Dict):
        """Send debate completion notification"""
        if debate_id in self.active_debates:
            session_id = self.active_debates[debate_id]['session_id']
            
            await self.connection_manager.send_to_session(session_id, {
                'type': 'debate_completed',
                'debate_id': debate_id,
                'final_data': final_data,
                'summary': {
                    'total_rounds': self.active_debates[debate_id]['round_count'],
                    'duration': self._calculate_duration(self.active_debates[debate_id]['started_at'])
                }
            })
            
            # Clean up tracking
            del self.active_debates[debate_id]
    
    def _calculate_duration(self, start_time: str) -> str:
        """Calculate debate duration"""
        try:
            start = datetime.fromisoformat(start_time)
            duration = datetime.now() - start
            return str(duration.total_seconds())
        except:
            return "unknown"
    
    def get_active_debates(self) -> List[str]:
        """Get list of currently active debate IDs"""
        return list(self.active_debates.keys())
