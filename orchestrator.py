#!/usr/bin/env python3
"""
Orchestrator for EchoForge.
Coordinates multi-agent debates and manages conversation flow.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import EchoForge components
try:
    from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
    from agents.stoic_clarifier import StoicClarifier
    from db import DatabaseManager
except ImportError as e:
    logging.error(f"Failed to import required components: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DebateSession:
    """Represents an active debate session."""
    session_id: str
    original_question: str
    current_phase: str = "clarification"  # clarification, debate, synthesis, complete
    participants: List[str] = None
    round_count: int = 0
    max_rounds: int = 6
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []

class Orchestrator:
    """
    Main orchestrator for EchoForge debates.
    Manages the flow from clarification through debate to synthesis.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.active_sessions: Dict[str, DebateSession] = {}
        self.db_manager = None
        self.connection_manager = None
        
        # Initialize database if available
        try:
            self.db_manager = DatabaseManager()
        except Exception as e:
            logger.warning(f"Database manager not available: {e}")
        
        logger.info("Orchestrator initialized")
    
    def set_connection_manager(self, connection_manager):
        """Set the connection manager for WebSocket communications."""
        self.connection_manager = connection_manager
        logger.info("✓ Connection manager linked to orchestrator")
    
    async def start_clarification(self, question: str, config: AgentConfig, session_id: str):
        """
        Start the clarification phase of a debate.
        
        Args:
            question: The user's initial question
            config: Agent configuration
            session_id: Session identifier
        """
        try:
            logger.info(f"Starting clarification for session {session_id}")
            
            # Create debate session
            session = DebateSession(
                session_id=session_id,
                original_question=question,
                current_phase="clarification"
            )
            self.active_sessions[session_id] = session
            
            # Save session to database
            if self.db_manager:
                self.db_manager.create_debate_session(session_id, question)
            
            # Create StoicClarifier agent
            clarifier = StoicClarifier(config)
            
            # Get clarification
            response = await clarifier.clarify_question(question)
            
            # Save agent response to database
            if self.db_manager:
                self.db_manager.save_agent_message(
                    session_id=session_id,
                    agent_name="StoicClarifier",
                    agent_type="clarifier",
                    content=response.content,
                    confidence=response.confidence
                )
            
            # Send response via WebSocket if available
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "agent_response",
                    "agent": "clarifier", 
                    "phase": "clarification",
                    "content": response.content,
                    "confidence": response.confidence,
                    "metadata": response.metadata
                }, session_id)
                logger.info(f"✓ Sent clarifier response to frontend for session {session_id}")
            else:
                logger.warning("⚠️ Connection manager not available - response not sent to frontend")
            
            logger.info(f"Clarification completed for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in clarification for session {session_id}: {e}")
            
            # Send error message via WebSocket
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "error",
                    "phase": "clarification",
                    "message": f"Error during clarification: {str(e)}"
                }, session_id)
            
            raise e
    
    async def process_clarification_response(self, user_response: str, session_id: str):
        """
        Process user's response to clarification questions.
        
        Args:
            user_response: User's response to clarification
            session_id: Session identifier
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"No active session found for {session_id}")
            
            logger.info(f"Processing clarification response for session {session_id}")
            
            # Create clarifier agent with session config
            config = AgentConfig(session_id=session_id)
            clarifier = StoicClarifier(config)
            
            # Process the response
            response = await clarifier.process_clarification_response(
                user_response, session.original_question
            )
            
            # Save response to database
            if self.db_manager:
                self.db_manager.save_agent_message(
                    session_id=session_id,
                    agent_name="StoicClarifier",
                    agent_type="clarifier",
                    content=response.content,
                    confidence=response.confidence
                )
            
            # Check if clarification is complete
            if response.metadata.get("clarification_stage") == "completed":
                session.current_phase = "ready_for_debate"
                
                # Send completion message
                if self.connection_manager:
                    await self.connection_manager.send_personal_message({
                        "type": "clarification_complete",
                        "refined_question": response.content,
                        "ready_for_debate": True
                    }, session_id)
            
            # Send response via WebSocket
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "agent_response",
                    "agent": "clarifier",
                    "phase": "clarification",
                    "content": response.content,
                    "confidence": response.confidence,
                    "metadata": response.metadata
                }, session_id)
                logger.info(f"✓ Sent clarification follow-up to frontend for session {session_id}")
            else:
                logger.warning("⚠️ Connection manager not available - response not sent to frontend")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing clarification response for {session_id}: {e}")
            raise e
    
    async def start_debate(self, refined_question: str, session_id: str):
        """
        Start the main debate phase.
        
        Args:
            refined_question: The clarified question
            session_id: Session identifier
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"No active session found for {session_id}")
            
            logger.info(f"Starting debate for session {session_id}")
            session.current_phase = "debate"
            
            # For now, send a placeholder response
            # TODO: Implement full debate logic with Proponent/Opponent agents
            
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "debate_started",
                    "phase": "debate",
                    "message": f"Debate started for refined question: {refined_question}",
                    "note": "Full debate implementation coming soon!"
                }, session_id)
                logger.info(f"✓ Sent debate start notification for session {session_id}")
            else:
                logger.warning("⚠️ Connection manager not available - debate start not sent to frontend")
            
        except Exception as e:
            logger.error(f"Error starting debate for {session_id}: {e}")
            raise e
    
    async def handle_user_message(self, message_data: Dict[str, Any], session_id: str):
        """
        Handle incoming user messages and route to appropriate handlers.
        
        Args:
            message_data: Message data from WebSocket
            session_id: Session identifier
        """
        try:
            message_type = message_data.get('type')
            logger.info(f"Processing message type: {message_type} for session {session_id}")
            
            if message_type == 'start_debate':
                question = message_data.get('question', '')
                config = AgentConfig(session_id=session_id)
                await self.start_clarification(question, config, session_id)
                
            elif message_type == 'clarification_response':
                user_response = message_data.get('response', '')
                await self.process_clarification_response(user_response, session_id)
                
            elif message_type == 'start_main_debate':
                refined_question = message_data.get('refined_question', '')
                await self.start_debate(refined_question, session_id)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
                # Send unknown message type response
                if self.connection_manager:
                    await self.connection_manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, session_id)
                
        except Exception as e:
            logger.error(f"Error handling user message for {session_id}: {e}")
            
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "error",
                    "message": f"Error processing your message: {str(e)}"
                }, session_id)
            else:
                logger.error("⚠️ Connection manager not available - error not sent to frontend")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a debate session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "original_question": session.original_question,
            "current_phase": session.current_phase,
            "participants": session.participants,
            "round_count": session.round_count,
            "max_rounds": session.max_rounds
        }
    
    def end_session(self, session_id: str):
        """End a debate session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Update database
            if self.db_manager:
                self.db_manager.update_session_status(session_id, "completed")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} ended")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions (run periodically)."""
        # TODO: Implement session timeout logic
        pass


async def main():
    """Main function for testing orchestrator."""
    print("EchoForge Orchestrator Test")
    print("=" * 40)
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Test session creation
    test_question = "Should AI be regulated?"
    test_session_id = "test_session_123"
    
    config = AgentConfig(session_id=test_session_id)
    
    print(f"Testing with question: '{test_question}'")
    print(f"Session ID: {test_session_id}")
    
    try:
        # Test clarification
        response = await orchestrator.start_clarification(test_question, config, test_session_id)
        print(f"Clarification response: {response.content[:150]}...")
        
        # Test session status
        status = orchestrator.get_session_status(test_session_id)
        print(f"Session status: {status}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Orchestrator test completed")


if __name__ == "__main__":
    asyncio.run(main())
