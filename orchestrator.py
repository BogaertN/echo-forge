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
    from agents.proponent import ProponentAgent
    from agents.opponent import OpponentAgent
    from agents.synthesizer import SynthesizerAgent
    from db import DatabaseManager
except ImportError as e:
    logging.error(f"Failed to import required components: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DebateSession:
    """Represents an active debate session."""
    session_id: str
    original_question: str
    refined_question: str = ""
    current_phase: str = "clarification"  # clarification, debate, synthesis, complete
    participants: List[str] = None
    round_count: int = 0
    max_rounds: int = 3
    debate_arguments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.debate_arguments is None:
            self.debate_arguments = []

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
            
            # Check if session already exists
            existing_session = self.active_sessions.get(session_id)
            if existing_session:
                logger.info(f"Session {session_id} already exists, continuing with existing session")
                session = existing_session
            else:
                # Create debate session
                session = DebateSession(
                    session_id=session_id,
                    original_question=question,
                    current_phase="clarification"
                )
                self.active_sessions[session_id] = session
                
                # Save session to database (only if new)
                if self.db_manager:
                    try:
                        self.db_manager.create_debate_session(session_id, question)
                    except Exception as e:
                        logger.warning(f"Session {session_id} may already exist in database: {e}")
            
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
                session.refined_question = response.content
                
                # Send completion message and auto-start debate
                if self.connection_manager:
                    await self.connection_manager.send_personal_message({
                        "type": "clarification_complete",
                        "refined_question": response.content,
                        "ready_for_debate": True
                    }, session_id)
                
                # Auto-start the debate with the refined question
                await asyncio.sleep(1)  # Brief pause for UI
                await self.start_debate(response.content, session_id)
            else:
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
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing clarification response for {session_id}: {e}")
            raise e
    
    async def start_debate(self, refined_question: str, session_id: str):
        """
        Start the main debate phase with Proponent vs Opponent.
        
        Args:
            refined_question: The clarified question
            session_id: Session identifier
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"No active session found for {session_id}")
            
            logger.info(f"Starting multi-agent debate for session {session_id}")
            session.current_phase = "debate"
            session.refined_question = refined_question
            
            # Create debate agents
            config = AgentConfig(session_id=session_id)
            proponent = ProponentAgent(config)
            opponent = OpponentAgent(config)
            
            # Send debate start notification
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "debate_started",
                    "phase": "debate",
                    "message": f"Starting multi-agent debate on: {refined_question}",
                    "participants": ["Proponent", "Opponent"]
                }, session_id)
            
            # Round 1: Proponent opens
            logger.info(f"Debate Round 1: Proponent opening for session {session_id}")
            proponent_opening = await proponent.make_opening_argument(refined_question)
            session.debate_arguments.append({
                "round": 1,
                "agent": "proponent",
                "content": proponent_opening.content,
                "confidence": proponent_opening.confidence
            })
            
            # Save and send proponent opening
            if self.db_manager:
                self.db_manager.save_agent_message(session_id, "Proponent", "proponent", 
                                                 proponent_opening.content, proponent_opening.confidence)
            
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "agent_response",
                    "agent": "proponent",
                    "phase": "debate",
                    "round": 1,
                    "content": proponent_opening.content,
                    "confidence": proponent_opening.confidence,
                    "metadata": proponent_opening.metadata
                }, session_id)
            
            await asyncio.sleep(2)  # Pause between responses
            
            # Round 1: Opponent responds
            logger.info(f"Debate Round 1: Opponent response for session {session_id}")
            opponent_response = await opponent.make_opening_argument(refined_question, proponent_opening.content)
            session.debate_arguments.append({
                "round": 1,
                "agent": "opponent", 
                "content": opponent_response.content,
                "confidence": opponent_response.confidence
            })
            
            # Save and send opponent response
            if self.db_manager:
                self.db_manager.save_agent_message(session_id, "Opponent", "opponent",
                                                 opponent_response.content, opponent_response.confidence)
            
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "agent_response",
                    "agent": "opponent",
                    "phase": "debate", 
                    "round": 1,
                    "content": opponent_response.content,
                    "confidence": opponent_response.confidence,
                    "metadata": opponent_response.metadata
                }, session_id)
            
            session.round_count = 1
            
            # Continue with additional rounds
            for round_num in range(2, session.max_rounds + 1):
                await asyncio.sleep(2)
                
                # Proponent rebuttal
                logger.info(f"Debate Round {round_num}: Proponent rebuttal for session {session_id}")
                proponent_rebuttal = await proponent.make_rebuttal(opponent_response.content, refined_question)
                session.debate_arguments.append({
                    "round": round_num,
                    "agent": "proponent",
                    "content": proponent_rebuttal.content,
                    "confidence": proponent_rebuttal.confidence
                })
                
                if self.db_manager:
                    self.db_manager.save_agent_message(session_id, "Proponent", "proponent",
                                                     proponent_rebuttal.content, proponent_rebuttal.confidence)
                
                if self.connection_manager:
                    await self.connection_manager.send_personal_message({
                        "type": "agent_response",
                        "agent": "proponent",
                        "phase": "debate",
                        "round": round_num,
                        "content": proponent_rebuttal.content,
                        "confidence": proponent_rebuttal.confidence,
                        "metadata": proponent_rebuttal.metadata
                    }, session_id)
                
                await asyncio.sleep(2)
                
                # Opponent rebuttal
                logger.info(f"Debate Round {round_num}: Opponent rebuttal for session {session_id}")
                opponent_rebuttal = await opponent.make_rebuttal(proponent_rebuttal.content, refined_question)
                session.debate_arguments.append({
                    "round": round_num,
                    "agent": "opponent",
                    "content": opponent_rebuttal.content,
                    "confidence": opponent_rebuttal.confidence
                })
                
                if self.db_manager:
                    self.db_manager.save_agent_message(session_id, "Opponent", "opponent",
                                                     opponent_rebuttal.content, opponent_rebuttal.confidence)
                
                if self.connection_manager:
                    await self.connection_manager.send_personal_message({
                        "type": "agent_response",
                        "agent": "opponent",
                        "phase": "debate",
                        "round": round_num,
                        "content": opponent_rebuttal.content,
                        "confidence": opponent_rebuttal.confidence,
                        "metadata": opponent_rebuttal.metadata
                    }, session_id)
                
                session.round_count = round_num
                
                # Update opponent_response for next round
                opponent_response = opponent_rebuttal
            
            # Debate completed, move to synthesis
            logger.info(f"Debate completed for session {session_id}, starting synthesis")
            await self.start_synthesis(session_id)
            
        except Exception as e:
            logger.error(f"Error starting debate for {session_id}: {e}")
            raise e
    
    async def start_synthesis(self, session_id: str):
        """
        Start the synthesis phase to find common ground.
        
        Args:
            session_id: Session identifier
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"No active session found for {session_id}")
            
            logger.info(f"Starting synthesis for session {session_id}")
            session.current_phase = "synthesis"
            
            # Create synthesizer agent
            config = AgentConfig(session_id=session_id)
            synthesizer = SynthesizerAgent(config)
            
            # Extract arguments by side
            proponent_args = [arg["content"] for arg in session.debate_arguments if arg["agent"] == "proponent"]
            opponent_args = [arg["content"] for arg in session.debate_arguments if arg["agent"] == "opponent"]
            
            # Generate synthesis
            synthesis_response = await synthesizer.synthesize_debate(
                session.refined_question, proponent_args, opponent_args
            )
            
            # Save synthesis
            if self.db_manager:
                self.db_manager.save_agent_message(session_id, "Synthesizer", "synthesizer",
                                                 synthesis_response.content, synthesis_response.confidence)
            
            # Send synthesis to frontend
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "agent_response",
                    "agent": "synthesizer",
                    "phase": "synthesis", 
                    "content": synthesis_response.content,
                    "confidence": synthesis_response.confidence,
                    "metadata": synthesis_response.metadata
                }, session_id)
            
            # Mark session as complete
            session.current_phase = "complete"
            
            # Send completion notification
            if self.connection_manager:
                await self.connection_manager.send_personal_message({
                    "type": "debate_complete",
                    "phase": "complete",
                    "message": "Debate and synthesis completed! You can now reflect on the insights gained.",
                    "summary": {
                        "question": session.refined_question,
                        "rounds": session.round_count,
                        "arguments": len(session.debate_arguments)
                    }
                }, session_id)
            
            logger.info(f"✓ Synthesis completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in synthesis for {session_id}: {e}")
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
                # Treat any other message as a clarification response if we're in that phase
                session = self.active_sessions.get(session_id)
                if session and session.current_phase == "clarification":
                    user_response = message_data.get('question', message_data.get('message', ''))
                    if user_response:
                        await self.process_clarification_response(user_response, session_id)
                    else:
                        logger.warning(f"No valid response content in message: {message_data}")
                else:
                    logger.warning(f"Unknown message type: {message_type} for session in phase: {session.current_phase if session else 'none'}")
                
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
            "refined_question": session.refined_question,
            "current_phase": session.current_phase,
            "participants": session.participants,
            "round_count": session.round_count,
            "max_rounds": session.max_rounds,
            "arguments_count": len(session.debate_arguments)
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
    print("EchoForge Multi-Agent Orchestrator Test")
    print("=" * 50)
    
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
        
        # Simulate clarification completion and auto-start debate
        clarification_complete_response = "I want to explore whether government regulation of artificial intelligence development and deployment should be implemented."
        await orchestrator.process_clarification_response(clarification_complete_response, test_session_id)
        
        # Test session status
        status = orchestrator.get_session_status(test_session_id)
        print(f"Final session status: {status}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Multi-agent orchestrator test completed")


if __name__ == "__main__":
    asyncio.run(main())
