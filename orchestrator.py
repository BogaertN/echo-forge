"""
orchestrator.py - Complete debate orchestration with all features
"""
import logging
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Complete orchestrator for managing multi-agent debates, journaling, and all EchoForge features
    """
    
    def __init__(self, manager=None, db_manager=None):
        self.connection_manager = manager
        self.db_manager = db_manager
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.agents = {}
        self.user_stats = {
            'total_debates': 0,
            'total_journal_entries': 0,
            'points': 0,
            'level': 1,
            'streak': 0,
            'badges': ['🌱 Beginner']
        }
        logger.info("Orchestrator initialized")
        
        if self.connection_manager:
            logger.info("✓ Connection manager linked to orchestrator")
    
    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "phase": "initial",
                "original_question": None,
                "clarified_question": None,
                "clarification_responses": [],
                "debate_history": [],
                "current_round": 0,
                "agents_used": [],
                "concepts": [],
                "waiting_for_user_response": False
            }
            logger.info(f"Debate session created: {session_id}")
            
            # Save to database if available
            if self.db_manager:
                try:
                    self.db_manager.create_debate_session(session_id)
                except Exception as e:
                    logger.error(f"Failed to save session to DB: {e}")
        else:
            logger.info(f"Session {session_id} already exists, continuing with existing session")
        
        return self.sessions[session_id]
    
    async def process_message(self, message_type: str, session_id: str, data: Dict[str, Any] = None):
        """Process incoming messages from WebSocket"""
        logger.info(f"Processing message type: {message_type} for session {session_id}")
        session = self.get_or_create_session(session_id)
        
        try:
            if message_type == "start_debate":
                # Only start if not already in progress
                if session["phase"] == "initial":
                    await self.start_clarification(session_id)
                else:
                    logger.info(f"Session {session_id} already in progress (phase: {session['phase']})")
                    
            elif message_type == "user_response":
                await self.handle_user_response(session_id, data)
                
            elif message_type == "request_debate":
                await self.start_debate(session_id, data)
                
            elif message_type == "create_journal_entry":
                await self.handle_journal_entry(session_id, data)
                
            elif message_type == "get_resonance_data":
                await self.send_resonance_data(session_id)
                
            elif message_type == "get_analytics":
                await self.send_analytics(session_id)
                
            elif message_type == "export_data":
                await self.handle_export_request(session_id, data)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send_error(session_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self.send_error(session_id, f"Error processing request: {str(e)}")
    
    async def start_clarification(self, session_id: str):
        """Start the Socratic clarification phase"""
        logger.info(f"Starting clarification for session {session_id}")
        session = self.get_or_create_session(session_id)
        
        # Don't restart if already in clarification
        if session["phase"] != "initial":
            return
            
        session["phase"] = "clarification_start"
        
        try:
            # Import and initialize StoicClarifier
            from agents.stoic_clarifier import StoicClarifier
            clarifier_id = f"clarifier_{session_id}"
            clarifier = StoicClarifier(clarifier_id)
            
            # Store agent reference
            self.agents[clarifier_id] = clarifier
            session["agents_used"].append("StoicClarifier")
            
            # Send initial prompt to user
            initial_message = """🌟 Welcome to EchoForge! I'm your Socratic guide.

I'm here to help you explore your thoughts through structured debate and deep reflection.

**What would you like to think through today?** This could be:
• 🤔 A decision you're considering
• 💭 A belief you want to examine  
• 🧩 A problem you're trying to solve
• 🌍 Any topic you'd like to explore from multiple angles

Please share what's on your mind, and I'll help you clarify and refine your thinking before we dive into a structured debate."""
            
            # Send to frontend
            await self.send_to_session(session_id, {
                "type": "clarifier_response",
                "agent": "StoicClarifier", 
                "content": initial_message,
                "phase": "clarification_start",
                "timestamp": datetime.now().isoformat()
            })
            
            session["waiting_for_user_response"] = True
            logger.info(f"✓ Sent initial clarifier prompt to session {session_id}")
            
        except Exception as e:
            logger.error(f"Clarification error: {str(e)}")
            await self.send_error(session_id, f"Clarification failed: {str(e)}")
    
    async def handle_user_response(self, session_id: str, data: Dict[str, Any]):
        """Handle user responses during any phase"""
        user_input = data.get("content", "").strip()
        if not user_input:
            return
            
        session = self.get_or_create_session(session_id)
        
        logger.info(f"Handling user response in phase: {session['phase']} - Input: {user_input[:50]}...")
        
        if session["phase"] == "clarification_start":
            # First response - store original question and ask clarifying questions
            logger.info(f"Processing original question: {user_input}")
            session["original_question"] = user_input
            session["phase"] = "clarification_questions"
            
            # Update database
            if self.db_manager:
                try:
                    self.db_manager.update_debate_session(session_id, 
                        original_question=user_input, phase="clarification_questions")
                except Exception as e:
                    logger.error(f"DB update failed: {e}")
            
            # Get clarifier and ask questions
            clarifier_key = f"clarifier_{session_id}"
            if clarifier_key in self.agents:
                clarifier = self.agents[clarifier_key]
                
                # Show typing indicator
                await self.send_typing_indicator(session_id, "StoicClarifier", True)
                
                response = await clarifier.clarify_question(user_input)
                
                await self.send_typing_indicator(session_id, "StoicClarifier", False)
                
                # Save agent message to database
                if self.db_manager:
                    try:
                        self.db_manager.save_agent_message(
                            session_id, "StoicClarifier", clarifier_key, response, "clarification"
                        )
                    except Exception as e:
                        logger.error(f"Failed to save agent message: {e}")
                
                await self.send_to_session(session_id, {
                    "type": "clarifier_response",
                    "agent": "StoicClarifier",
                    "content": response,
                    "phase": "clarification_questions",
                    "show_response_input": True,
                    "timestamp": datetime.now().isoformat()
                })
                
                session["waiting_for_user_response"] = True
                logger.info(f"✓ Sent clarifying questions for session {session_id}")
            
        elif session["phase"] == "clarification_questions":
            # User answered clarifying questions - now refine and start debate
            logger.info(f"Processing clarification response: {user_input}")
            session["clarification_responses"].append(user_input)
            session["phase"] = "refining_question"
            
            clarifier_key = f"clarifier_{session_id}"
            if clarifier_key in self.agents:
                clarifier = self.agents[clarifier_key]
                
                await self.send_typing_indicator(session_id, "StoicClarifier", True)
                
                # Combine all user responses
                all_responses = " ".join(session["clarification_responses"])
                refined_response = await clarifier.refine_question(
                    session["original_question"], 
                    all_responses
                )
                
                await self.send_typing_indicator(session_id, "StoicClarifier", False)
                
                session["clarified_question"] = refined_response
                session["phase"] = "ready_for_debate"
                session["waiting_for_user_response"] = False
                
                # Update database
                if self.db_manager:
                    try:
                        self.db_manager.update_debate_session(session_id, 
                            clarified_question=refined_response, phase="ready_for_debate")
                        self.db_manager.save_agent_message(
                            session_id, "StoicClarifier", clarifier_key, refined_response, "refinement"
                        )
                    except Exception as e:
                        logger.error(f"DB update failed: {e}")
                
                await self.send_to_session(session_id, {
                    "type": "question_refined",
                    "agent": "StoicClarifier",
                    "content": refined_response,
                    "original_question": session["original_question"],
                    "phase": "ready_for_debate",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Automatically start debate after brief pause
                logger.info(f"Starting auto-debate for session {session_id}")
                await asyncio.sleep(2)
                await self.start_debate(session_id, {"question": refined_response})
    
    async def start_debate(self, session_id: str, data: Dict[str, Any]):
        """Start the multi-agent debate phase"""
        session = self.get_or_create_session(session_id)
        question = data.get("question", session.get("clarified_question", "No question provided"))
        
        session["phase"] = "debate"
        session["current_round"] = 1
        
        # Extract clean question for debate
        if "Refined question for debate:" in question:
            clean_question = question.split("Refined question for debate:")[-1].strip()
        else:
            clean_question = question
        
        logger.info(f"Starting debate for session {session_id} with question: {clean_question[:100]}...")
        
        try:
            await self.send_to_session(session_id, {
                "type": "debate_started",
                "question": clean_question,
                "phase": "debate",
                "timestamp": datetime.now().isoformat()
            })
            
            # Add concepts for resonance mapping
            concepts = self.extract_concepts(clean_question)
            session["concepts"].extend(concepts)
            
            # Simulate multi-agent debate with enhanced content
            await self.conduct_debate_round(session_id, clean_question)
            
        except Exception as e:
            logger.error(f"Debate error: {str(e)}")
            await self.send_error(session_id, f"Debate failed: {str(e)}")
    
    async def conduct_debate_round(self, session_id: str, question: str):
        """Conduct a complete debate round with multiple agents"""
        session = self.sessions[session_id]
        
        # Proponent Agent Response
        await self.send_typing_indicator(session_id, "Proponent", True)
        await asyncio.sleep(2)  # Simulate thinking time
        
        proponent_response = await self.generate_agent_response("proponent", question, "argue in favor")
        
        await self.send_typing_indicator(session_id, "Proponent", False)
        
        await self.send_to_session(session_id, {
            "type": "agent_response",
            "agent": "Proponent",
            "content": proponent_response,
            "round": 1,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to database
        if self.db_manager:
            try:
                self.db_manager.save_agent_message(
                    session_id, "Proponent", f"proponent_{session_id}", 
                    proponent_response, "debate_argument", 1
                )
            except Exception as e:
                logger.error(f"Failed to save proponent message: {e}")
        
        await asyncio.sleep(2)
        
        # Opponent Agent Response  
        await self.send_typing_indicator(session_id, "Opponent", True)
        await asyncio.sleep(2)
        
        opponent_response = await self.generate_agent_response("opponent", question, "argue against")
        
        await self.send_typing_indicator(session_id, "Opponent", False)
        
        await self.send_to_session(session_id, {
            "type": "agent_response", 
            "agent": "Opponent",
            "content": opponent_response,
            "round": 1,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to database
        if self.db_manager:
            try:
                self.db_manager.save_agent_message(
                    session_id, "Opponent", f"opponent_{session_id}", 
                    opponent_response, "debate_argument", 1
                )
            except Exception as e:
                logger.error(f"Failed to save opponent message: {e}")
        
        await asyncio.sleep(2)
        
        # Synthesizer Response
        await self.send_typing_indicator(session_id, "Synthesizer", True)
        await asyncio.sleep(3)
        
        synthesis_response = await self.generate_synthesis(question, proponent_response, opponent_response)
        
        await self.send_typing_indicator(session_id, "Synthesizer", False)
        
        await self.send_to_session(session_id, {
            "type": "synthesis",
            "agent": "Synthesizer",
            "content": synthesis_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Complete the debate
        await self.complete_debate(session_id)
    
    async def generate_agent_response(self, agent_type: str, question: str, instruction: str) -> str:
        """Generate response using Ollama for debate agents"""
        try:
            # Import here to avoid circular imports
            import httpx
            
            system_prompts = {
                "proponent": f"""You are a skilled debater arguing in favor of positions. Your role is to:

1. Present strong, evidence-based arguments supporting the position
2. Use logical reasoning and real-world examples
3. Address potential counterarguments proactively  
4. Be persuasive but respectful
5. Keep responses focused and well-structured (3-4 paragraphs max)

Question to argue FOR: {question}""",

                "opponent": f"""You are a skilled debater presenting counter-arguments. Your role is to:

1. Present strong arguments that challenge the position
2. Point out potential flaws, risks, or negative consequences
3. Use evidence and logical reasoning
4. Offer alternative perspectives and solutions
5. Be respectful but thorough in your critique (3-4 paragraphs max)

Question to argue AGAINST: {question}"""
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "llama3.1:8b",
                        "messages": [
                            {"role": "system", "content": system_prompts[agent_type]},
                            {"role": "user", "content": f"Present your {agent_type} argument for: {question}"}
                        ],
                        "stream": False
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("message", {}).get("content", f"No {agent_type} response generated")
                    return content
                else:
                    return f"Error generating {agent_type} response: HTTP {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error generating {agent_type} response: {e}")
            return f"Error generating {agent_type} response: {str(e)}"
    
    async def generate_synthesis(self, question: str, proponent_arg: str, opponent_arg: str) -> str:
        """Generate synthesis using Ollama"""
        try:
            import httpx
            
            system_prompt = """You are a wise synthesizer who finds common ground and integrates different perspectives. Your role is to:

1. Identify valid points from both sides
2. Find areas of agreement and common ground
3. Suggest balanced approaches that address concerns from both perspectives
4. Provide actionable insights and recommendations
5. Help the user think more deeply about the nuanced reality

Be thoughtful, balanced, and help the user see the complexity while providing clear guidance."""
            
            synthesis_prompt = f"""Question: {question}

PROPONENT ARGUMENT:
{proponent_arg}

OPPONENT ARGUMENT:  
{opponent_arg}

Please provide a thoughtful synthesis that integrates both perspectives and helps the user gain deeper insight into this question."""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "llama3.1:8b", 
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        "stream": False
                    },
                    timeout=90.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "No synthesis generated")
                else:
                    return f"Error generating synthesis: HTTP {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return f"Error generating synthesis: {str(e)}"
    
    async def complete_debate(self, session_id: str):
        """Complete the debate and update statistics"""
        session = self.sessions[session_id]
        session["phase"] = "complete"
        
        # Update user statistics
        self.user_stats['total_debates'] += 1
        self.user_stats['points'] += 50
        
        # Check for level up
        level_threshold = self.user_stats['level'] * 100
        if self.user_stats['points'] >= level_threshold:
            self.user_stats['level'] += 1
            self.user_stats['badges'].append(f"🏆 Level {self.user_stats['level']}")
        
        # Create auto-journal entry
        await self.create_auto_journal_entry(session_id)
        
        # Send completion message
        await self.send_to_session(session_id, {
            "type": "debate_complete",
            "session_summary": {
                "original_question": session.get("original_question"),
                "clarified_question": session.get("clarified_question"), 
                "rounds_completed": 1,
                "agents_used": session.get("agents_used", []),
                "concepts": session.get("concepts", [])
            },
            "user_stats": self.user_stats,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Debate completed for session {session_id}")
    
    async def create_auto_journal_entry(self, session_id: str):
        """Create automatic journal entry from completed debate"""
        session = self.sessions[session_id]
        
        entry_content = f"""# Debate Reflection: {session.get('original_question', 'Unknown Topic')}

## Original Question
{session.get('original_question', 'N/A')}

## Refined Question  
{session.get('clarified_question', 'N/A')}

## Key Insights
After exploring this topic through structured debate, consider:

• What new perspectives did you gain?
• Which arguments were most compelling? 
• How has your thinking evolved?
• What questions remain for further exploration?

## Next Steps
Based on this exploration, what actions or further research might be valuable?

---
*Auto-generated from debate session {session_id} on {datetime.now().strftime('%Y-%m-%d')}*
"""
        
        # Save journal entry
        if self.db_manager:
            try:
                self.db_manager.create_journal_entry(
                    content=entry_content,
                    session_id=session_id,
                    title=f"Debate Reflection: {session.get('original_question', 'Topic')[:50]}...",
                    metadata=json.dumps({
                        "auto_generated": True,
                        "debate_session": session_id,
                        "agents_used": session.get("agents_used", [])
                    })
                )
                self.user_stats['total_journal_entries'] += 1
                logger.info(f"Auto-journal entry created for session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to create auto-journal entry: {e}")
    
    async def handle_journal_entry(self, session_id: str, data: Dict[str, Any]):
        """Handle manual journal entry creation"""
        try:
            title = data.get('title', 'Untitled Entry')
            content = data.get('content', '')
            tags = data.get('tags', [])
            
            if self.db_manager:
                entry_id = self.db_manager.create_journal_entry(
                    content=content,
                    session_id=session_id, 
                    title=title,
                    metadata=json.dumps({"tags": tags, "manual": True})
                )
                
                self.user_stats['total_journal_entries'] += 1
                self.user_stats['points'] += 20
                
                await self.send_to_session(session_id, {
                    "type": "journal_entry_saved",
                    "entry_id": entry_id,
                    "message": "Journal entry saved successfully!",
                    "user_stats": self.user_stats
                })
                
        except Exception as e:
            logger.error(f"Error saving journal entry: {e}")
            await self.send_error(session_id, f"Failed to save journal entry: {str(e)}")
    
    async def send_resonance_data(self, session_id: str):
        """Send resonance mapping data to frontend"""
        session = self.sessions.get(session_id, {})
        concepts = session.get("concepts", [])
        
        # Create nodes and links for resonance map
        nodes = []
        links = []
        
        # Add session node
        nodes.append({
            "id": session_id,
            "name": "Current Session",
            "type": "session",
            "size": 3
        })
        
        # Add concept nodes
        for i, concept in enumerate(concepts):
            concept_id = f"concept_{i}"
            nodes.append({
                "id": concept_id,
                "name": concept,
                "type": "concept", 
                "size": 1
            })
            
            # Link to session
            links.append({
                "source": session_id,
                "target": concept_id,
                "strength": 1
            })
        
        await self.send_to_session(session_id, {
            "type": "resonance_data",
            "nodes": nodes,
            "links": links
        })
    
    async def send_analytics(self, session_id: str):
        """Send analytics data to frontend"""
        analytics_data = {
            "user_stats": self.user_stats,
            "session_stats": {
                "total_sessions": len(self.sessions),
                "active_sessions": sum(1 for s in self.sessions.values() if s.get("phase") != "complete")
            },
            "recent_activity": list(self.sessions.keys())[-10:]  # Last 10 sessions
        }
        
        await self.send_to_session(session_id, {
            "type": "analytics_data",
            "data": analytics_data
        })
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for resonance mapping"""
        # Simple concept extraction - can be enhanced with NLP
        words = text.lower().split()
        # Filter out common words and short words
        concepts = [word for word in words 
                   if len(word) > 4 and word not in ['should', 'would', 'could', 'might', 'about', 'through', 'between']]
        return concepts[:5]  # Return top 5 concepts
    
    async def send_typing_indicator(self, session_id: str, agent_name: str, is_typing: bool):
        """Send typing indicator to frontend"""
        if self.connection_manager:
            await self.connection_manager.send_to_session(session_id, {
                "type": "typing_indicator",
                "agent": agent_name,
                "is_typing": is_typing,
                "timestamp": datetime.now().isoformat()
            })
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to session through connection manager"""
        if self.connection_manager:
            await self.connection_manager.send_to_session(session_id, message)
        else:
            logger.warning(f"No connection manager available for session {session_id}")
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error message to session"""
        await self.send_to_session(session_id, {
            "type": "error",
            "content": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        return self.sessions.get(session_id)
    
    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Clean up associated agents
        keys_to_remove = [key for key in self.agents.keys() if session_id in key]
        for key in keys_to_remove:
            del self.agents[key]
        
        logger.info(f"Session {session_id} cleaned up")
