import logging
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from agents import AgentPool, ProponentAgent, OpponentAgent, SpecialistAgent
from connection_manager import ConnectionManager, DebateUpdateManager
from encrypted_db import EncryptedDB
from journaling import JournalManager
from resonance_map import ResonanceMap
from gamification import GamificationEngine
from tools import ToolManager
from config import load_config
from utils import log_error, generate_timestamp

logger = logging.getLogger(__name__)

class EchoForgeOrchestrator:
    """
    Core orchestrator for EchoForge: Manages end-to-end flows including
    clarification, debates, synthesis, journaling, resonance mapping,
    and gamification. Integrates all components with real-time updates.
    """
    
    def __init__(self):
        self.config = load_config()
        self.agent_pool = AgentPool()
        self.db = EncryptedDB(passphrase=self.config['db']['passphrase'])
        self.connection_manager = ConnectionManager()
        self.debate_update_manager = DebateUpdateManager(self.connection_manager)
        self.journal_manager = JournalManager(self.db)
        self.resonance_map = ResonanceMap(self.db)
        self.gamification = GamificationEngine(self.db)
        self.tool_manager = ToolManager(self.config['tools'])
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        
        logger.info("EchoForgeOrchestrator initialized")
    
    def shutdown(self):
        """Clean shutdown of orchestrator"""
        self.active_sessions.clear()
        logger.info("Orchestrator shutdown complete")
    
    # Session Management
    
    def start_new_session(self, initial_question: str) -> str:
        """Start a new user session"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'initial_question': initial_question,
            'clarification_history': [],
            'clarified_prompt': None,
            'debate_id': None,
            'started_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        logger.info(f"New session started: {session_id}")
        return session_id
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get status of a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        raise ValueError(f"Session not found: {session_id}")
    
    # Clarification Flow
    
    async def clarify_question(self, session_id: str, initial_question: str) -> Dict:
        """Start question clarification process"""
        clarifier = self.agent_pool.get_or_create_agent(
            'clarifier', 
            self.config['models']['clarifier']
        )
        
        clarifier_question = clarifier.start_clarification(initial_question)
        
        # Update session
        self.active_sessions[session_id]['clarification_history'].append({
            'question': clarifier_question
        })
        
        # Send real-time update
        await self.connection_manager.send_to_session(session_id, {
            'type': 'clarification_question',
            'question': clarifier_question
        })
        
        return {
            'status': 'success',
            'clarifier_question': clarifier_question
        }
    
    async def continue_clarification(self, session_id: str, user_response: str, history: List[Dict]) -> Dict:
        """Continue clarification dialogue"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        clarifier = self.agent_pool.get_or_create_agent(
            'clarifier', 
            self.config['models']['clarifier']
        )
        
        result = clarifier.continue_clarification(user_response, history)
        
        # Update history
        self.active_sessions[session_id]['clarification_history'][-1]['response'] = user_response
        
        if result['complete']:
            clarified_prompt = result['clarified_prompt']
            self.active_sessions[session_id]['clarified_prompt'] = clarified_prompt
            
            # Send update
            await self.connection_manager.send_to_session(session_id, {
                'type': 'clarification_complete',
                'clarified_prompt': clarified_prompt
            })
            
            return {
                'status': 'complete',
                'clarified_prompt': clarified_prompt
            }
        else:
            next_question = result['next_question']
            self.active_sessions[session_id]['clarification_history'].append({
                'question': next_question
            })
            
            # Send update
            await self.connection_manager.send_to_session(session_id, {
                'type': 'clarification_question',
                'question': next_question
            })
            
            return {
                'status': 'continue',
                'next_question': next_question
            }
    
    # Debate Flow
    
    async def start_debate(self, session_id: str, config: Dict) -> Dict:
        """Initialize a debate session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        clarified_prompt = self.active_sessions[session_id].get('clarified_prompt')
        if not clarified_prompt:
            raise ValueError("Clarification must be complete before starting debate")
        
        debate_id = self.db.save_debate(session_id, clarified_prompt, config)
        self.active_sessions[session_id]['debate_id'] = debate_id
        
        # Initialize agents
        proponent = self.agent_pool.get_or_create_agent(
            'proponent', 
            self.config['models']['proponent']
        )
        opponent = self.agent_pool.get_or_create_agent(
            'opponent', 
            self.config['models']['opponent']
        )
        
        specialists = {}
        for spec_type in config.get('specialists', []):
            specialists[spec_type] = self.agent_pool.get_or_create_agent(
                'specialist',
                self.config['models']['specialist'],
                specialist_type=spec_type
            )
        
        # Start debate updates
        await self.debate_update_manager.start_debate_updates(session_id, debate_id, config)
        
        logger.info(f"Debate started: {debate_id}")
        
        return {
            'status': 'success',
            'debate_id': debate_id
        }
    
    async def run_debate_round(self, debate_id: str, round_num: int) -> Dict:
        """Execute a single debate round with real-time updates"""
        # Get session and config
        session = next((s for s in self.active_sessions.values() if s['debate_id'] == debate_id), None)
        if not session:
            raise ValueError(f"Debate not found: {debate_id}")
        
        session_id = session['session_id']  # Assuming stored in session
        config = self.db.get_debate_config(debate_id)  # Add method to EncryptedDB if needed
        
        await self.debate_update_manager.send_agent_thinking(debate_id, 'proponent', 'thinking')
        proponent_arg = self.agent_pool.get_or_create_agent('proponent', self.config['models']['proponent']).generate_argument(session['clarified_prompt'])
        
        if config['enable_tools'] and isinstance(proponent_arg, ProponentAgent) and proponent_arg.should_use_tools(proponent_arg):
            query = proponent_arg.extract_search_query(proponent_arg)
            tool_data = await self.tool_manager.execute_tool('web_search', query)
            proponent_arg = proponent_arg.enhance_argument_with_data(proponent_arg, tool_data)
        
        await self.debate_update_manager.send_agent_thinking(debate_id, 'opponent', 'thinking')
        opponent_arg = self.agent_pool.get_or_create_agent('opponent', self.config['models']['opponent']).generate_argument(session['clarified_prompt'])
        
        if config['enable_tools'] and isinstance(opponent_arg, OpponentAgent) and opponent_arg.should_use_tools(opponent_arg):
            query = opponent_arg.extract_search_query(opponent_arg)
            tool_data = await self.tool_manager.execute_tool('web_search', query)
            opponent_arg = opponent_arg.enhance_argument_with_data(opponent_arg, tool_data)
        
        specialist_args = {}
        for spec_type in config.get('specialists', []):
            await self.debate_update_manager.send_agent_thinking(debate_id, spec_type, 'thinking')
            specialist = self.agent_pool.get_or_create_agent('specialist', self.config['models']['specialist'], specialist_type=spec_type)
            specialist_args[spec_type] = specialist.generate_argument(session['clarified_prompt'])
        
        round_data = {
            'round': round_num,
            'arguments': {
                'proponent': {'content': proponent_arg},
                'opponent': {'content': opponent_arg},
                **{k: {'content': v} for k, v in specialist_args.items()}
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Update transcript in DB
        self.db.append_to_transcript(debate_id, round_data)  # Add method to EncryptedDB
        
        # Send real-time update
        await self.debate_update_manager.send_round_update(debate_id, round_num, round_data)
        
        return round_data
    
    async def synthesize_debate(self, debate_id: str, tone: str = "neutral") -> Dict:
        """Synthesize completed debate"""
        transcript = self.db.get_debate_transcript(debate_id)
        clarified_prompt = self.db.get_clarified_prompt(debate_id)  # Add method
        
        auditor = self.agent_pool.get_or_create_agent('auditor', self.config['models']['auditor'])
        auditor_findings = auditor.analyze_debate(json.dumps(transcript))
        
        decider = self.agent_pool.get_or_create_agent('decider', self.config['models']['decider'])
        synthesis = decider.synthesize_debate(clarified_prompt, transcript, auditor_findings, tone)
        
        self.db.finalize_debate(debate_id, synthesis, auditor_findings)
        
        # Complete updates
        await self.debate_update_manager.complete_debate_updates(debate_id, {
            'synthesis': synthesis,
            'auditor_findings': auditor_findings
        })
        
        return {
            'status': 'success',
            'synthesis': synthesis,
            'auditor_findings': auditor_findings
        }
    
    # Journaling Integration
    
    async def create_journal_entry(self, session_id: str, synthesis_data: Dict, user_edits: Optional[str]) -> Dict:
        """Create journal entry from synthesis"""
        journaling_assistant = self.agent_pool.get_or_create_agent(
            'journaling_assistant',
            self.config['models']['journaling_assistant']
        )
        
        content = journaling_assistant.rephrase_for_journal(synthesis_data['content'])
        metadata = journaling_assistant.generate_metadata(content, {'debate_id': self.active_sessions[session_id]['debate_id']})
        
        entry_id = self.journal_manager.create_entry(
            content=content,
            metadata=metadata,
            session_id=session_id,
            debate_id=self.active_sessions[session_id]['debate_id'],
            user_edits=user_edits
        )
        
        # Update resonance map
        node_id = self.resonance_map.add_node('entry', metadata['summary'])
        if metadata['ghost_loop']:
            self.resonance_map.add_edge(node_id, 'ghost_loops', 'belongs_to')  # Assuming ghost_loops node
        
        # Update gamification
        self.gamification.update_on_entry()
        
        # Send update
        await self.connection_manager.send_to_session(session_id, {
            'type': 'journal_entry_created',
            'entry_id': entry_id,
            'metadata': metadata
        })
        
        return {
            'status': 'success',
            'entry_id': entry_id
        }
    
    def search_journal(self, query: str) -> List[Dict]:
        """Search journal entries"""
        return self.journal_manager.search_entries(query)
    
    # Gamification
    
    def get_gamification_stats(self) -> Dict:
        """Get gamification stats"""
        return self.gamification.get_stats()
    
    # Tool Integration (Optional)
    
    async def execute_tool_in_debate(self, debate_id: str, tool_name: str, params: Dict) -> Any:
        """Execute tool during debate if enabled"""
        config = self.db.get_debate_config(debate_id)
        if not config['enable_tools']:
            raise PermissionError("Tools not enabled for this debate")
        
        result = await self.tool_manager.execute_tool(tool_name, **params)
        return result
