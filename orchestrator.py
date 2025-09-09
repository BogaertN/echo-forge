import asyncio
import json
import logging
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback

from agents.base_agent  import BaseAgent
from agents.stoic_clarifier  import StoicClarifier
from agents.proponent_agent  import ProponentAgent
from agents.opponent_agent  import OpponentAgent
from agents.decider_agent  import DeciderAgent
from agents.auditor_agent  import AuditorAgent
from agents.journaling_assistant  import JournalingAssistant
from agents.specialist_agent  import SpecialistAgent
from connection_manager import ConnectionManager
from models import *
from utils import generate_uuid, calculate_similarity, extract_key_concepts

logger = logging.getLogger(__name__)

class DebatePhase(Enum):
    """Debate phases for orchestration"""
    CLARIFICATION = "clarification"
    OPENING_STATEMENTS = "opening_statements"
    MAIN_DEBATE = "main_debate"
    SPECIALIST_INPUT = "specialist_input"
    SYNTHESIS = "synthesis"
    AUDIT = "audit"
    JOURNALING = "journaling"
    COMPLETE = "complete"

class AgentRole(Enum):
    """Agent roles in the system"""
    CLARIFIER = "clarifier"
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    SPECIALIST = "specialist"
    DECIDER = "decider"
    AUDITOR = "auditor"
    JOURNALING_ASSISTANT = "journaling_assistant"

@dataclass
class DebateConfig:
    """Configuration for debate sessions"""
    max_rounds: int = 6
    round_timeout: int = 120  # seconds
    enable_specialists: bool = True
    specialist_domains: List[str] = None
    tone_modifier: str = "balanced"  # balanced, gentle, analytical, creative
    tools_enabled: bool = False
    fact_checking: bool = True
    ghost_loop_detection: bool = True
    synthesis_style: str = "comprehensive"  # brief, comprehensive, creative
    
    def __post_init__(self):
        if self.specialist_domains is None:
            self.specialist_domains = ["ethics", "logic", "practical", "emotional"]

@dataclass
class SessionState:
    """State tracking for debate sessions"""
    session_id: str
    phase: DebatePhase
    current_round: int
    question: str
    clarified_question: str
    debate_history: List[Dict]
    agent_contexts: Dict[str, Dict]
    tools_used: List[str]
    ghost_loops_detected: List[Dict]
    performance_metrics: Dict
    created_at: datetime
    last_update: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "phase": self.phase.value,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat()
        }

class EchoForgeOrchestrator:
    """
    Core orchestration engine for EchoForge multi-agent system.
    
    Manages agent lifecycle, debate flows, context isolation, and system coordination.
    """
    
    def __init__(self, db, journal_manager, resonance_manager, tool_manager):
        self.db = db
        self.journal_manager = journal_manager
        self.resonance_manager = resonance_manager
        self.tool_manager = tool_manager
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize agents
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self._initialize_agents()
        
        # Session management
        self.active_sessions: Dict[str, SessionState] = {}
        self.agent_instances: Dict[str, Dict[AgentRole, BaseAgent]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "sessions_completed": 0,
            "average_session_duration": 0,
            "ghost_loops_detected": 0,
            "synthesis_quality_score": 0.0,
            "agent_response_times": {},
            "error_rates": {}
        }
        
        logger.info("EchoForge Orchestrator initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load agent routing and model configuration"""
        try:
            config_path = Path("configs/agent_routing.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Agent routing config not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if config file not found"""
        return {
            "models": {
                "clarifier": "gemma2:2b",
                "proponent": "llama3.1:8b",
                "opponent": "llama3.1:8b",
                "specialist": "llama3.1:8b",
                "decider": "llama3.1:8b",
                "auditor": "gemma2:2b",
                "journaling_assistant": "gemma2:2b"
            },
            "agent_settings": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "timeout": 60
            },
            "debate_defaults": {
                "max_rounds": 6,
                "specialist_trigger_threshold": 0.7,
                "ghost_loop_threshold": 0.8,
                "synthesis_complexity": "balanced"
            }
        }
    
    def _initialize_agents(self):
        """Initialize agent templates (not instances)"""
        try:
            self.agents = {
                AgentRole.CLARIFIER: StoicClarifier(
                    model=self.config["models"]["clarifier"],
                    **self.config["agent_settings"]
                ),
                AgentRole.PROPONENT: ProponentAgent(
                    model=self.config["models"]["proponent"],
                    **self.config["agent_settings"]
                ),
                AgentRole.OPPONENT: OpponentAgent(
                    model=self.config["models"]["opponent"],
                    **self.config["agent_settings"]
                ),
                AgentRole.DECIDER: DeciderAgent(
                    model=self.config["models"]["decider"],
                    **self.config["agent_settings"]
                ),
                AgentRole.AUDITOR: AuditorAgent(
                    model=self.config["models"]["auditor"],
                    **self.config["agent_settings"]
                ),
                AgentRole.JOURNALING_ASSISTANT: JournalingAssistant(
                    model=self.config["models"]["journaling_assistant"],
                    **self.config["agent_settings"]
                )
            }
            logger.info("Agent templates initialized")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def _create_session_agents(self, session_id: str) -> Dict[AgentRole, BaseAgent]:
        """Create isolated agent instances for a session"""
        session_agents = {}
        
        for role, template_agent in self.agents.items():
            # Create new instance with isolated context
            session_agents[role] = template_agent.__class__(
                model=template_agent.model,
                temperature=template_agent.temperature,
                max_tokens=template_agent.max_tokens,
                timeout=template_agent.timeout,
                session_id=session_id
            )
        
        return session_agents
    
    async def start_clarification(self, question: str, session_id: str) -> AsyncGenerator[Dict, None]:
        """
        Start Socratic clarification process.
        
        Args:
            question: Initial user question
            session_id: Session identifier
            
        Yields:
            Real-time updates about clarification progress
        """
        try:
            # Initialize session
            session_state = SessionState(
                session_id=session_id,
                phase=DebatePhase.CLARIFICATION,
                current_round=0,
                question=question,
                clarified_question="",
                debate_history=[],
                agent_contexts={},
                tools_used=[],
                ghost_loops_detected=[],
                performance_metrics={},
                created_at=datetime.now(),
                last_update=datetime.now()
            )
            
            self.active_sessions[session_id] = session_state
            
            # Create session-specific agents
            session_agents = self._create_session_agents(session_id)
            self.agent_instances[session_id] = session_agents
            
            # Get clarifier agent
            clarifier = session_agents[AgentRole.CLARIFIER]
            
            yield {
                "type": "clarification_update",
                "payload": {
                    "phase": "starting",
                    "message": "Beginning Socratic clarification process..."
                }
            }
            
            # Start clarification
            clarification_complete = False
            clarification_rounds = 0
            max_clarification_rounds = 5
            
            current_question = question
            clarification_history = []
            
            while not clarification_complete and clarification_rounds < max_clarification_rounds:
                clarification_rounds += 1
                
                # Get clarification question from agent
                start_time = time.time()
                
                clarification_response = await clarifier.get_clarification_question(
                    question=current_question,
                    history=clarification_history
                )
                
                response_time = time.time() - start_time
                self._update_performance_metrics("clarifier", response_time)
                
                # Check if clarification is complete
                if clarification_response.get("clarification_complete", False):
                    clarification_complete = True
                    session_state.clarified_question = clarification_response.get("final_question", current_question)
                    
                    yield {
                        "type": "clarification_complete",
                        "payload": {
                            "clarified_question": session_state.clarified_question,
                            "rounds": clarification_rounds,
                            "summary": clarification_response.get("summary", "")
                        }
                    }
                else:
                    # Send clarification question to user
                    clarification_question = clarification_response.get("question", "")
                    
                    yield {
                        "type": "clarification_question",
                        "payload": {
                            "question": clarification_question,
                            "round": clarification_rounds,
                            "context": clarification_response.get("context", ""),
                            "suggestions": clarification_response.get("suggestions", [])
                        }
                    }
                    
                    # Wait for user response (handled in continue_clarification)
                    break
            
            # Update session state
            session_state.last_update = datetime.now()
            session_state.current_round = clarification_rounds
            
        except Exception as e:
            logger.error(f"Error in clarification: {e}")
            yield {
                "type": "error",
                "payload": {"message": str(e), "traceback": traceback.format_exc()}
            }
    
    async def continue_clarification(self, user_response: str, session_id: str) -> AsyncGenerator[Dict, None]:
        """
        Continue clarification process with user response.
        
        Args:
            user_response: User's response to clarification question
            session_id: Session identifier
            
        Yields:
            Real-time updates about clarification progress
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session_state = self.active_sessions[session_id]
            clarifier = self.agent_instances[session_id][AgentRole.CLARIFIER]
            
            # Process user response
            clarification_response = await clarifier.process_clarification_response(
                user_response=user_response,
                session_context=session_state.agent_contexts.get("clarifier", {})
            )
            
            # Check if clarification is complete
            if clarification_response.get("clarification_complete", False):
                session_state.clarified_question = clarification_response.get("final_question", session_state.question)
                session_state.phase = DebatePhase.OPENING_STATEMENTS
                
                yield {
                    "type": "clarification_complete",
                    "payload": {
                        "clarified_question": session_state.clarified_question,
                        "summary": clarification_response.get("summary", ""),
                        "key_concepts": clarification_response.get("key_concepts", [])
                    }
                }
            else:
                # Continue with next clarification question
                clarification_question = clarification_response.get("question", "")
                
                yield {
                    "type": "clarification_question",
                    "payload": {
                        "question": clarification_question,
                        "round": session_state.current_round + 1,
                        "context": clarification_response.get("context", ""),
                        "suggestions": clarification_response.get("suggestions", [])
                    }
                }
            
            # Update session state
            session_state.last_update = datetime.now()
            session_state.agent_contexts["clarifier"] = clarification_response.get("context", {})
            
        except Exception as e:
            logger.error(f"Error continuing clarification: {e}")
            yield {
                "type": "error",
                "payload": {"message": str(e)}
            }
    
    async def start_debate(self, question: str, session_id: str, config: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Start multi-agent debate process.
        
        Args:
            question: Clarified question for debate
            session_id: Session identifier
            config: Debate configuration parameters
            
        Yields:
            Real-time updates about debate progress
        """
        try:
            # Get session state
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session_state = self.active_sessions[session_id]
            session_agents = self.agent_instances[session_id]
            
            # Apply configuration
            debate_config = DebateConfig(**(config or {}))
            
            # Update session state
            session_state.phase = DebatePhase.OPENING_STATEMENTS
            session_state.clarified_question = question
            
            yield {
                "type": "debate_started",
                "payload": {
                    "question": question,
                    "config": asdict(debate_config),
                    "estimated_duration": debate_config.max_rounds * 2
                }
            }
            
            # Phase 1: Opening Statements
            async for update in self._run_opening_statements(session_state, session_agents, debate_config):
                yield update
            
            # Phase 2: Main Debate Rounds
            session_state.phase = DebatePhase.MAIN_DEBATE
            async for update in self._run_main_debate(session_state, session_agents, debate_config):
                yield update
            
            # Phase 3: Specialist Input (if enabled)
            if debate_config.enable_specialists:
                session_state.phase = DebatePhase.SPECIALIST_INPUT
                async for update in self._run_specialist_input(session_state, session_agents, debate_config):
                    yield update
            
            # Phase 4: Synthesis
            session_state.phase = DebatePhase.SYNTHESIS
            async for update in self._run_synthesis(session_state, session_agents, debate_config):
                yield update
            
            # Phase 5: Audit
            session_state.phase = DebatePhase.AUDIT
            async for update in self._run_audit(session_state, session_agents, debate_config):
                yield update
            
            # Phase 6: Journaling Assistance
            session_state.phase = DebatePhase.JOURNALING
            async for update in self._run_journaling_assistance(session_state, session_agents, debate_config):
                yield update
            
            # Complete
            session_state.phase = DebatePhase.COMPLETE
            
            yield {
                "type": "debate_complete",
                "payload": {
                    "session_id": session_id,
                    "total_rounds": session_state.current_round,
                    "performance_metrics": session_state.performance_metrics,
                    "summary": "Debate completed successfully"
                }
            }
            
            # Update global performance metrics
            self._update_session_completion_metrics(session_state)
            
        except Exception as e:
            logger.error(f"Error in debate: {e}")
            yield {
                "type": "error",
                "payload": {"message": str(e), "traceback": traceback.format_exc()}
            }
    
    async def _run_opening_statements(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run opening statements phase"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "opening_statements", "description": "Agents preparing opening positions"}
        }
        
        proponent = agents[AgentRole.PROPONENT]
        opponent = agents[AgentRole.OPPONENT]
        
        # Get proponent opening statement
        proponent_start = time.time()
        proponent_statement = await proponent.generate_opening_statement(
            question=session_state.clarified_question,
            tools_enabled=config.tools_enabled,
            tool_manager=self.tool_manager if config.tools_enabled else None
        )
        proponent_time = time.time() - proponent_start
        
        yield {
            "type": "agent_response",
            "payload": {
                "agent": "proponent",
                "role": "opening_statement",
                "content": proponent_statement["statement"],
                "reasoning": proponent_statement.get("reasoning", ""),
                "sources": proponent_statement.get("sources", []),
                "response_time": proponent_time
            }
        }
        
        # Get opponent opening statement
        opponent_start = time.time()
        opponent_statement = await opponent.generate_opening_statement(
            question=session_state.clarified_question,
            proponent_statement=proponent_statement["statement"],
            tools_enabled=config.tools_enabled,
            tool_manager=self.tool_manager if config.tools_enabled else None
        )
        opponent_time = time.time() - opponent_start
        
        yield {
            "type": "agent_response",
            "payload": {
                "agent": "opponent",
                "role": "opening_statement",
                "content": opponent_statement["statement"],
                "reasoning": opponent_statement.get("reasoning", ""),
                "sources": opponent_statement.get("sources", []),
                "response_time": opponent_time
            }
        }
        
        # Update session history
        session_state.debate_history.extend([
            {
                "round": 0,
                "agent": "proponent",
                "type": "opening_statement",
                "content": proponent_statement["statement"],
                "timestamp": datetime.now().isoformat(),
                "response_time": proponent_time
            },
            {
                "round": 0,
                "agent": "opponent",
                "type": "opening_statement",
                "content": opponent_statement["statement"],
                "timestamp": datetime.now().isoformat(),
                "response_time": opponent_time
            }
        ])
        
        # Update performance metrics
        self._update_performance_metrics("proponent", proponent_time)
        self._update_performance_metrics("opponent", opponent_time)
    
    async def _run_main_debate(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run main debate rounds"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "main_debate", "description": f"Starting {config.max_rounds} debate rounds"}
        }
        
        proponent = agents[AgentRole.PROPONENT]
        opponent = agents[AgentRole.OPPONENT]
        
        for round_num in range(1, config.max_rounds + 1):
            session_state.current_round = round_num
            
            yield {
                "type": "round_started",
                "payload": {"round": round_num, "max_rounds": config.max_rounds}
            }
            
            # Proponent response
            proponent_start = time.time()
            
            # Detect ghost loops before response
            ghost_loop_detected = await self._detect_ghost_loops(session_state, "proponent")
            if ghost_loop_detected and config.ghost_loop_detection:
                yield {
                    "type": "ghost_loop_detected",
                    "payload": {
                        "agent": "proponent",
                        "loop_type": ghost_loop_detected["type"],
                        "description": ghost_loop_detected["description"]
                    }
                }
            
            proponent_response = await proponent.generate_debate_response(
                question=session_state.clarified_question,
                debate_history=session_state.debate_history,
                round_number=round_num,
                tools_enabled=config.tools_enabled,
                tool_manager=self.tool_manager if config.tools_enabled else None
            )
            
            proponent_time = time.time() - proponent_start
            
            yield {
                "type": "agent_response",
                "payload": {
                    "agent": "proponent",
                    "round": round_num,
                    "content": proponent_response["response"],
                    "key_points": proponent_response.get("key_points", []),
                    "sources": proponent_response.get("sources", []),
                    "response_time": proponent_time
                }
            }
            
            # Opponent response
            opponent_start = time.time()
            
            # Detect ghost loops before response
            ghost_loop_detected = await self._detect_ghost_loops(session_state, "opponent")
            if ghost_loop_detected and config.ghost_loop_detection:
                yield {
                    "type": "ghost_loop_detected",
                    "payload": {
                        "agent": "opponent",
                        "loop_type": ghost_loop_detected["type"],
                        "description": ghost_loop_detected["description"]
                    }
                }
            
            opponent_response = await opponent.generate_debate_response(
                question=session_state.clarified_question,
                debate_history=session_state.debate_history + [{
                    "round": round_num,
                    "agent": "proponent",
                    "type": "debate_response",
                    "content": proponent_response["response"],
                    "timestamp": datetime.now().isoformat()
                }],
                round_number=round_num,
                tools_enabled=config.tools_enabled,
                tool_manager=self.tool_manager if config.tools_enabled else None
            )
            
            opponent_time = time.time() - opponent_start
            
            yield {
                "type": "agent_response",
                "payload": {
                    "agent": "opponent",
                    "round": round_num,
                    "content": opponent_response["response"],
                    "key_points": opponent_response.get("key_points", []),
                    "sources": opponent_response.get("sources", []),
                    "response_time": opponent_time
                }
            }
            
            # Update session history
            session_state.debate_history.extend([
                {
                    "round": round_num,
                    "agent": "proponent",
                    "type": "debate_response",
                    "content": proponent_response["response"],
                    "timestamp": datetime.now().isoformat(),
                    "response_time": proponent_time
                },
                {
                    "round": round_num,
                    "agent": "opponent",
                    "type": "debate_response",
                    "content": opponent_response["response"],
                    "timestamp": datetime.now().isoformat(),
                    "response_time": opponent_time
                }
            ])
            
            # Update performance metrics
            self._update_performance_metrics("proponent", proponent_time)
            self._update_performance_metrics("opponent", opponent_time)
            
            # Check for early termination conditions
            if await self._should_terminate_early(session_state, round_num):
                yield {
                    "type": "early_termination",
                    "payload": {
                        "reason": "Convergence detected",
                        "round": round_num,
                        "description": "Agents have reached sufficient consensus"
                    }
                }
                break
            
            # Brief pause between rounds
            await asyncio.sleep(0.5)
    
    async def _run_specialist_input(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run specialist input phase"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "specialist_input", "description": "Consulting specialist agents"}
        }
        
        # Determine which specialists to call based on debate content
        specialist_domains = await self._determine_specialist_needs(session_state, config)
        
        for domain in specialist_domains:
            # Create specialist agent for this domain
            specialist = SpecialistAgent(
                model=self.config["models"]["specialist"],
                domain=domain,
                session_id=session_state.session_id,
                **self.config["agent_settings"]
            )
            
            specialist_start = time.time()
            specialist_input = await specialist.provide_specialist_input(
                question=session_state.clarified_question,
                debate_history=session_state.debate_history,
                domain_focus=domain
            )
            specialist_time = time.time() - specialist_start
            
            yield {
                "type": "specialist_input",
                "payload": {
                    "domain": domain,
                    "input": specialist_input["analysis"],
                    "recommendations": specialist_input.get("recommendations", []),
                    "concerns": specialist_input.get("concerns", []),
                    "response_time": specialist_time
                }
            }
            
            # Add to debate history
            session_state.debate_history.append({
                "round": session_state.current_round,
                "agent": f"specialist_{domain}",
                "type": "specialist_input",
                "content": specialist_input["analysis"],
                "timestamp": datetime.now().isoformat(),
                "response_time": specialist_time
            })
    
    async def _run_synthesis(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run synthesis phase"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "synthesis", "description": "Generating balanced synthesis"}
        }
        
        decider = agents[AgentRole.DECIDER]
        
        synthesis_start = time.time()
        synthesis = await decider.generate_synthesis(
            question=session_state.clarified_question,
            debate_history=session_state.debate_history,
            tone_modifier=config.tone_modifier,
            synthesis_style=config.synthesis_style
        )
        synthesis_time = time.time() - synthesis_start
        
        yield {
            "type": "synthesis_generated",
            "payload": {
                "synthesis": synthesis["synthesis"],
                "key_insights": synthesis.get("key_insights", []),
                "balanced_perspective": synthesis.get("balanced_perspective", ""),
                "action_items": synthesis.get("action_items", []),
                "confidence_score": synthesis.get("confidence_score", 0.0),
                "response_time": synthesis_time
            }
        }
        
        # Add to session history
        session_state.debate_history.append({
            "round": session_state.current_round,
            "agent": "decider",
            "type": "synthesis",
            "content": synthesis["synthesis"],
            "timestamp": datetime.now().isoformat(),
            "response_time": synthesis_time
        })
        
        # Store synthesis for journaling
        session_state.agent_contexts["synthesis"] = synthesis
    
    async def _run_audit(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run audit phase"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "audit", "description": "Quality assurance and fact checking"}
        }
        
        auditor = agents[AgentRole.AUDITOR]
        
        audit_start = time.time()
        audit_result = await auditor.audit_debate(
            question=session_state.clarified_question,
            debate_history=session_state.debate_history,
            synthesis=session_state.agent_contexts.get("synthesis", {}),
            fact_checking=config.fact_checking
        )
        audit_time = time.time() - audit_start
        
        yield {
            "type": "audit_complete",
            "payload": {
                "quality_score": audit_result.get("quality_score", 0.0),
                "logical_consistency": audit_result.get("logical_consistency", 0.0),
                "factual_accuracy": audit_result.get("factual_accuracy", 0.0),
                "bias_detection": audit_result.get("bias_detection", []),
                "recommendations": audit_result.get("recommendations", []),
                "red_flags": audit_result.get("red_flags", []),
                "response_time": audit_time
            }
        }
        
        # Store audit results
        session_state.agent_contexts["audit"] = audit_result
    
    async def _run_journaling_assistance(self, session_state: SessionState, agents: Dict, config: DebateConfig) -> AsyncGenerator[Dict, None]:
        """Run journaling assistance phase"""
        yield {
            "type": "phase_started",
            "payload": {"phase": "journaling", "description": "Preparing journal entry"}
        }
        
        journaling_assistant = agents[AgentRole.JOURNALING_ASSISTANT]
        
        journaling_start = time.time()
        journal_entry = await journaling_assistant.prepare_journal_entry(
            question=session_state.clarified_question,
            debate_history=session_state.debate_history,
            synthesis=session_state.agent_contexts.get("synthesis", {}),
            audit_results=session_state.agent_contexts.get("audit", {})
        )
        journaling_time = time.time() - journaling_start
        
        yield {
            "type": "journal_entry_prepared",
            "payload": {
                "title": journal_entry.get("title", ""),
                "summary": journal_entry.get("summary", ""),
                "key_insights": journal_entry.get("key_insights", []),
                "personal_reflections": journal_entry.get("personal_reflections", ""),
                "action_items": journal_entry.get("action_items", []),
                "tags": journal_entry.get("tags", []),
                "metadata": journal_entry.get("metadata", {}),
                "response_time": journaling_time
            }
        }
        
        # Store journal entry for saving
        session_state.agent_contexts["journal_entry"] = journal_entry
    
    async def _detect_ghost_loops(self, session_state: SessionState, agent_name: str) -> Optional[Dict]:
        """Detect ghost loops (repetitive arguments) in debate"""
        if len(session_state.debate_history) < 4:
            return None
        
        # Get recent responses from this agent
        agent_responses = [
            entry for entry in session_state.debate_history[-6:]
            if entry["agent"] == agent_name
        ]
        
        if len(agent_responses) < 3:
            return None
        
        # Check for similarity in content
        recent_contents = [response["content"] for response in agent_responses[-3:]]
        
        for i in range(len(recent_contents) - 1):
            similarity = calculate_similarity(recent_contents[i], recent_contents[i + 1])
            if similarity > 0.8:  # High similarity threshold
                ghost_loop = {
                    "type": "repetitive_argument",
                    "description": f"Agent {agent_name} appears to be repeating similar arguments",
                    "similarity_score": similarity,
                    "detected_at": datetime.now().isoformat()
                }
                session_state.ghost_loops_detected.append(ghost_loop)
                return ghost_loop
        
        return None
    
    async def _should_terminate_early(self, session_state: SessionState, round_num: int) -> bool:
        """Determine if debate should terminate early due to convergence"""
        if round_num < 3:  # Minimum rounds
            return False
        
        # Get recent responses from both agents
        recent_responses = session_state.debate_history[-4:]
        if len(recent_responses) < 4:
            return False
        
        proponent_responses = [r for r in recent_responses if r["agent"] == "proponent"]
        opponent_responses = [r for r in recent_responses if r["agent"] == "opponent"]
        
        if len(proponent_responses) < 2 or len(opponent_responses) < 2:
            return False
        
        # Check for convergence indicators
        # (Simplified - in practice would use more sophisticated NLP)
        convergence_keywords = ["agree", "consensus", "similar", "aligned", "common ground"]
        
        recent_content = " ".join([r["content"] for r in recent_responses])
        convergence_score = sum(1 for keyword in convergence_keywords if keyword in recent_content.lower())
        
        return convergence_score >= 3
    
    async def _determine_specialist_needs(self, session_state: SessionState, config: DebateConfig) -> List[str]:
        """Determine which specialist domains are needed based on debate content"""
        # Extract key concepts from debate
        all_content = " ".join([entry["content"] for entry in session_state.debate_history])
        key_concepts = extract_key_concepts(all_content)
        
        # Map concepts to specialist domains
        domain_mapping = {
            "ethics": ["moral", "ethical", "right", "wrong", "justice", "fairness"],
            "logic": ["logical", "reasoning", "evidence", "proof", "conclusion"],
            "practical": ["implementation", "practical", "feasible", "cost", "resource"],
            "emotional": ["emotional", "feeling", "impact", "wellbeing", "psychological"]
        }
        
        needed_domains = []
        for domain, keywords in domain_mapping.items():
            if any(keyword in key_concepts for keyword in keywords):
                needed_domains.append(domain)
        
        # Limit to available specialist domains
        return [domain for domain in needed_domains if domain in config.specialist_domains][:2]  # Max 2 specialists
    
    def _update_performance_metrics(self, agent_name: str, response_time: float):
        """Update performance metrics for agents"""
        if agent_name not in self.performance_metrics["agent_response_times"]:
            self.performance_metrics["agent_response_times"][agent_name] = []
        
        self.performance_metrics["agent_response_times"][agent_name].append(response_time)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics["agent_response_times"][agent_name]) > 100:
            self.performance_metrics["agent_response_times"][agent_name] = \
                self.performance_metrics["agent_response_times"][agent_name][-100:]
    
    def _update_session_completion_metrics(self, session_state: SessionState):
        """Update metrics when session completes"""
        self.performance_metrics["sessions_completed"] += 1
        
        session_duration = (session_state.last_update - session_state.created_at).total_seconds()
        
        # Update average session duration
        current_avg = self.performance_metrics["average_session_duration"]
        sessions_count = self.performance_metrics["sessions_completed"]
        
        self.performance_metrics["average_session_duration"] = (
            (current_avg * (sessions_count - 1) + session_duration) / sessions_count
        )
        
        # Update ghost loop detection count
        self.performance_metrics["ghost_loops_detected"] += len(session_state.ghost_loops_detected)
    
    async def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get current session state"""
        return self.active_sessions.get(session_id)
    
    async def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.agent_instances:
            # Clean up agent instances
            for agent in self.agent_instances[session_id].values():
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
            del self.agent_instances[session_id]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.config["models"].values())
    
    async def update_model_config(self, new_config: Dict):
        """Update model configuration"""
        self.config.update(new_config)
        
        # Reinitialize agents with new config
        self._initialize_agents()
        
        logger.info("Model configuration updated")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        # Calculate average response times
        avg_response_times = {}
        for agent, times in self.performance_metrics["agent_response_times"].items():
            if times:
                avg_response_times[agent] = sum(times) / len(times)
        
        return {
            **self.performance_metrics,
            "average_response_times": avg_response_times,
            "active_sessions": len(self.active_sessions)
        }
