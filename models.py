from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime
import uuid

# Session Models
class SessionStartRequest(BaseModel):
    initial_question: str = Field(..., description="The initial question or prompt from the user")

class SessionStatus(BaseModel):
    session_id: str
    status: str = Field(..., example="active")
    started_at: str
    clarified_prompt: Optional[str]
    debate_id: Optional[str]

# Clarification Models
class ClarificationRequest(BaseModel):
    session_id: str
    user_response: str
    conversation_history: List[Dict[str, str]]

class ClarificationResponse(BaseModel):
    status: str = Field(..., example="continue")
    next_question: Optional[str]
    clarified_prompt: Optional[str]

# Debate Models
class DebateConfig(BaseModel):
    rounds: int = Field(3, ge=1, le=10, description="Number of debate rounds")
    tone: str = Field("neutral", description="Synthesis tone: neutral, optimistic, cautious, skeptical")
    specialists: List[str] = Field([], description="List of specialist types: ethics, statistics, etc.")
    enable_tools: bool = Field(False, description="Enable tool use for agents")

class DebateStartRequest(BaseModel):
    session_id: str
    config: DebateConfig

class DebateRoundData(BaseModel):
    round: int
    arguments: Dict[str, Dict[str, str]]  # role: {content: str}

class DebateSynthesis(BaseModel):
    synthesis: str
    auditor_findings: Dict[str, Any]

# Journal Models
class JournalMetadata(BaseModel):
    title: str
    tags: List[str]
    weights: Dict[str, int]  # relevance, emotion, priority
    ghost_loop: bool
    ghost_loop_reason: Optional[str]
    summary: str

class JournalEntryCreate(BaseModel):
    content: str
    metadata: JournalMetadata
    session_id: str
    debate_id: Optional[str]
    user_edits: Optional[str]

class JournalEntry(BaseModel):
    id: str
    content: str
    metadata: JournalMetadata
    created_at: str
    updated_at: str
    session_id: str
    debate_id: Optional[str]
    user_edits: Optional[str]

class JournalSearchRequest(BaseModel):
    query: Optional[str] = ""
    tags: Optional[List[str]] = []
    ghost_loops_only: bool = False
    limit: int = 10

# Resonance Map Models
class ResonanceNode(BaseModel):
    id: str
    type: str  # entry, debate, ghost_loop, concept
    content_summary: str
    created_at: str

class ResonanceEdge(BaseModel):
    from_id: str
    to_id: str
    relation_type: str
    strength: float

class ResonanceMapView(BaseModel):
    nodes: Dict[str, ResonanceNode]
    edges: List[ResonanceEdge]
    layout: Optional[Dict[str, Dict[str, float]]]  # x, y positions

# Gamification Models
class GamificationStats(BaseModel):
    streak_count: int
    badges: List[str]
    clarity_metrics: Dict[str, float]  # average, highest, etc.
    weekly_report: Dict[str, Any]

# Tool Models
class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolLog(BaseModel):
    tool_name: str
    session_id: str
    agent_id: str
    inputs: Dict[str, Any]
    output: Any
    timestamp: str

# Voice Models
class VoiceTranscriptionRequest(BaseModel):
    audio_path: str  # Local path to audio file
    session_id: str

class VoiceTranscriptionResponse(BaseModel):
    transcription: str
    language: str
    session_id: str
    timestamp: str

# General Error Model
class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
