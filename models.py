import json
import re
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from uuid import UUID
from typing import Literal

from pydantic import (
    BaseModel, Field, validator, root_validator,
    EmailStr, HttpUrl, ValidationError
)

# === ENUMS AND CONSTANTS ===

class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

class DebatePhase(str, Enum):
    """Debate phase enumeration"""
    CLARIFICATION = "clarification"
    OPENING_STATEMENTS = "opening_statements"
    MAIN_DEBATE = "main_debate"
    SPECIALIST_INPUT = "specialist_input"
    SYNTHESIS = "synthesis"
    AUDIT = "audit"
    JOURNALING = "journaling"
    COMPLETE = "complete"

class AgentRole(str, Enum):
    """Agent role enumeration"""
    CLARIFIER = "clarifier"
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    SPECIALIST = "specialist"
    DECIDER = "decider"
    AUDITOR = "auditor"
    JOURNALING_ASSISTANT = "journaling_assistant"

class MessageType(str, Enum):
    """WebSocket message type enumeration"""
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
    PHASE_STARTED = "phase_started"
    ROUND_STARTED = "round_started"
    AGENT_RESPONSE = "agent_response"
    SPECIALIST_INPUT = "specialist_input"
    SYNTHESIS_GENERATED = "synthesis_generated"
    EARLY_TERMINATION = "early_termination"
    DEBATE_COMPLETE = "debate_complete"
    
    # Quality assurance
    GHOST_LOOP_DETECTED = "ghost_loop_detected"
    AUDIT_COMPLETE = "audit_complete"
    
    # Journal messages
    JOURNAL_ENTRY_PREPARED = "journal_entry_prepared"
    JOURNAL_SAVED = "journal_saved"
    
    # Resonance map
    RESONANCE_MAP_DATA = "resonance_map_data"
    
    # Preferences and config
    PREFERENCES_UPDATED = "preferences_updated"

class ConnectionType(str, Enum):
    """Resonance map connection types"""
    RELATES_TO = "relates_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    LEADS_TO = "leads_to"
    EXPLORES = "explores"
    SYNTHESIZES = "synthesizes"
    QUESTIONS = "questions"
    CLARIFIES = "clarifies"

class NodeType(str, Enum):
    """Resonance map node types"""
    CONCEPT = "concept"
    THEME = "theme"
    QUESTION = "question"
    INSIGHT = "insight"
    DEBATE_POINT = "debate_point"
    SYNTHESIS = "synthesis"

class BadgeRarity(str, Enum):
    """Badge rarity levels"""
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

class ToolCategory(str, Enum):
    """Tool category enumeration"""
    WEB_SEARCH = "web_search"
    FACT_CHECK = "fact_check"
    KNOWLEDGE_BASE = "knowledge_base"
    TRANSLATION = "translation"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    COMPUTATION = "computation"

class ArgumentType(str, Enum):
    """Argument type enumeration"""
    PRO = "pro"
    CON = "con"
    EVIDENCE = "evidence"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"
    QUESTION = "question"
    CLARIFICATION = "clarification"
    AUDIT = "audit"

class SynthesisType(str, Enum):
    """Synthesis type enumeration for debate synthesis styles"""
    BRIEF = "brief"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"

# === REQUEST/RESPONSE MODELS ===

class BaseRequest(BaseModel):
    """Base request model with common fields"""
    session_id: str = Field(..., description="Session identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: Literal[False]  = False
    error_code: Optional[str] = Field(None, description="Error code for categorization")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    traceback: Optional[str] = Field(None, description="Error traceback for debugging")

# === CLARIFICATION MODELS ===

class ClarificationRequest(BaseRequest):
    """Request to start clarification process"""
    question: str = Field(..., min_length=5, max_length=1000, description="Initial question")
    context: Optional[str] = Field(None, max_length=2000, description="Additional context")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class ClarificationResponse(BaseRequest):
    """Response to clarification question"""
    response: str = Field(..., min_length=1, max_length=2000, description="User response")
    skip_clarification: bool = Field(False, description="Skip remaining clarification")

class ClarificationComplete(BaseModel):
    """Clarification completion notification"""
    original_question: str
    clarified_question: str
    rounds_completed: int
    key_concepts: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

# === DEBATE MODELS ===

class DebateConfig(BaseModel):
    """Configuration for debate sessions"""
    max_rounds: int = Field(6, ge=1, le=20, description="Maximum debate rounds")
    round_timeout: int = Field(120, ge=30, le=600, description="Timeout per round in seconds")
    enable_specialists: bool = Field(True, description="Enable specialist agent consultation")
    specialist_domains: List[str] = Field(default_factory=lambda: ["ethics", "logic", "practical", "emotional"])
    tone_modifier: str = Field("balanced", pattern=r"^(balanced|gentle|analytical|creative)$")
    tools_enabled: bool = Field(False, description="Enable external tool integration")
    fact_checking: bool = Field(True, description="Enable fact-checking during debate")
    ghost_loop_detection: bool = Field(True, description="Enable ghost loop detection")
    synthesis_style: str = Field("comprehensive", pattern=r"^(brief|comprehensive|creative)$")
    
    @validator('specialist_domains')
    def validate_domains(cls, v):
        valid_domains = ["ethics", "logic", "practical", "emotional", "legal", "technical", "social"]
        invalid_domains = [d for d in v if d not in valid_domains]
        if invalid_domains:
            raise ValueError(f"Invalid specialist domains: {invalid_domains}")
        return v

class DebateRequest(BaseRequest):
    """Request to start debate"""
    clarified_question: str = Field(..., min_length=10, max_length=500)
    config: Optional[DebateConfig] = Field(default_factory=DebateConfig)

class AgentResponseModel(BaseModel):
    """Agent response structure"""
    agent: AgentRole
    role: str  # opening_statement, debate_response, synthesis, etc.
    round: Optional[int] = None
    content: str = Field(..., min_length=10)
    reasoning: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    response_time: Optional[float] = Field(None, ge=0.0)
    token_count: Optional[int] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SpecialistInput(BaseModel):
    """Specialist agent input"""
    domain: str
    analysis: str
    recommendations: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    response_time: float = Field(..., ge=0.0)

class SynthesisResult(BaseModel):
    """Synthesis generation result"""
    synthesis: str = Field(..., min_length=50)
    key_insights: List[str] = Field(default_factory=list)
    balanced_perspective: Optional[str] = None
    action_items: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    tone_modifier: str
    synthesis_style: str
    response_time: float = Field(..., ge=0.0)

class AuditResult(BaseModel):
    """Audit result for quality assurance"""
    quality_score: float = Field(..., ge=0.0, le=1.0)
    logical_consistency: float = Field(..., ge=0.0, le=1.0)
    factual_accuracy: float = Field(..., ge=0.0, le=1.0)
    bias_detection: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    issues_found: List[Dict[str, Any]] = Field(default_factory=list)
    response_time: float = Field(..., ge=0.0)

class GhostLoopDetection(BaseModel):
    """Ghost loop detection result"""
    loop_type: str
    nodes: List[str]
    strength: float = Field(..., ge=0.0, le=1.0)
    description: str
    detected_at: datetime

# === JOURNAL MODELS ===

class JournalEntryRequest(BaseRequest):
    """Request to create journal entry"""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10, max_length=50000)
    insights: Optional[List[str]] = Field(None, max_items=20)
    personal_reflections: Optional[str] = Field(None, max_length=10000)
    action_items: Optional[List[str]] = Field(None, max_items=10)
    tags: Optional[List[str]] = Field(None, max_items=20)
    mood_rating: Optional[int] = Field(None, ge=1, le=10)
    complexity_rating: Optional[int] = Field(None, ge=1, le=10)
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=10)
    debate_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is None:
            return v
        # Clean and validate tags
        cleaned_tags = []
        for tag in v:
            if isinstance(tag, str) and len(tag.strip()) > 0:
                cleaned_tag = re.sub(r'[^\w\s-]', '', tag.strip().lower())
                if cleaned_tag and len(cleaned_tag) <= 50:
                    cleaned_tags.append(cleaned_tag)
        return cleaned_tags[:20]  # Limit to 20 tags

class JournalEntryUpdate(BaseModel):
    """Request to update journal entry"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=10, max_length=50000)
    insights: Optional[List[str]] = Field(None, max_items=20)
    personal_reflections: Optional[str] = Field(None, max_length=10000)
    action_items: Optional[List[str]] = Field(None, max_items=10)
    tags: Optional[List[str]] = Field(None, max_items=20)
    mood_rating: Optional[int] = Field(None, ge=1, le=10)
    complexity_rating: Optional[int] = Field(None, ge=1, le=10)
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None

class JournalSearchRequest(BaseModel):
    """Journal search request"""
    session_id: str
    text: Optional[str] = Field(None, max_length=200)
    tags: Optional[List[str]] = Field(None, max_items=10)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    mood_range: Optional[Tuple[int, int]] = Field(None)
    complexity_range: Optional[Tuple[int, int]] = Field(None)
    satisfaction_range: Optional[Tuple[int, int]] = Field(None)
    has_debate: Optional[bool] = None
    min_word_count: Optional[int] = Field(None, ge=0)
    max_word_count: Optional[int] = Field(None, ge=1)
    sort_by: str = Field("created_at", pattern=r"^(created_at|updated_at|word_count|title|mood_rating)$")
    sort_order: str = Field("desc", pattern=r"^(asc|desc)$")
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    
    @validator('mood_range', 'complexity_range', 'satisfaction_range')
    def validate_ranges(cls, v):
        if v is None:
            return v
        if len(v) != 2 or v[0] > v[1] or v[0] < 1 or v[1] > 10:
            raise ValueError("Range must be tuple of (min, max) with values 1-10")
        return v

class JournalEntryResponse(BaseModel):
    """Journal entry response"""
    id: str
    session_id: str
    debate_id: Optional[str]
    title: str
    content: str
    summary: str
    insights: List[str]
    personal_reflections: str
    action_items: List[str]
    tags: List[str]
    mood_rating: Optional[int]
    complexity_rating: Optional[int]
    satisfaction_rating: Optional[int]
    word_count: int
    reading_time_minutes: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

# === RESONANCE MAP MODELS ===

class ResonanceNodeModel(BaseModel):
    """Resonance map node model"""
    id: str
    session_id: str
    node_type: NodeType
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    strength: float = Field(..., ge=0.0, le=1.0)
    frequency: int = Field(..., ge=1)
    last_activated: datetime
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResonanceConnectionModel(BaseModel):
    """Resonance map connection model"""
    id: str
    session_id: str
    source_node_id: str
    target_node_id: str
    connection_type: ConnectionType
    strength: float = Field(..., ge=0.0, le=1.0)
    created_from: str
    created_at: datetime
    last_strengthened: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResonanceMapRequest(BaseModel):
    """Request for resonance map data"""
    session_id: str
    node_types: Optional[List[NodeType]] = None
    connection_types: Optional[List[ConnectionType]] = None
    min_strength: float = Field(0.1, ge=0.0, le=1.0)
    min_connection_strength: float = Field(0.1, ge=0.0, le=1.0)
    max_nodes: int = Field(100, ge=10, le=500)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    depth: int = Field(2, ge=1, le=5)

class VisualizationNode(BaseModel):
    """Node for visualization"""
    id: str
    title: str
    type: NodeType
    strength: float
    frequency: int
    description: str
    x: float
    y: float
    size: float
    centrality: float

class VisualizationEdge(BaseModel):
    """Edge for visualization"""
    source: str
    target: str
    type: ConnectionType
    strength: float
    width: float
    opacity: float
    created_from: str

class VisualizationCluster(BaseModel):
    """Cluster for visualization"""
    id: str
    nodes: List[str]
    central_concept: str
    strength: float
    density: float
    center_x: float
    center_y: float
    node_count: int

class ResonanceMapResponse(BaseModel):
    """Resonance map visualization response"""
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    clusters: List[VisualizationCluster]
    metrics: Dict[str, Any]
    ghost_loops: List[Dict[str, Any]]
    has_data: bool
    generated_at: datetime

# === GAMIFICATION MODELS ===

class BadgeModel(BaseModel):
    """Badge definition model"""
    id: str
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    category: str = Field(..., min_length=1, max_length=50)
    icon: Optional[str] = None
    requirements: Dict[str, Any]
    points_reward: int = Field(..., ge=1, le=10000)
    rarity: BadgeRarity
    created_at: datetime

class UserBadge(BaseModel):
    """User badge achievement"""
    id: str
    session_id: str
    badge_id: str
    earned_at: datetime
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GamificationStats(BaseModel):
    """User gamification statistics"""
    session_id: str
    total_debates: int = Field(..., ge=0)
    total_journal_entries: int = Field(..., ge=0)
    total_questions_clarified: int = Field(..., ge=0)
    current_streak_days: int = Field(..., ge=0)
    longest_streak_days: int = Field(..., ge=0)
    last_activity_date: Optional[date]
    total_words_written: int = Field(..., ge=0)
    total_insights_generated: int = Field(..., ge=0)
    level: int = Field(..., ge=1, le=100)
    experience_points: int = Field(..., ge=0)
    badges_earned: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

# === TOOL MODELS ===

class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(10, ge=1, le=50)
    region: str = Field("us-en", max_length=10)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class SearchResultModel(BaseModel):
    """Search result model"""
    title: str
    url: HttpUrl
    snippet: str
    source: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FactCheckRequest(BaseModel):
    """Fact-checking request"""
    claim: str = Field(..., min_length=10, max_length=1000)
    
    @validator('claim')
    def validate_claim(cls, v):
        if not v.strip():
            raise ValueError("Claim cannot be empty")
        return v.strip()

class FactCheckResultModel(BaseModel):
    """Fact-checking result"""
    claim: str
    verdict: str = Field(..., pattern=r"^(true|false|mixed|unknown)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[Dict[str, Any]]
    explanation: str
    fact_checker: str
    checked_at: datetime

class ComputationRequest(BaseModel):
    """Mathematical computation request"""
    expression: str = Field(..., min_length=1, max_length=500)
    
    @validator('expression')
    def validate_expression(cls, v):
        # Basic validation for mathematical expressions
        if not re.match(r'^[0-9+\-*/().\s\w]+$', v):
            raise ValueError("Expression contains invalid characters")
        return v.strip()

class ComputationResult(BaseModel):
    """Computation result"""
    success: bool
    result: Optional[Union[int, float, str]] = None
    expression: str
    type: Optional[str] = None
    error: Optional[str] = None

# === WEBSOCKET MODELS ===

class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(UUID))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConnectionInfo(BaseModel):
    """WebSocket connection information"""
    session_id: str
    client_ip: str
    connected_at: datetime
    uptime_seconds: float
    state: str
    message_count: int
    error_count: int
    last_activity: datetime
    is_healthy: bool

# === CONFIGURATION MODELS ===

class ModelConfig(BaseModel):
    """LLM model configuration"""
    clarifier_model: str = Field("gemma2:2b", max_length=50)
    proponent_model: str = Field("llama3.1:8b", max_length=50)
    opponent_model: str = Field("llama3.1:8b", max_length=50)
    specialist_model: str = Field("llama3.1:8b", max_length=50)
    decider_model: str = Field("llama3.1:8b", max_length=50)
    auditor_model: str = Field("gemma2:2b", max_length=50)
    journaling_model: str = Field("gemma2:2b", max_length=50)
    
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=100, le=4096)
    timeout: int = Field(60, ge=10, le=300)

class UserPreferences(BaseModel):
    """User preferences model"""
    session_id: str
    theme: str = Field("auto", pattern=r"^(light|dark|auto)$")
    language: str = Field("en", max_length=5)
    timezone: str = Field("UTC", max_length=50)
    notifications_enabled: bool = True
    voice_enabled: bool = True
    auto_save: bool = True
    debate_preferences: Dict[str, Any] = Field(default_factory=dict)
    journal_preferences: Dict[str, Any] = Field(default_factory=dict)
    privacy_settings: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)

class SystemConfig(BaseModel):
    """System configuration model"""
    version: str = Field("1.0.0", max_length=20)
    debug_mode: bool = False
    max_sessions: int = Field(1000, ge=10, le=10000)
    session_timeout_hours: int = Field(24, ge=1, le=168)
    backup_enabled: bool = True
    backup_interval_hours: int = Field(24, ge=1, le=168)
    encryption_enabled: bool = True
    tools_enabled: bool = False
    logging_level: str = Field("INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

# === ANALYTICS MODELS ===

class AnalyticsQuery(BaseModel):
    """Analytics query parameters"""
    session_id: str
    metric_type: str = Field(..., pattern=r"^(writing_patterns|mood_trends|topic_analysis|productivity)$")
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    granularity: str = Field("daily", pattern=r"^(hourly|daily|weekly|monthly)$")

class WritingPatterns(BaseModel):
    """Writing patterns analytics"""
    total_entries: int
    total_words: int
    average_words_per_entry: float
    median_words_per_entry: float
    most_productive_day: Optional[str]
    most_productive_hour: Optional[int]
    daily_average: float
    writing_frequency: int
    longest_entry: int
    shortest_entry: int

class MoodTrends(BaseModel):
    """Mood trends analytics"""
    has_data: bool
    average_mood: Optional[float] = None
    median_mood: Optional[float] = None
    mood_range: Optional[Tuple[float, float]] = None
    mood_distribution: Dict[int, int] = Field(default_factory=dict)
    trend_data: List[Dict[str, Any]] = Field(default_factory=list)
    entries_with_mood: int = 0
    total_entries: int = 0

class TopicAnalysis(BaseModel):
    """Topic analysis results"""
    most_common_concepts: List[Tuple[str, int]]
    most_common_tags: List[Tuple[str, int]]
    unique_concepts: int
    unique_tags: int
    total_concepts: int
    total_tags: int

class ProductivityInsights(BaseModel):
    """Productivity insights"""
    insights: List[str]
    recommendations: List[str]
    productivity_score: int = Field(..., ge=0, le=100)
    consistency_rating: str = Field(..., pattern=r"^(starting|needs_improvement|fair|good|excellent)$")
    growth_areas: List[str]

class AnalyticsResponse(BaseModel):
    """Analytics response"""
    has_data: bool
    writing_patterns: Optional[WritingPatterns] = None
    mood_trends: Optional[MoodTrends] = None
    topic_analysis: Optional[TopicAnalysis] = None
    productivity_insights: Optional[ProductivityInsights] = None
    summary: Dict[str, Any] = Field(default_factory=dict)

# === PERFORMANCE MODELS ===

class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    component: str
    operation: str
    execution_time: float = Field(..., ge=0.0)
    memory_usage: Optional[int] = Field(None, ge=0)
    cpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0)
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    timestamp: datetime
    version: str
    components: Dict[str, bool]
    uptime_seconds: Optional[float] = None
    active_sessions: Optional[int] = None
    memory_usage: Optional[Dict[str, Any]] = None
    database_status: Optional[str] = None

# === UTILITY FUNCTIONS ===

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    try:
        UUID(session_id)
        return True
    except ValueError:
        return False

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_html(text: str) -> str:
    """Basic HTML sanitization"""
    import html
    return html.escape(text)

def validate_json_schema(data: Dict[str, Any], model_class: BaseModel) -> bool:
    """Validate JSON data against Pydantic model"""
    try:
        model_class(**data)
        return True
    except ValidationError:
        return False

# === ERROR HANDLING ===

class EchoForgeException(Exception):
    """Base exception for EchoForge"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(EchoForgeException):
    """Data validation exception"""
    pass

class DatabaseException(EchoForgeException):
    """Database operation exception"""
    pass

class AgentException(EchoForgeException):
    """Agent execution exception"""
    pass

class ToolException(EchoForgeException):
    """Tool integration exception"""
    pass

class AuthenticationException(EchoForgeException):
    """Authentication/authorization exception"""
    pass

# === EXPORT UTILITIES ===

def model_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    """Convert Pydantic model to dictionary"""
    return model.dict(exclude_none=exclude_none)

def dict_to_model(data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
    """Convert dictionary to Pydantic model with validation"""
    return model_class(**data)

def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format"""
    return dt.isoformat()

def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize datetime from ISO format"""
    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

# === SCHEMA VALIDATION HELPERS ===

def validate_rating(value: Any) -> Optional[int]:
    """Validate and normalize rating (1-10)"""
    if value is None:
        return None
    try:
        rating = int(value)
        return max(1, min(10, rating))
    except (ValueError, TypeError):
        return None

def validate_tags(tags: List[str]) -> List[str]:
    """Validate and clean tags"""
    if not tags:
        return []
    
    cleaned_tags = []
    for tag in tags:
        if isinstance(tag, str) and len(tag.strip()) > 0:
            cleaned_tag = re.sub(r'[^\w\s-]', '', tag.strip().lower())
            if cleaned_tag and len(cleaned_tag) <= 50:
                cleaned_tags.append(cleaned_tag)
    
    return cleaned_tags[:20]  # Limit to 20 tags

def validate_word_count(text: str) -> int:
    """Calculate and validate word count"""
    if not text:
        return 0
    return len(text.split())

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes"""
    word_count = validate_word_count(text)
    return max(1, round(word_count / words_per_minute))

# === TYPE ALIASES ===

SessionID = str
EntryID = str
DebateID = str
NodeID = str
ConnectionID = str
BadgeID = str
Timestamp = datetime
Rating = Optional[int]  # 1-10 scale

from enum import Enum

class AuditSeverity(Enum):
    """Severity levels for audit findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class JournalEntryType(Enum):
    """Types of journal entries."""
    REFLECTION = "reflection"
    INSIGHT = "insight"
    DEBATE_SUMMARY = "debate_summary"
    PERSONAL_NOTE = "personal_note"
    LEARNING_OUTCOME = "learning_outcome"

class ReflectionDepth(Enum):
    """Depth levels for reflection analysis."""
    SURFACE = "surface"
    MODERATE = "moderate"
    DEEP = "deep"
    PROFOUND = "profound"

class SpecialistType(Enum):
    """Types of specialist agents."""
    SCIENCE = "science"
    ETHICS = "ethics"
    ECONOMICS = "economics"
    HISTORY = "history"
    LEGAL = "legal"
    TECHNOLOGY = "technology"
    PHILOSOPHY = "philosophy"
    PSYCHOLOGY = "psychology"
