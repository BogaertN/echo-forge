import logging

logger = logging.getLogger(__name__)

# SQL Schema Definitions
# These are used in encrypted_db.py for _initialize_schema

JOURNAL_ENTRIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    tags JSON,
    weights JSON,
    ghost_loop BOOLEAN DEFAULT FALSE,
    ghost_loop_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    debate_id TEXT,
    user_edits TEXT,
    auto_suggestions JSON,  -- For auto-generated suggestions
    resolution TEXT  -- For ghost loop resolutions
)
"""

DEBATES_SCHEMA = """
CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    clarified_prompt TEXT NOT NULL,
    config JSON,
    transcript JSON,
    synthesis TEXT,
    auditor_findings JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

RESONANCE_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS resonance_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- 'entry', 'debate', 'ghost_loop', 'concept'
    content_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

RESONANCE_EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS resonance_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_id, to_id, relation_type)
)
"""

GAMIFICATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS gamification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streak_count INTEGER DEFAULT 0,
    badges JSON DEFAULT '[]',
    clarity_metrics JSON DEFAULT '{}',
    last_journal_date DATE,
    weekly_report JSON DEFAULT '{}',
    total_ghost_closures INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

TOOL_LOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_logs (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    session_id TEXT,
    agent_id TEXT,
    inputs JSON,
    output TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_entries_created ON journal_entries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_entries_ghost ON journal_entries(ghost_loop)",
    "CREATE INDEX IF NOT EXISTS idx_debates_session ON debates(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_type ON resonance_nodes(type)",
    "CREATE INDEX IF NOT EXISTS idx_edges_from ON resonance_edges(from_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_to ON resonance_edges(to_id)",
    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON tool_logs(timestamp)"
]

def get_all_schemas() -> List[str]:
    """Get all schema creation SQL statements"""
    return [
        JOURNAL_ENTRIES_SCHEMA,
        DEBATES_SCHEMA,
        RESONANCE_NODES_SCHEMA,
        RESONANCE_EDGES_SCHEMA,
        GAMIFICATION_SCHEMA,
        TOOL_LOGS_SCHEMA
    ] + INDEXES

def migrate_schema(current_version: int, target_version: int) -> List[str]:
    """Migration scripts between versions (extend as needed)"""
    migrations = []
    if current_version < 2:
        migrations.append("ALTER TABLE journal_entries ADD COLUMN resolution TEXT")
    # Add more migrations
    return migrations

logger.info("Database schemas loaded")
