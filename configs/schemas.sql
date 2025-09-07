-- EchoForge Full Data Schemas
-- From appendices D. Full Data Schemas

-- Journal Entries
CREATE TABLE journal_entries (
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
    auto_suggestions JSON,
    resolution TEXT
);

-- Debates
CREATE TABLE debates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    clarified_prompt TEXT NOT NULL,
    config JSON,
    transcript JSON,
    synthesis TEXT,
    auditor_findings JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resonance Nodes
CREATE TABLE resonance_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resonance Edges
CREATE TABLE resonance_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_id, to_id, relation_type)
);

-- Gamification
CREATE TABLE gamification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streak_count INTEGER DEFAULT 0,
    badges JSON DEFAULT '[]',
    clarity_metrics JSON DEFAULT '{}',
    last_journal_date DATE,
    weekly_report JSON DEFAULT '{}',
    total_ghost_closures INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool Logs
CREATE TABLE tool_logs (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    session_id TEXT,
    agent_id TEXT,
    inputs JSON,
    output TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_entries_created ON journal_entries(created_at);
CREATE INDEX idx_entries_ghost ON journal_entries(ghost_loop);
CREATE INDEX idx_debates_session ON debates(session_id);
CREATE INDEX idx_nodes_type ON resonance_nodes(type);
CREATE INDEX idx_edges_from ON resonance_edges(from_id);
CREATE INDEX idx_edges_to ON resonance_edges(to_id);
CREATE INDEX idx_logs_timestamp ON tool_logs(timestamp);
