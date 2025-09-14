"""
db.py - Database management for EchoForge
"""
import sqlite3
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for EchoForge"""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.environ.get('ECHO_FORGE_DATA_DIR', os.path.expanduser('~/echo_forge_data'))
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / 'echoforge.db'
        
        logger.info(f"Database initialized at: {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        return conn
    
    def initialize_database(self):
        """Initialize database tables and indexes"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Debate sessions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS debate_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    original_question TEXT,
                    clarified_question TEXT,
                    phase TEXT DEFAULT 'initial',
                    status TEXT DEFAULT 'active',
                    rounds_completed INTEGER DEFAULT 0,
                    metadata TEXT  -- JSON for additional data
                )
                ''')
                
                # Agent messages table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    message_type TEXT,
                    round_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER,
                    response_time_ms INTEGER,
                    FOREIGN KEY (session_id) REFERENCES debate_sessions (session_id)
                )
                ''')
                
                # Journal entries table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    title TEXT,
                    metadata TEXT,  -- JSON for tags, categories, etc.
                    sentiment REAL,  -- Sentiment analysis score (-1 to 1)
                    word_count INTEGER,
                    ghost_loop BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Audit logs table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    session_id TEXT,
                    user_id TEXT,
                    details TEXT,  -- JSON for event details
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # User sessions table (for multi-user support)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_activity TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    operation_type TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_debate_sessions_session_id ON debate_sessions(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_debate_sessions_created_at ON debate_sessions(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id ON agent_messages(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_messages_created_at ON agent_messages(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_journal_entries_session_id ON journal_entries(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_journal_entries_created_at ON journal_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_session_id ON audit_logs(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_session_id ON performance_metrics(session_id)')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def create_debate_session(self, session_id: str, original_question: str = None, metadata: str = None) -> int:
        """Create a new debate session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO debate_sessions (session_id, original_question, metadata)
                VALUES (?, ?, ?)
                ''', (session_id, original_question, metadata))
                conn.commit()
                
                session_row_id = cursor.lastrowid
                logger.info(f"Debate session created: {session_id}")
                
                # Log the event
                self.log_audit_event("session_created", session_id, f"New debate session created")
                
                return session_row_id
        except Exception as e:
            logger.error(f"Failed to create debate session: {str(e)}")
            raise
    
    def update_debate_session(self, session_id: str, **kwargs):
        """Update debate session fields"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                set_clauses = []
                values = []
                valid_fields = ['clarified_question', 'phase', 'status', 'rounds_completed', 'metadata']
                
                for key, value in kwargs.items():
                    if key in valid_fields:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                if set_clauses:
                    query = f"UPDATE debate_sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                    values.append(session_id)
                    cursor.execute(query, values)
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.debug(f"Updated debate session {session_id}: {kwargs}")
                    
        except Exception as e:
            logger.error(f"Failed to update debate session: {str(e)}")
            raise
    
    def save_agent_message(self, session_id: str, agent_type: str, agent_id: str, 
                          content: str, message_type: str = None, round_number: int = None,
                          token_count: int = None, response_time_ms: int = None) -> int:
        """Save an agent message to the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO agent_messages 
                (session_id, agent_type, agent_id, message_content, message_type, round_number, token_count, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, agent_type, agent_id, content, message_type, round_number, token_count, response_time_ms))
                conn.commit()
                
                message_id = cursor.lastrowid
                logger.info(f"Agent message saved: {agent_type} in session {session_id}")
                return message_id
                
        except Exception as e:
            logger.error(f"Failed to save agent message: {str(e)}")
            raise
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all messages for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM agent_messages 
                WHERE session_id = ? 
                ORDER BY created_at ASC
                LIMIT ?
                ''', (session_id, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get session messages: {str(e)}")
            return []
    
    def create_journal_entry(self, content: str, session_id: str = None, title: str = None,
                           metadata: str = None, sentiment: float = None, word_count: int = None,
                           ghost_loop: bool = False) -> int:
        """Create a journal entry"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate word count if not provided
                if word_count is None and content:
                    word_count = len(content.split())
                
                cursor.execute('''
                INSERT INTO journal_entries (session_id, content, title, metadata, sentiment, word_count, ghost_loop)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, content, title, metadata, sentiment, word_count, ghost_loop))
                conn.commit()
                
                entry_id = cursor.lastrowid
                logger.info(f"Journal entry created: {entry_id}")
                return entry_id
                
        except Exception as e:
            logger.error(f"Failed to create journal entry: {str(e)}")
            raise
    
    def search_journal_entries(self, query: str = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Search journal entries with pagination"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if query:
                    cursor.execute('''
                    SELECT * FROM journal_entries 
                    WHERE content LIKE ? OR title LIKE ? OR metadata LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit, offset))
                else:
                    cursor.execute('''
                    SELECT * FROM journal_entries 
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (limit, offset))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search journal entries: {str(e)}")
            return []
    
    def log_audit_event(self, event_type: str, session_id: str = None, details: str = None,
                       user_id: str = None, ip_address: str = None, user_agent: str = None):
        """Log an audit event"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO audit_logs (event_type, session_id, user_id, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (event_type, session_id, user_id, details, ip_address, user_agent))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    def log_performance_metric(self, operation_type: str, duration_ms: int, session_id: str = None,
                             success: bool = True, error_message: str = None):
        """Log a performance metric"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO performance_metrics (session_id, operation_type, duration_ms, success, error_message)
                VALUES (?, ?, ?, ?, ?)
                ''', (session_id, operation_type, duration_ms, success, error_message))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log performance metric: {str(e)}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM debate_sessions WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get session info: {str(e)}")
            return None
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get message count by agent type
                cursor.execute('''
                SELECT agent_type, COUNT(*) as count, AVG(response_time_ms) as avg_response_time
                FROM agent_messages 
                WHERE session_id = ?
                GROUP BY agent_type
                ''', (session_id,))
                agent_stats = [dict(row) for row in cursor.fetchall()]
                
                # Get total message count
                cursor.execute('''
                SELECT COUNT(*) as total_messages, 
                       SUM(token_count) as total_tokens,
                       AVG(response_time_ms) as avg_response_time
                FROM agent_messages 
                WHERE session_id = ?
                ''', (session_id,))
                totals = dict(cursor.fetchone())
                
                return {
                    "agent_statistics": agent_stats,
                    "totals": totals
                }
                
        except Exception as e:
            logger.error(f"Failed to get session statistics: {str(e)}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete old sessions and related data
                cursor.execute('''
                DELETE FROM debate_sessions 
                WHERE created_at < datetime('now', '-{} days')
                '''.format(days_old))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old sessions")
                    self.log_audit_event("cleanup", details=f"Removed {deleted_count} sessions older than {days_old} days")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {str(e)}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['debate_sessions', 'agent_messages', 'journal_entries', 'audit_logs']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Get database file size
                stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
                stats["db_size_mb"] = round(stats["db_size_bytes"] / (1024 * 1024), 2)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = str(self.data_dir / f"echoforge_backup_{timestamp}.db")
            
            # Use VACUUM INTO for a clean backup
            with self.get_connection() as conn:
                conn.execute(f"VACUUM INTO '{backup_path}'")
            
            logger.info(f"Database backed up to: {backup_path}")
            self.log_audit_event("database_backup", details=f"Backup created at {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup database: {str(e)}")
            raise
