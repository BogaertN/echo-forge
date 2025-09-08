import sqlite3
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import uuid

logger = logging.getLogger(__name__)

# Database schema version for migrations
SCHEMA_VERSION = 1
DB_FILE_PATH = "data/echo_forge.db"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_path: str = DB_FILE_PATH
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    cache_size: int = -64000  # 64MB cache
    temp_store: str = "MEMORY"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    page_size: int = 4096
    auto_vacuum: str = "INCREMENTAL"
    backup_interval_hours: int = 24
    max_backup_files: int = 7

# SQL Schema Definitions
SCHEMA_SQL = {
    
    # Core system tables
    "schema_info": """
        CREATE TABLE IF NOT EXISTS schema_info (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    # Sessions table for tracking user sessions
    "sessions": """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            client_ip TEXT,
            user_agent TEXT,
            preferences TEXT DEFAULT '{}',  -- JSON
            status TEXT DEFAULT 'active',
            metadata TEXT DEFAULT '{}'      -- JSON
        )
    """,
    
    # Questions and clarification process
    "questions": """
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            original_question TEXT NOT NULL,
            clarified_question TEXT,
            clarification_history TEXT DEFAULT '[]',  -- JSON array
            clarification_complete BOOLEAN DEFAULT FALSE,
            key_concepts TEXT DEFAULT '[]',           -- JSON array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """,
    
    # Debate sessions and rounds
    "debates": """
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            status TEXT DEFAULT 'active',  -- active, completed, aborted
            config TEXT DEFAULT '{}',      -- JSON debate configuration
            phase TEXT DEFAULT 'clarification',  -- clarification, opening, main, synthesis, etc.
            current_round INTEGER DEFAULT 0,
            max_rounds INTEGER DEFAULT 6,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            performance_metrics TEXT DEFAULT '{}',  -- JSON
            ghost_loops_detected TEXT DEFAULT '[]', -- JSON array
            tools_used TEXT DEFAULT '[]',           -- JSON array
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
        )
    """,
    
    # Individual agent responses in debates
    "agent_responses": """
        CREATE TABLE IF NOT EXISTS agent_responses (
            id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            agent_role TEXT NOT NULL,  -- clarifier, proponent, opponent, specialist, etc.
            agent_model TEXT,
            round_number INTEGER DEFAULT 0,
            response_type TEXT NOT NULL,  -- opening_statement, debate_response, synthesis, etc.
            content TEXT NOT NULL,
            reasoning TEXT,
            key_points TEXT DEFAULT '[]',     -- JSON array
            sources TEXT DEFAULT '[]',        -- JSON array
            confidence_score REAL DEFAULT 0.0,
            response_time REAL DEFAULT 0.0,  -- seconds
            token_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
        )
    """,
    
    # Specialist consultations
    "specialist_consultations": """
        CREATE TABLE IF NOT EXISTS specialist_consultations (
            id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            domain TEXT NOT NULL,  -- ethics, logic, practical, emotional
            analysis TEXT NOT NULL,
            recommendations TEXT DEFAULT '[]',  -- JSON array
            concerns TEXT DEFAULT '[]',         -- JSON array
            confidence_score REAL DEFAULT 0.0,
            response_time REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
        )
    """,
    
    # Synthesis and decision results
    "syntheses": """
        CREATE TABLE IF NOT EXISTS syntheses (
            id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            synthesis_type TEXT DEFAULT 'final',  -- intermediate, final
            content TEXT NOT NULL,
            key_insights TEXT DEFAULT '[]',        -- JSON array
            balanced_perspective TEXT,
            action_items TEXT DEFAULT '[]',        -- JSON array
            confidence_score REAL DEFAULT 0.0,
            tone_modifier TEXT DEFAULT 'balanced',
            synthesis_style TEXT DEFAULT 'comprehensive',
            response_time REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
        )
    """,
    
    # Audit results and quality checks
    "audit_logs": """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            debate_id TEXT,
            synthesis_id TEXT,
            audit_type TEXT NOT NULL,  -- quality, fact_check, bias_check
            quality_score REAL DEFAULT 0.0,
            logical_consistency REAL DEFAULT 0.0,
            factual_accuracy REAL DEFAULT 0.0,
            bias_detection TEXT DEFAULT '[]',     -- JSON array
            recommendations TEXT DEFAULT '[]',   -- JSON array
            red_flags TEXT DEFAULT '[]',         -- JSON array
            issues_found TEXT DEFAULT '[]',      -- JSON array
            response_time REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE,
            FOREIGN KEY (synthesis_id) REFERENCES syntheses(id) ON DELETE CASCADE
        )
    """,
    
    # Journal entries
    "journal_entries": """
        CREATE TABLE IF NOT EXISTS journal_entries (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            debate_id TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            insights TEXT DEFAULT '[]',           -- JSON array
            personal_reflections TEXT,
            action_items TEXT DEFAULT '[]',       -- JSON array
            tags TEXT DEFAULT '[]',              -- JSON array
            mood_rating INTEGER,                 -- 1-10 scale
            complexity_rating INTEGER,          -- 1-10 scale
            satisfaction_rating INTEGER,        -- 1-10 scale
            word_count INTEGER DEFAULT 0,
            reading_time_minutes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE SET NULL
        )
    """,
    
    # Resonance map nodes (concepts, ideas, themes)
    "resonance_nodes": """
        CREATE TABLE IF NOT EXISTS resonance_nodes (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            node_type TEXT DEFAULT 'concept',    -- concept, theme, question, insight
            title TEXT NOT NULL,
            description TEXT,
            content_hash TEXT,                   -- For deduplication
            strength REAL DEFAULT 1.0,          -- Node importance/strength
            frequency INTEGER DEFAULT 1,        -- How often referenced
            last_activated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """,
    
    # Connections between resonance nodes
    "resonance_connections": """
        CREATE TABLE IF NOT EXISTS resonance_connections (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            connection_type TEXT DEFAULT 'relates_to',  -- relates_to, contradicts, supports, etc.
            strength REAL DEFAULT 1.0,
            created_from TEXT,                          -- journal_entry, debate, etc.
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_strengthened TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (source_node_id) REFERENCES resonance_nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES resonance_nodes(id) ON DELETE CASCADE,
            UNIQUE(source_node_id, target_node_id, connection_type)
        )
    """,
    
    # Gamification system
    "user_stats": """
        CREATE TABLE IF NOT EXISTS user_stats (
            session_id TEXT PRIMARY KEY,
            total_debates INTEGER DEFAULT 0,
            total_journal_entries INTEGER DEFAULT 0,
            total_questions_clarified INTEGER DEFAULT 0,
            current_streak_days INTEGER DEFAULT 0,
            longest_streak_days INTEGER DEFAULT 0,
            last_activity_date DATE,
            total_words_written INTEGER DEFAULT 0,
            total_insights_generated INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            experience_points INTEGER DEFAULT 0,
            badges_earned TEXT DEFAULT '[]',         -- JSON array
            achievements TEXT DEFAULT '[]',         -- JSON array
            preferences TEXT DEFAULT '{}',          -- JSON object
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """,
    
    # Badge definitions and tracking
    "badges": """
        CREATE TABLE IF NOT EXISTS badges (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            category TEXT DEFAULT 'general',      -- general, debate, journal, streak, etc.
            icon TEXT,
            requirements TEXT DEFAULT '{}',       -- JSON criteria
            points_reward INTEGER DEFAULT 10,
            rarity TEXT DEFAULT 'common',         -- common, rare, epic, legendary
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    # User badge achievements
    "user_badges": """
        CREATE TABLE IF NOT EXISTS user_badges (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            badge_id TEXT NOT NULL,
            earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            context TEXT,                         -- What triggered the badge
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (badge_id) REFERENCES badges(id) ON DELETE CASCADE,
            UNIQUE(session_id, badge_id)
        )
    """,
    
    # Tool usage tracking
    "tool_usage": """
        CREATE TABLE IF NOT EXISTS tool_usage (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            debate_id TEXT,
            tool_name TEXT NOT NULL,
            tool_action TEXT NOT NULL,
            input_data TEXT,                      -- JSON
            output_data TEXT,                     -- JSON
            success BOOLEAN DEFAULT TRUE,
            response_time REAL DEFAULT 0.0,
            cost REAL DEFAULT 0.0,               -- If applicable
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
        )
    """,
    
    # System performance and monitoring
    "performance_logs": """
        CREATE TABLE IF NOT EXISTS performance_logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            component TEXT NOT NULL,             -- orchestrator, agent, db, etc.
            operation TEXT NOT NULL,
            execution_time REAL NOT NULL,
            memory_usage INTEGER,                -- bytes
            cpu_usage REAL,                     -- percentage
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """,
    
    # Error tracking and debugging
    "error_logs": """
        CREATE TABLE IF NOT EXISTS error_logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            stack_trace TEXT,
            component TEXT,
            operation TEXT,
            severity TEXT DEFAULT 'error',       -- debug, info, warning, error, critical
            resolved BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """,
    
    # Backup and maintenance tracking
    "maintenance_logs": """
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id TEXT PRIMARY KEY,
            operation_type TEXT NOT NULL,        -- backup, cleanup, optimize, migrate
            status TEXT DEFAULT 'started',       -- started, completed, failed
            details TEXT,
            files_affected INTEGER DEFAULT 0,
            size_before INTEGER DEFAULT 0,      -- bytes
            size_after INTEGER DEFAULT 0,       -- bytes
            duration REAL DEFAULT 0.0,          -- seconds
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            metadata TEXT DEFAULT '{}'
        )
    """
}

# Index definitions for performance optimization
INDEXES_SQL = {
    # Session indexes
    "idx_sessions_last_activity": "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)",
    "idx_sessions_status": "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
    
    # Questions indexes
    "idx_questions_session": "CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id)",
    "idx_questions_created": "CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at)",
    "idx_questions_complete": "CREATE INDEX IF NOT EXISTS idx_questions_complete ON questions(clarification_complete)",
    
    # Debates indexes
    "idx_debates_session": "CREATE INDEX IF NOT EXISTS idx_debates_session ON debates(session_id)",
    "idx_debates_question": "CREATE INDEX IF NOT EXISTS idx_debates_question ON debates(question_id)",
    "idx_debates_status": "CREATE INDEX IF NOT EXISTS idx_debates_status ON debates(status)",
    "idx_debates_started": "CREATE INDEX IF NOT EXISTS idx_debates_started ON debates(started_at)",
    
    # Agent responses indexes
    "idx_agent_responses_debate": "CREATE INDEX IF NOT EXISTS idx_agent_responses_debate ON agent_responses(debate_id)",
    "idx_agent_responses_role": "CREATE INDEX IF NOT EXISTS idx_agent_responses_role ON agent_responses(agent_role)",
    "idx_agent_responses_round": "CREATE INDEX IF NOT EXISTS idx_agent_responses_round ON agent_responses(round_number)",
    "idx_agent_responses_created": "CREATE INDEX IF NOT EXISTS idx_agent_responses_created ON agent_responses(created_at)",
    
    # Journal indexes
    "idx_journal_session": "CREATE INDEX IF NOT EXISTS idx_journal_session ON journal_entries(session_id)",
    "idx_journal_debate": "CREATE INDEX IF NOT EXISTS idx_journal_debate ON journal_entries(debate_id)",
    "idx_journal_created": "CREATE INDEX IF NOT EXISTS idx_journal_created ON journal_entries(created_at)",
    "idx_journal_updated": "CREATE INDEX IF NOT EXISTS idx_journal_updated ON journal_entries(updated_at)",
    "idx_journal_tags": "CREATE INDEX IF NOT EXISTS idx_journal_tags ON journal_entries(tags)",
    
    # Resonance map indexes
    "idx_resonance_nodes_session": "CREATE INDEX IF NOT EXISTS idx_resonance_nodes_session ON resonance_nodes(session_id)",
    "idx_resonance_nodes_type": "CREATE INDEX IF NOT EXISTS idx_resonance_nodes_type ON resonance_nodes(node_type)",
    "idx_resonance_nodes_hash": "CREATE INDEX IF NOT EXISTS idx_resonance_nodes_hash ON resonance_nodes(content_hash)",
    "idx_resonance_nodes_activated": "CREATE INDEX IF NOT EXISTS idx_resonance_nodes_activated ON resonance_nodes(last_activated)",
    
    "idx_resonance_connections_session": "CREATE INDEX IF NOT EXISTS idx_resonance_connections_session ON resonance_connections(session_id)",
    "idx_resonance_connections_source": "CREATE INDEX IF NOT EXISTS idx_resonance_connections_source ON resonance_connections(source_node_id)",
    "idx_resonance_connections_target": "CREATE INDEX IF NOT EXISTS idx_resonance_connections_target ON resonance_connections(target_node_id)",
    "idx_resonance_connections_type": "CREATE INDEX IF NOT EXISTS idx_resonance_connections_type ON resonance_connections(connection_type)",
    
    # Gamification indexes
    "idx_user_stats_session": "CREATE INDEX IF NOT EXISTS idx_user_stats_session ON user_stats(session_id)",
    "idx_user_stats_activity": "CREATE INDEX IF NOT EXISTS idx_user_stats_activity ON user_stats(last_activity_date)",
    "idx_user_stats_level": "CREATE INDEX IF NOT EXISTS idx_user_stats_level ON user_stats(level)",
    
    "idx_user_badges_session": "CREATE INDEX IF NOT EXISTS idx_user_badges_session ON user_badges(session_id)",
    "idx_user_badges_earned": "CREATE INDEX IF NOT EXISTS idx_user_badges_earned ON user_badges(earned_at)",
    
    # Performance indexes
    "idx_performance_component": "CREATE INDEX IF NOT EXISTS idx_performance_component ON performance_logs(component)",
    "idx_performance_created": "CREATE INDEX IF NOT EXISTS idx_performance_created ON performance_logs(created_at)",
    "idx_performance_session": "CREATE INDEX IF NOT EXISTS idx_performance_session ON performance_logs(session_id)",
    
    # Error tracking indexes
    "idx_error_logs_session": "CREATE INDEX IF NOT EXISTS idx_error_logs_session ON error_logs(session_id)",
    "idx_error_logs_type": "CREATE INDEX IF NOT EXISTS idx_error_logs_type ON error_logs(error_type)",
    "idx_error_logs_created": "CREATE INDEX IF NOT EXISTS idx_error_logs_created ON error_logs(created_at)",
    "idx_error_logs_severity": "CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity)"
}

# Default badges to insert
DEFAULT_BADGES = [
    {
        "id": "first_debate",
        "name": "First Debate",
        "description": "Completed your first debate session",
        "category": "milestone",
        "requirements": '{"debates_completed": 1}',
        "points_reward": 50,
        "rarity": "common"
    },
    {
        "id": "active_debater",
        "name": "Active Debater",
        "description": "Completed 10 debate sessions",
        "category": "milestone",
        "requirements": '{"debates_completed": 10}',
        "points_reward": 200,
        "rarity": "rare"
    },
    {
        "id": "deep_thinker",
        "name": "Deep Thinker",
        "description": "Generated 100 insights across all sessions",
        "category": "insight",
        "requirements": '{"insights_generated": 100}',
        "points_reward": 300,
        "rarity": "rare"
    },
    {
        "id": "streak_starter",
        "name": "Streak Starter",
        "description": "Used EchoForge for 3 consecutive days",
        "category": "streak",
        "requirements": '{"streak_days": 3}',
        "points_reward": 100,
        "rarity": "common"
    },
    {
        "id": "weekly_warrior",
        "name": "Weekly Warrior",
        "description": "Used EchoForge for 7 consecutive days",
        "category": "streak",
        "requirements": '{"streak_days": 7}',
        "points_reward": 250,
        "rarity": "rare"
    },
    {
        "id": "prolific_writer",
        "name": "Prolific Writer",
        "description": "Written over 10,000 words in journal entries",
        "category": "journal",
        "requirements": '{"words_written": 10000}',
        "points_reward": 400,
        "rarity": "epic"
    },
    {
        "id": "question_master",
        "name": "Question Master",
        "description": "Completed clarification for 50 questions",
        "category": "clarification",
        "requirements": '{"questions_clarified": 50}',
        "points_reward": 350,
        "rarity": "epic"
    }
]

class DatabaseManager:
    """
    Database management class for EchoForge.
    
    Handles database initialization, migrations, connections, and maintenance.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.db_path = self.config.db_path
        self._connection_lock = threading.Lock()
        self._last_backup = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with proper configuration"""
        with self._connection_lock:
            conn = sqlite3.connect(self.db_path)
            
            # Configure connection
            conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            conn.execute(f"PRAGMA temp_store = {self.config.temp_store}")
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            conn.execute(f"PRAGMA page_size = {self.config.page_size}")
            conn.execute(f"PRAGMA auto_vacuum = {self.config.auto_vacuum}")
            
            if self.config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON")
            
            # Set row factory for easier data access
            conn.row_factory = sqlite3.Row
            
            try:
                yield conn
            finally:
                conn.close()
    
    def init_database(self) -> bool:
        """
        Initialize database with schema and default data.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Create all tables
                for table_name, sql in SCHEMA_SQL.items():
                    logger.debug(f"Creating table: {table_name}")
                    conn.execute(sql)
                
                # Create indexes
                for index_name, sql in INDEXES_SQL.items():
                    logger.debug(f"Creating index: {index_name}")
                    conn.execute(sql)
                
                # Initialize schema version
                self._init_schema_version(conn)
                
                # Insert default badges
                self._insert_default_badges(conn)
                
                # Commit all changes
                conn.commit()
                
                logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def _init_schema_version(self, conn):
        """Initialize schema version tracking"""
        # Check if schema_info exists and has data
        cursor = conn.execute("SELECT COUNT(*) FROM schema_info")
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.execute(
                "INSERT INTO schema_info (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
            logger.info(f"Schema version {SCHEMA_VERSION} initialized")
    
    def _insert_default_badges(self, conn):
        """Insert default badge definitions"""
        try:
            for badge in DEFAULT_BADGES:
                conn.execute("""
                    INSERT OR IGNORE INTO badges 
                    (id, name, description, category, requirements, points_reward, rarity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    badge["id"],
                    badge["name"],
                    badge["description"],
                    badge["category"],
                    badge["requirements"],
                    badge["points_reward"],
                    badge["rarity"]
                ))
            
            logger.info(f"Inserted {len(DEFAULT_BADGES)} default badges")
            
        except Exception as e:
            logger.error(f"Error inserting default badges: {e}")
    
    def get_schema_version(self) -> int:
        """Get current schema version"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT version FROM schema_info ORDER BY id DESC LIMIT 1")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting schema version: {e}")
            return 0
    
    def migrate_schema(self, target_version: int = None) -> bool:
        """
        Migrate database schema to target version.
        
        Args:
            target_version: Target schema version (defaults to latest)
            
        Returns:
            True if migration successful, False otherwise
        """
        if target_version is None:
            target_version = SCHEMA_VERSION
        
        current_version = self.get_schema_version()
        
        if current_version >= target_version:
            logger.info(f"Schema already at version {current_version}, no migration needed")
            return True
        
        logger.info(f"Migrating schema from version {current_version} to {target_version}")
        
        try:
            with self.get_connection() as conn:
                # Perform migrations step by step
                for version in range(current_version + 1, target_version + 1):
                    if self._migrate_to_version(conn, version):
                        logger.info(f"Successfully migrated to version {version}")
                    else:
                        logger.error(f"Failed to migrate to version {version}")
                        return False
                
                conn.commit()
                logger.info(f"Schema migration completed to version {target_version}")
                return True
                
        except Exception as e:
            logger.error(f"Error during schema migration: {e}")
            return False
    
    def _migrate_to_version(self, conn, version: int) -> bool:
        """Migrate to specific version (placeholder for future migrations)"""
        # Future migration logic would go here
        # For now, just update the version
        conn.execute(
            "INSERT INTO schema_info (version) VALUES (?)",
            (version,)
        )
        return True
    
    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space and optimize performance"""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuum completed")
                return True
        except Exception as e:
            logger.error(f"Error during database vacuum: {e}")
            return False
    
    def analyze_database(self) -> bool:
        """Analyze database to update query planner statistics"""
        try:
            with self.get_connection() as conn:
                conn.execute("ANALYZE")
                logger.info("Database analyze completed")
                return True
        except Exception as e:
            logger.error(f"Error during database analyze: {e}")
            return False
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Path for backup file (auto-generated if None)
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path}.backup_{timestamp}"
            
            # Create backup using file copy (simple but effective)
            shutil.copy2(self.db_path, backup_path)
            
            # Log backup operation
            self._log_maintenance_operation(
                operation_type="backup",
                status="completed",
                details=f"Backup created at {backup_path}",
                files_affected=1,
                size_after=os.path.getsize(backup_path)
            )
            
            self._last_backup = datetime.now()
            logger.info(f"Database backup created: {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            self._log_maintenance_operation(
                operation_type="backup",
                status="failed",
                details=str(e)
            )
            return False
    
    def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            backup_dir = os.path.dirname(self.db_path)
            backup_pattern = f"{os.path.basename(self.db_path)}.backup_"
            
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith(backup_pattern):
                    file_path = os.path.join(backup_dir, file)
                    backup_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups beyond max_backup_files
            for file_path, _ in backup_files[self.config.max_backup_files:]:
                os.remove(file_path)
                logger.info(f"Removed old backup: {file_path}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Create backup of current database before restore
            current_backup = f"{self.db_path}.pre_restore_{int(time.time())}"
            shutil.copy2(self.db_path, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            # Log restore operation
            self._log_maintenance_operation(
                operation_type="restore",
                status="completed",
                details=f"Restored from {backup_path}",
                files_affected=1,
                size_after=os.path.getsize(self.db_path)
            )
            
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            self._log_maintenance_operation(
                operation_type="restore",
                status="failed",
                details=str(e)
            )
            return False
    
    def _log_maintenance_operation(self, operation_type: str, status: str, 
                                 details: str = "", files_affected: int = 0,
                                 size_before: int = 0, size_after: int = 0,
                                 duration: float = 0.0):
        """Log maintenance operation"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO maintenance_logs 
                    (id, operation_type, status, details, files_affected, 
                     size_before, size_after, duration, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    operation_type,
                    status,
                    details,
                    files_affected,
                    size_before,
                    size_after,
                    duration,
                    datetime.now() if status in ["completed", "failed"] else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging maintenance operation: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information"""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Basic file info
                stats["file_size"] = os.path.getsize(self.db_path)
                stats["schema_version"] = self.get_schema_version()
                
                # Table row counts
                stats["table_counts"] = {}
                for table_name in SCHEMA_SQL.keys():
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    stats["table_counts"][table_name] = cursor.fetchone()[0]
                
                # Performance info
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                stats["total_pages"] = page_count
                stats["page_size"] = page_size
                stats["estimated_size"] = page_count * page_size
                
                # Index usage
                cursor = conn.execute("PRAGMA index_list('journal_entries')")
                stats["indexes"] = len(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def check_integrity(self) -> bool:
        """Check database integrity"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                
                if result == "ok":
                    logger.info("Database integrity check passed")
                    return True
                else:
                    logger.error(f"Database integrity check failed: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """
        Clean up old data beyond retention period.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.get_connection() as conn:
                # Clean up old sessions and related data
                cursor = conn.execute("""
                    DELETE FROM sessions 
                    WHERE last_activity < ? AND status = 'inactive'
                """, (cutoff_date,))
                
                deleted_sessions = cursor.rowcount
                
                # Clean up old performance logs
                cursor = conn.execute("""
                    DELETE FROM performance_logs 
                    WHERE created_at < ?
                """, (cutoff_date,))
                
                deleted_performance = cursor.rowcount
                
                # Clean up old error logs
                cursor = conn.execute("""
                    DELETE FROM error_logs 
                    WHERE created_at < ? AND resolved = TRUE
                """, (cutoff_date,))
                
                deleted_errors = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {deleted_sessions} sessions, "
                          f"{deleted_performance} performance logs, "
                          f"{deleted_errors} error logs removed")
                
                return True
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for external use
def init_database() -> bool:
    """Initialize the database"""
    return db_manager.init_database()

def get_db_connection():
    """Get a database connection"""
    return db_manager.get_connection()

def backup_database(backup_path: str = None) -> bool:
    """Create database backup"""
    return db_manager.backup_database(backup_path)

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    return db_manager.get_database_stats()

def check_database_integrity() -> bool:
    """Check database integrity"""
    return db_manager.check_integrity()
