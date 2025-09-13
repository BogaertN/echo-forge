#!/usr/bin/env python3
"""
Database management for EchoForge.
Handles SQLite database operations with optional encryption support.
"""

import sqlite3
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for EchoForge."""
    
    def __init__(self, db_path: Optional[str] = None, encryption_key: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to database file. If None, uses ECHO_FORGE_DATA_DIR env var.
            encryption_key: Encryption key for SQLCipher (currently disabled).
        """
        if db_path is None:
            data_dir = os.getenv('ECHO_FORGE_DATA_DIR', './data')
            Path(data_dir).mkdir(exist_ok=True)
            db_path = f"{data_dir}/echoforge.db"
        
        self.db_path = db_path
        # For now, disable encryption until sqlcipher3 is working properly
        self.encryption_enabled = False
        self.encryption_key = encryption_key
        
        logger.info(f"Database initialized at: {self.db_path}")
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn
    
    def initialize_database(self) -> None:
        """Initialize database with required tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Create journal entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,
                    insights TEXT,
                    session_id TEXT,
                    word_count INTEGER DEFAULT 0,
                    sentiment_score REAL DEFAULT 0.0
                )
            ''')
            
            # Create debate sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS debate_sessions (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    final_synthesis TEXT,
                    participant_count INTEGER DEFAULT 0,
                    round_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create agent messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_name TEXT,
                    agent_type TEXT,
                    message_content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL DEFAULT 0.0,
                    response_time_ms INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES debate_sessions (id)
                )
            ''')
            
            # Create user settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create gamification table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gamification (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default',
                    points INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    badges TEXT DEFAULT '[]',
                    streak_days INTEGER DEFAULT 0,
                    last_activity DATE,
                    total_debates INTEGER DEFAULT 0,
                    total_journal_entries INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON journal_entries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_journal_session ON journal_entries(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON agent_messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON agent_messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_created ON debate_sessions(created_at)')
            
            conn.commit()
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def save_journal_entry(self, content: str, session_id: Optional[str] = None, 
                          tags: Optional[str] = None, insights: Optional[str] = None) -> int:
        """Save a journal entry and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            word_count = len(content.split())
            
            cursor.execute('''
                INSERT INTO journal_entries (content, session_id, tags, insights, word_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (content, session_id, tags, insights, word_count))
            
            entry_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Journal entry saved with ID: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error saving journal entry: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_journal_entries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get journal entries with pagination."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, content, timestamp, tags, insights, session_id, word_count
                FROM journal_entries
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            entries = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(entries)} journal entries")
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving journal entries: {e}")
            return []
        finally:
            conn.close()
    
    def create_debate_session(self, session_id: str, question: str) -> bool:
        """Create a new debate session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO debate_sessions (id, question, created_at)
                VALUES (?, ?, ?)
            ''', (session_id, question, datetime.now().isoformat()))
            
            conn.commit()
            logger.info(f"Debate session created: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating debate session: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def save_agent_message(self, session_id: str, agent_name: str, agent_type: str,
                          content: str, confidence: float = 0.0) -> bool:
        """Save an agent message."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO agent_messages 
                (session_id, agent_name, agent_type, message_content, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, agent_name, agent_type, content, confidence))
            
            conn.commit()
            logger.info(f"Agent message saved: {agent_name} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent message: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a debate session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT agent_name, agent_type, message_content, timestamp, confidence_score
                FROM agent_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            ''', (session_id,))
            
            messages = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving session messages: {e}")
            return []
        finally:
            conn.close()
    
    def update_session_status(self, session_id: str, status: str, 
                             final_synthesis: Optional[str] = None) -> bool:
        """Update debate session status."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if status == 'completed' and final_synthesis:
                cursor.execute('''
                    UPDATE debate_sessions 
                    SET status = ?, completed_at = ?, final_synthesis = ?
                    WHERE id = ?
                ''', (status, datetime.now().isoformat(), final_synthesis, session_id))
            else:
                cursor.execute('''
                    UPDATE debate_sessions 
                    SET status = ?
                    WHERE id = ?
                ''', (status, session_id))
            
            conn.commit()
            logger.info(f"Session {session_id} status updated to: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count journal entries
            cursor.execute('SELECT COUNT(*) FROM journal_entries')
            stats['journal_entries'] = cursor.fetchone()[0]
            
            # Count debate sessions
            cursor.execute('SELECT COUNT(*) FROM debate_sessions')
            stats['debate_sessions'] = cursor.fetchone()[0]
            
            # Count agent messages
            cursor.execute('SELECT COUNT(*) FROM agent_messages')
            stats['agent_messages'] = cursor.fetchone()[0]
            
            # Count active sessions
            cursor.execute("SELECT COUNT(*) FROM debate_sessions WHERE status = 'active'")
            stats['active_sessions'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            conn.close()


def main():
    """Main function for testing database functionality."""
    print("EchoForge Database Manager")
    print("=" * 50)
    
    # Initialize database
    db = DatabaseManager()
    db.initialize_database()
    
    # Show stats
    stats = db.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\nDatabase initialization completed successfully!")


if __name__ == "__main__":
    main()
