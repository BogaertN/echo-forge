import sqlite3
import os
from logging import getLogger

logger = getLogger(__name__)

DB_PATH = os.path.join(os.environ['ECHO_FORGE_DATA_DIR'], 'echo_forge.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Journal Entries Schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        content TEXT NOT NULL,
        metadata TEXT,  -- JSON for weights, emotion, priority
        tags TEXT,  -- Comma-separated
        ghost_loop BOOLEAN DEFAULT FALSE
    )
    ''')
    # Debate Logs Schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS debate_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        question TEXT NOT NULL,
        proponent_args TEXT,
        opponent_args TEXT,
        synthesis TEXT,
        journal_entry_id INTEGER,
        FOREIGN KEY (journal_entry_id) REFERENCES journal_entries(id)
    )
    ''')
    # Audit Logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT,
        details TEXT
    )
    ''')
    conn.commit()
    conn.close()
    logger.info("DB initialized at %s", DB_PATH)

if __name__ == "__main__":
    init_db()
