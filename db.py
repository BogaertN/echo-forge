from sqlcipher3 import dbapi2 as sqlite
import os
from logging import getLogger

logger = getLogger(__name__)

DB_PATH = os.path.join(os.environ['ECHO_FORGE_DATA_DIR'], 'echo_forge.db')

def init_db(passphrase='default_pass'):  # Change passphrase in production for security
    """Initialize the encrypted SQLite database with all schemas, including upgrades for existing DBs."""
    conn = sqlite.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA key = '{passphrase}'")
    
    # Journal Entries Schema: Stores user journal entries with metadata and gamification flags
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        content TEXT NOT NULL,
        metadata TEXT,  -- JSON for weights, emotion, priority (e.g., {"relevance": 8, "emotion": "neutral"})
        tags TEXT,  -- Comma-separated tags for searchability
        ghost_loop BOOLEAN DEFAULT FALSE  -- Flag for unresolved "ghost loops"
    )
    ''')
    
    # Debate Logs Schema: Logs full debate sessions with versioning and links
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS debate_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        question TEXT NOT NULL,
        proponent_args TEXT,  -- JSON of round-based arguments
        opponent_args TEXT,  -- JSON of round-based arguments
        synthesis TEXT,
        journal_entry_id INTEGER,
        rounds INTEGER,
        locked BOOLEAN DEFAULT FALSE,  -- Lock after completion to prevent edits
        original_debate_id INTEGER,  -- For replays/counterfactuals
        FOREIGN KEY (journal_entry_id) REFERENCES journal_entries(id),
        FOREIGN KEY (original_debate_id) REFERENCES debate_logs(id)
    )
    ''')
    
    # Ensure backward compatibility by adding columns if missing
    cursor.execute("PRAGMA table_info(debate_logs)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'rounds' not in columns:
        cursor.execute("ALTER TABLE debate_logs ADD COLUMN rounds INTEGER")
    if 'locked' not in columns:
        cursor.execute("ALTER TABLE debate_logs ADD COLUMN locked BOOLEAN DEFAULT FALSE")
    if 'original_debate_id' not in columns:
        cursor.execute("ALTER TABLE debate_logs ADD COLUMN original_debate_id INTEGER")
    
    # Audit Logs Schema: Tracks system events for security and debugging
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT,
        details TEXT  -- JSON or text details of the event
    )
    ''')
    
    # Optional indexes for performance (advanced: speeds up searches)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON journal_entries(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_timestamp ON debate_logs(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)")
    
    conn.commit()
    conn.close()
    logger.info("DB initialized at %s with encryption and indexes for performance", DB_PATH)

if __name__ == "__main__":
    init_db()
