import logging
import sqlite3
import sqlcipher3  # For encrypted SQLite
import os
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import shutil
import tempfile

logger = logging.getLogger(__name__)

class EncryptedDB:
    """
    Encrypted database handler for EchoForge using SQLCipher.
    Manages journal entries, debates, resonance maps, gamification data,
    with full encryption at rest. Supports backups, restores, migrations,
    and secure queries.
    """
    
    def __init__(self, db_path: str = "data/journal.db", passphrase: str = None):
        """
        Initialize the encrypted database.
        
        Args:
            db_path: Path to the database file
            passphrase: Encryption passphrase (if None, generate one)
        """
        self.db_path = db_path
        self.passphrase = passphrase or self._generate_passphrase()
        self._ensure_db_directory()
        self._initialize_schema()
        
        logger.info(f"EncryptedDB initialized at {db_path}")
    
    def _generate_passphrase(self) -> str:
        """Generate a secure passphrase if none provided"""
        # In production, this should be user-provided or from secure storage
        passphrase = hashlib.sha256(os.urandom(32)).hexdigest()
        logger.warning("Generated temporary passphrase. In production, use secure storage.")
        return passphrase
    
    def _ensure_db_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for encrypted database connections"""
        conn = sqlcipher3.connect(self.db_path)
        try:
            conn.execute(f"PRAGMA key = '{self.passphrase}'")
            conn.execute("PRAGMA cipher_memory_security = ON")
            conn.execute("PRAGMA cipher_page_size = 4096")
            yield conn
        finally:
            conn.close()
    
    def _initialize_schema(self):
        """Initialize database schema if not exists"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Journal Entries Table
            cursor.execute("""
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
                    user_edits TEXT
                )
            """)
            
            # Debates Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debates (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    clarified_prompt TEXT NOT NULL,
                    config JSON,
                    transcript JSON,
                    synthesis JSON,
                    auditor_findings JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Resonance Map (Graph Structure)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resonance_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,  -- 'entry', 'debate', 'ghost_loop'
                    content_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resonance_edges (
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,  -- 'resolves', 'contradicts', 'relates_to', etc.
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (from_id, to_id, relation_type)
                )
            """)
            
            # Gamification Data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gamification (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    streak_count INTEGER DEFAULT 0,
                    badges JSON DEFAULT '[]',
                    clarity_metrics JSON,
                    last_journal_date DATE,
                    weekly_report JSON,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ensure at least one gamification row
            cursor.execute("INSERT OR IGNORE INTO gamification (id) VALUES (1)")
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_created ON journal_entries(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_debates_session ON debates(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON resonance_edges(from_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON resonance_edges(to_id)")
            
            conn.commit()
        
        logger.info("Database schema initialized/verified")
    
    # Journaling Methods
    
    def create_journal_entry(self, content: str, metadata: Dict, session_id: str, debate_id: Optional[str] = None, user_edits: Optional[str] = None) -> str:
        """Create a new journal entry"""
        entry_id = str(uuid.uuid4())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO journal_entries (
                    id, content, title, summary, tags, weights, 
                    ghost_loop, ghost_loop_reason, session_id, debate_id, user_edits
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                content,
                metadata.get('title'),
                metadata.get('summary'),
                json.dumps(metadata.get('tags', [])),
                json.dumps(metadata.get('weights', {})),
                metadata.get('ghost_loop', False),
                metadata.get('ghost_loop_reason', ''),
                session_id,
                debate_id,
                user_edits
            ))
            conn.commit()
        
        self._update_gamification_on_entry()
        
        logger.info(f"Journal entry created: {entry_id}")
        return entry_id
    
    def search_journal(self, query: str = "", tags: List[str] = None, ghost_loops_only: bool = False, limit: int = 50) -> List[Dict]:
        """Search journal entries with filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            sql = "SELECT * FROM journal_entries WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (content LIKE ? OR title LIKE ? OR summary LIKE ?)"
                like_query = f"%{query}%"
                params.extend([like_query, like_query, like_query])
            
            if tags:
                sql += " AND json_array_length(tags) > 0"
                for tag in tags:
                    sql += " AND json_extract(tags, '$[*]') LIKE ?"
                    params.append(f"%{tag}%")
            
            if ghost_loops_only:
                sql += " AND ghost_loop = 1"
            
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                entry = dict(zip(columns, row))
                entry['tags'] = json.loads(entry['tags']) if entry['tags'] else []
                entry['weights'] = json.loads(entry['weights']) if entry['weights'] else {}
                results.append(entry)
            
            return results
    
    def update_journal_entry(self, entry_id: str, updates: Dict):
        """Update an existing journal entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            set_clause = []
            params = []
            for key, value in updates.items():
                if key in ['tags', 'weights']:
                    value = json.dumps(value)
                set_clause.append(f"{key} = ?")
                params.append(value)
            
            params.append(entry_id)
            
            sql = f"UPDATE journal_entries SET {', '.join(set_clause)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            cursor.execute(sql, params)
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Journal entry updated: {entry_id}")
                return True
            return False
    
    # Debate Methods
    
    def save_debate(self, session_id: str, clarified_prompt: str, config: Dict) -> str:
        """Save a new debate session"""
        debate_id = str(uuid.uuid4())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO debates (
                    id, session_id, clarified_prompt, config,
                    transcript, synthesis, auditor_findings
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                debate_id,
                session_id,
                clarified_prompt,
                json.dumps(config),
                json.dumps([]),  # Initial empty transcript
                None,
                None
            ))
            conn.commit()
        
        logger.info(f"Debate saved: {debate_id}")
        return debate_id
    
    def update_debate_transcript(self, debate_id: str, transcript: List[Dict]):
        """Update debate transcript"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE debates SET transcript = ? WHERE id = ?",
                (json.dumps(transcript), debate_id)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Debate transcript updated: {debate_id}")
                return True
            return False
    
    def finalize_debate(self, debate_id: str, synthesis: str, auditor_findings: Dict):
        """Finalize debate with synthesis and auditor results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE debates SET synthesis = ?, auditor_findings = ? WHERE id = ?",
                (synthesis, json.dumps(auditor_findings), debate_id)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Debate finalized: {debate_id}")
                return True
            return False
    
    # Resonance Mapping Methods
    
    def add_resonance_node(self, node_type: str, content_summary: str) -> str:
        """Add a new node to the resonance map"""
        node_id = str(uuid.uuid4())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO resonance_nodes (id, type, content_summary) VALUES (?, ?, ?)",
                (node_id, node_type, content_summary)
            )
            conn.commit()
        
        logger.info(f"Resonance node added: {node_id} ({node_type})")
        return node_id
    
    def add_resonance_edge(self, from_id: str, to_id: str, relation_type: str, strength: float = 1.0):
        """Add or update an edge between nodes"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO resonance_edges 
                (from_id, to_id, relation_type, strength) 
                VALUES (?, ?, ?, ?)
            """, (from_id, to_id, relation_type, strength))
            conn.commit()
        
        logger.info(f"Resonance edge added: {from_id} -> {to_id} ({relation_type})")
    
    def get_resonance_map(self, node_id: Optional[str] = None, max_depth: int = 3) -> Dict:
        """Get resonance map graph, optionally centered on a node"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all nodes
            cursor.execute("SELECT * FROM resonance_nodes")
            nodes = {row[0]: {'type': row[1], 'summary': row[2]} for row in cursor.fetchall()}
            
            # Get all edges
            cursor.execute("SELECT * FROM resonance_edges")
            edges = []
            for row in cursor.fetchall():
                edges.append({
                    'from': row[0],
                    'to': row[1],
                    'relation': row[2],
                    'strength': row[3]
                })
            
            # If centered on node, perform BFS to limit depth
            if node_id:
                from collections import deque
                
                visited = set()
                queue = deque([(node_id, 0)])
                filtered_nodes = {}
                filtered_edges = []
                
                while queue:
                    current, depth = queue.popleft()
                    if current in visited or depth > max_depth:
                        continue
                    visited.add(current)
                    if current in nodes:
                        filtered_nodes[current] = nodes[current]
                    
                    # Add outgoing edges
                    for edge in edges:
                        if edge['from'] == current:
                            filtered_edges.append(edge)
                            queue.append((edge['to'], depth + 1))
                        elif edge['to'] == current:
                            filtered_edges.append(edge)
                            queue.append((edge['from'], depth + 1))
                
                return {'nodes': filtered_nodes, 'edges': filtered_edges}
            
            return {'nodes': nodes, 'edges': edges}
    
    def find_ghost_loops(self) -> List[Dict]:
        """Find unresolved ghost loops"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM journal_entries 
                WHERE ghost_loop = 1 
                ORDER BY priority DESC, created_at DESC
            """)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            ghost_loops = [dict(zip(columns, row)) for row in rows]
            for gl in ghost_loops:
                gl['tags'] = json.loads(gl['tags']) if gl['tags'] else []
                gl['weights'] = json.loads(gl['weights']) if gl['weights'] else {}
            return ghost_loops
    
    # Gamification Methods
    
    def _update_gamification_on_entry(self):
        """Update gamification stats after new entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current streak
            cursor.execute("""
                SELECT streak_count, last_journal_date 
                FROM gamification WHERE id = 1
            """)
            current_streak, last_date = cursor.fetchone() or (0, None)
            
            today = datetime.now().date()
            if last_date:
                last_date = datetime.fromisoformat(last_date).date()
                if (today - last_date).days == 1:
                    new_streak = current_streak + 1
                elif (today - last_date).days > 1:
                    new_streak = 1
                else:
                    new_streak = current_streak
            else:
                new_streak = 1
            
            # Update badges if needed
            cursor.execute("SELECT badges FROM gamification WHERE id = 1")
            badges = json.loads(cursor.fetchone()[0] or '[]')
            
            # Example badge logic: Consistency badge for 7-day streak
            if new_streak >= 7 and 'consistency' not in badges:
                badges.append('consistency')
            
            # Update
            cursor.execute("""
                UPDATE gamification SET 
                    streak_count = ?,
                    last_journal_date = ?,
                    badges = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (new_streak, today.isoformat(), json.dumps(badges)))
            conn.commit()
    
    def get_gamification_stats(self) -> Dict:
        """Get current gamification statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gamification WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return {
                    'streak_count': row[1],
                    'badges': json.loads(row[2] or '[]'),
                    'clarity_metrics': json.loads(row[3] or '{}'),
                    'last_journal_date': row[4],
                    'weekly_report': json.loads(row[5] or '{}')
                }
            return {}
    
    def generate_weekly_report(self) -> Dict:
        """Generate weekly clarity snapshot"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get entries from last week
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*), AVG(json_extract(weights, '$.clarity')) 
                FROM journal_entries 
                WHERE created_at >= ?
            """, (week_ago,))
            count, avg_clarity = cursor.fetchone()
            
            report = {
                'entries_count': count or 0,
                'average_clarity': avg_clarity or 0,
                'ghost_loops_closed': self._count_closed_ghost_loops(week_ago),
                'generated_at': datetime.now().isoformat()
            }
            
            # Save report
            cursor.execute(
                "UPDATE gamification SET weekly_report = ? WHERE id = 1",
                (json.dumps(report),)
            )
            conn.commit()
            
            return report
    
    def _count_closed_ghost_loops(self, since: str) -> int:
        """Count ghost loops closed since date"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM journal_entries 
                WHERE ghost_loop = 0 AND updated_at >= ? AND created_at < updated_at
            """, (since,))
            return cursor.fetchone()[0]
    
    # Backup and Restore
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create encrypted backup of database"""
        backup_path = backup_path or f"{self.db_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
        shutil.copy(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        return backup_path
    
    def restore_database(self, backup_path: str):
        """Restore from backup"""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Verify backup with passphrase
        try:
            temp_conn = sqlcipher3.connect(backup_path)
            temp_conn.execute(f"PRAGMA key = '{self.passphrase}'")
            temp_conn.execute("SELECT count(*) FROM sqlite_master")
            temp_conn.close()
        except sqlite3.DatabaseError:
            raise ValueError("Invalid backup or wrong passphrase")
        
        shutil.copy(backup_path, self.db_path)
        logger.info(f"Database restored from {backup_path}")
    
    def export_to_json(self, export_path: str):
        """Export all data to JSON for migration"""
        data = {
            'journal_entries': self.search_journal(limit=999999),
            'debates': self._get_all_debates(),
            'resonance_map': self.get_resonance_map(),
            'gamification': self.get_gamification_stats()
        }
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data exported to {export_path}")
    
    def _get_all_debates(self) -> List[Dict]:
        """Get all debates"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM debates")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            debates = []
            for row in rows:
                debate = dict(zip(columns, row))
                debate['config'] = json.loads(debate['config']) if debate['config'] else {}
                debate['transcript'] = json.loads(debate['transcript']) if debate['transcript'] else []
                debate['auditor_findings'] = json.loads(debate['auditor_findings']) if debate['auditor_findings'] else {}
                debates.append(debate)
            return debates
    
    def import_from_json(self, import_path: str):
        """Import data from JSON export"""
        with open(import_path, 'r') as f:
            data = json.load(f)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Import journal entries
            for entry in data.get('journal_entries', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO journal_entries (
                        id, content, title, summary, tags, weights, 
                        ghost_loop, ghost_loop_reason, created_at, updated_at,
                        session_id, debate_id, user_edits
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry['id'],
                    entry['content'],
                    entry['title'],
                    entry['summary'],
                    json.dumps(entry['tags']),
                    json.dumps(entry['weights']),
                    entry['ghost_loop'],
                    entry['ghost_loop_reason'],
                    entry['created_at'],
                    entry['updated_at'],
                    entry['session_id'],
                    entry['debate_id'],
                    entry['user_edits']
                ))
            
            # Import debates
            for debate in data.get('debates', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO debates (
                        id, session_id, clarified_prompt, config,
                        transcript, synthesis, auditor_findings, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    debate['id'],
                    debate['session_id'],
                    debate['clarified_prompt'],
                    json.dumps(debate['config']),
                    json.dumps(debate['transcript']),
                    debate['synthesis'],
                    json.dumps(debate['auditor_findings']),
                    debate['created_at']
                ))
            
            # Import resonance map
            for node_id, node in data['resonance_map'].get('nodes', {}).items():
                cursor.execute("""
                    INSERT OR REPLACE INTO resonance_nodes 
                    (id, type, content_summary, created_at) 
                    VALUES (?, ?, ?, ?)
                """, (
                    node_id,
                    node['type'],
                    node['summary'],
                    datetime.now().isoformat()  # Use current if not provided
                ))
            
            for edge in data['resonance_map'].get('edges', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO resonance_edges 
                    (from_id, to_id, relation_type, strength, created_at) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    edge['from'],
                    edge['to'],
                    edge['relation'],
                    edge['strength'],
                    datetime.now().isoformat()
                ))
            
            # Import gamification
            gam = data.get('gamification', {})
            cursor.execute("""
                UPDATE gamification SET 
                    streak_count = ?,
                    badges = ?,
                    clarity_metrics = ?,
                    last_journal_date = ?,
                    weekly_report = ?
                WHERE id = 1
            """, (
                gam.get('streak_count', 0),
                json.dumps(gam.get('badges', [])),
                json.dumps(gam.get('clarity_metrics', {})),
                gam.get('last_journal_date'),
                json.dumps(gam.get('weekly_report', {}))
            ))
            
            conn.commit()
        
        logger.info(f"Data imported from {import_path}")
    
    def change_passphrase(self, new_passphrase: str):
        """Change database encryption passphrase"""
        with self._get_connection() as conn:
            conn.execute(f"PRAGMA rekey = '{new_passphrase}'")
        
        self.passphrase = new_passphrase
        logger.info("Database passphrase changed")
    
    def vacuum_database(self):
        """Optimize and vacuum database"""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
        logger.info("Database vacuumed")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM journal_entries")
            entry_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM debates")
            debate_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM resonance_nodes")
            node_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM resonance_edges")
            edge_count = cursor.fetchone()[0]
        
        file_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
        
        return {
            'entry_count': entry_count,
            'debate_count': debate_count,
            'resonance_nodes': node_count,
            'resonance_edges': edge_count,
            'file_size_mb': round(file_size, 2),
            'last_backup': self._get_last_backup_time()
        }
    
    def _get_last_backup_time(self) -> Optional[str]:
        """Get timestamp of last backup (scan for .bak files)"""
        backups = [f for f in os.listdir(os.path.dirname(self.db_path)) if f.endswith('.bak')]
        if backups:
            latest = max(backups, key=lambda f: os.path.getmtime(os.path.join(os.path.dirname(self.db_path), f)))
            return datetime.fromtimestamp(os.path.getmtime(os.path.join(os.path.dirname(self.db_path), latest))).isoformat()
        return None
