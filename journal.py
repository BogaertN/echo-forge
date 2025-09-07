import datetime
from db import DB_PATH
from encrypted_db import EncryptedDB
import logging
import json  # For advanced metadata handling

logger = logging.getLogger(__name__)

class Journal:
    def __init__(self, passphrase='default_pass'):
        """Initialize Journal with encrypted DB connection."""
        self.db = EncryptedDB(DB_PATH, passphrase)
        self.conn = self.db.connect()

    def create_entry(self, content, metadata=None, tags=None, ghost_loop=False):
        """Create a new journal entry with metadata, tags, and ghost loop flag."""
        # Convert metadata to JSON if not already
        if metadata and not isinstance(metadata, str):
            metadata = json.dumps(metadata)
        # Convert tags to comma-separated string if list
        if isinstance(tags, list):
            tags = ','.join(tags)
        
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO journal_entries (content, metadata, tags, ghost_loop)
        VALUES (?, ?, ?, ?)
        """, (content, metadata, tags, ghost_loop))
        self.conn.commit()
        entry_id = cursor.lastrowid
        self.update_gamification(entry_id)
        logger.info(f"Journal entry created with ID {entry_id}")
        return entry_id

    def search(self, query):
        """Search journal entries by content or tags."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM journal_entries WHERE content LIKE ? OR tags LIKE ?", (f"%{query}%", f"%{query}%"))
        return cursor.fetchall()

    def update_gamification(self, entry_id):
        """Update gamification metrics like streaks after entry creation."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp FROM journal_entries ORDER BY timestamp DESC LIMIT 2")
        rows = cursor.fetchall()
        if len(rows) > 1:
            last = datetime.datetime.strptime(rows[1][0], '%Y-%m-%d %H:%M:%S')
            current = datetime.datetime.strptime(rows[0][0], '%Y-%m-%d %H:%M:%S')
            if (current - last).days == 1:
                logger.info("Streak continued")
                # Advanced: Could update a separate gamification table or metadata here

    def get_streak(self):
        """Calculate current journaling streak."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp FROM journal_entries ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        streak = 1
        if len(rows) > 1:
            prev = datetime.datetime.strptime(rows[0][0], '%Y-%m-%d %H:%M:%S')
            for r in rows[1:]:
                curr = datetime.datetime.strptime(r[0], '%Y-%m-%d %H:%M:%S')
                if (prev - curr).days == 1:
                    streak += 1
                    prev = curr
                else:
                    break
        return streak

    def get_weekly_report(self):
        """Generate weekly journaling report with entry count and streak."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM journal_entries WHERE timestamp >= date('now', '-7 days')")
        count = cursor.fetchone()[0]
        streak = self.get_streak()
        report = f"Weekly entries: {count}, Current streak: {streak} days"
        logger.info(report)
        return report

    # Advanced method: Update entry (e.g., for editing or resolving ghost loops)
    def update_entry(self, entry_id, content=None, metadata=None, tags=None, ghost_loop=None):
        """Update an existing journal entry."""
        cursor = self.conn.cursor()
        updates = []
        params = []
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if metadata is not None:
            if not isinstance(metadata, str):
                metadata = json.dumps(metadata)
            updates.append("metadata = ?")
            params.append(metadata)
        if tags is not None:
            if isinstance(tags, list):
                tags = ','.join(tags)
            updates.append("tags = ?")
            params.append(tags)
        if ghost_loop is not None:
            updates.append("ghost_loop = ?")
            params.append(ghost_loop)
        
        if updates:
            query = f"UPDATE journal_entries SET {', '.join(updates)} WHERE id = ?"
            params.append(entry_id)
            cursor.execute(query, params)
            self.conn.commit()
            logger.info(f"Journal entry {entry_id} updated")

    # Advanced method: Delete entry (with audit logging)
    def delete_entry(self, entry_id):
        """Delete a journal entry and log the action."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM journal_entries WHERE id = ?", (entry_id,))
        self.conn.commit()
        logger.warning(f"Journal entry {entry_id} deleted")
