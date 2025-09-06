import sqlite3
import datetime
from db import DB_PATH
import logging

logger = logging.getLogger(__name__)

class Journal:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)

    def create_entry(self, content, metadata=None, tags=None, ghost_loop=False):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO journal_entries (content, metadata, tags, ghost_loop)
        VALUES (?, ?, ?, ?)
        """, (content, metadata, tags, ghost_loop))
        self.conn.commit()
        entry_id = cursor.lastrowid
        self.update_gamification(entry_id)
        return entry_id

    def search(self, query):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM journal_entries WHERE content LIKE ?", (f"%{query}%",))
        return cursor.fetchall()

    def update_gamification(self, entry_id):
        # Basic streak logic (expand for badges)
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp FROM journal_entries ORDER BY timestamp DESC LIMIT 2")
        rows = cursor.fetchall()
        if len(rows) > 1 and (datetime.datetime.now() - datetime.datetime.strptime(rows[1][0], '%Y-%m-%d %H:%M:%S')).days == 1:
            logger.info("Streak continued")
