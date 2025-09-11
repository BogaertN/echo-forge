import sqlite3
import os
from typing import Optional

class EncryptedDatabase:
    """Simple database wrapper - encryption disabled for now."""
    
    def __init__(self, db_path: str, password: Optional[str] = None):
        self.db_path = db_path
        self.password = password
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def close(self):
        """Close database connection."""
        pass
    
    def initialize(self):
        """Initialize database."""
        # For now, just ensure the database file exists
        conn = self.get_connection()
        conn.close()
        return True

# Backward compatibility
EncryptedDB = EncryptedDatabase

