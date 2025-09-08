import os
import sqlite3
import hashlib
import secrets
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import base64
import hmac

try:
    import sqlcipher3
    SQLCIPHER_AVAILABLE = True
except ImportError:
    SQLCIPHER_AVAILABLE = False
    logging.warning("SQLCipher not available, falling back to standard SQLite (data will not be encrypted)")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class EncryptionConfig:
    """Configuration for database encryption"""
    cipher: str = "aes-256-cbc"
    kdf_algorithm: str = "PBKDF2"
    kdf_iterations: int = 256000
    key_size: int = 32  # 256 bits
    salt_size: int = 16
    iv_size: int = 16
    page_size: int = 4096
    cache_size: int = -64000  # 64MB
    auto_vacuum: str = "INCREMENTAL"
    secure_delete: bool = True
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456  # 256MB

class SecureString:
    """Secure string class for handling sensitive data like passwords"""
    
    def __init__(self, data: str):
        self._data = data.encode('utf-8')
        self._hash = hashlib.sha256(self._data).hexdigest()
    
    def get_value(self) -> str:
        """Get the string value (use sparingly)"""
        return self._data.decode('utf-8')
    
    def get_hash(self) -> str:
        """Get hash of the string for comparison"""
        return self._hash
    
    def verify(self, other: str) -> bool:
        """Verify if another string matches this one"""
        other_hash = hashlib.sha256(other.encode('utf-8')).hexdigest()
        return hmac.compare_digest(self._hash, other_hash)
    
    def __del__(self):
        """Securely clear data when object is destroyed"""
        if hasattr(self, '_data'):
            # Overwrite memory (best effort)
            self._data = b'\x00' * len(self._data)
    
    def __str__(self):
        return "*" * 8
    
    def __repr__(self):
        return f"SecureString(hash={self._hash[:8]}...)"

class EncryptionKeyManager:
    """Manages encryption keys and key derivation"""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self._master_key: Optional[bytes] = None
        self._salt: Optional[bytes] = None
        self._key_derivation_cache = {}
        
    def derive_key_from_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Master password
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(self.config.salt_size)
        
        # Create cache key
        cache_key = hashlib.sha256(password.encode() + salt).hexdigest()
        
        # Check cache first
        if cache_key in self._key_derivation_cache:
            return self._key_derivation_cache[cache_key], salt
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_size,
            salt=salt,
            iterations=self.config.kdf_iterations,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode())
        
        # Cache the result (limit cache size)
        if len(self._key_derivation_cache) > 10:
            # Remove oldest entry
            oldest_key = next(iter(self._key_derivation_cache))
            del self._key_derivation_cache[oldest_key]
        
        self._key_derivation_cache[cache_key] = key
        
        return key, salt
    
    def generate_database_key(self, password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """
        Generate SQLCipher database key from password.
        
        Args:
            password: Master password
            salt: Salt for key derivation
            
        Returns:
            Tuple of (hex_key, salt)
        """
        key, salt = self.derive_key_from_password(password, salt)
        hex_key = key.hex()
        return hex_key, salt
    
    def rotate_key(self, old_password: str, new_password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """
        Generate new key for key rotation.
        
        Args:
            old_password: Current password
            new_password: New password
            salt: Salt for new key derivation
            
        Returns:
            Tuple of (new_hex_key, new_salt)
        """
        logger.info("Rotating encryption key")
        return self.generate_database_key(new_password, salt)
    
    def generate_file_encryption_key(self) -> bytes:
        """Generate key for file-level encryption (backups, exports)"""
        return Fernet.generate_key()
    
    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-CBC"""
        iv = secrets.token_bytes(self.config.iv_size)
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Add PKCS7 padding
        pad_len = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_len] * pad_len)
        
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Prepend IV to encrypted data
        return iv + encrypted
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-CBC"""
        iv = encrypted_data[:self.config.iv_size]
        ciphertext = encrypted_data[self.config.iv_size:]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        pad_len = padded_data[-1]
        return padded_data[:-pad_len]

class EncryptedDatabase:
    """
    Encrypted database wrapper using SQLCipher for privacy-first data storage.
    
    Provides transparent encryption/decryption with secure key management.
    """
    
    def __init__(self, db_path: str, password: str = None, config: EncryptionConfig = None):
        self.db_path = db_path
        self.config = config or EncryptionConfig()
        self.key_manager = EncryptionKeyManager(self.config)
        self._connection_lock = threading.Lock()
        
        # Secure password handling
        if password is None:
            password = self._get_default_password()
        
        self.password = SecureString(password)
        
        # Generate or load encryption key
        self._db_key, self._salt = self._initialize_encryption()
        
        # Track encryption operations
        self._encryption_stats = {
            "operations_count": 0,
            "last_key_rotation": None,
            "database_created": datetime.now(),
            "total_encrypted_size": 0
        }
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database if it doesn't exist
        if not os.path.exists(self.db_path):
            self._create_encrypted_database()
        
        logger.info(f"EncryptedDatabase initialized: {self.db_path}")
    
    def _get_default_password(self) -> str:
        """Get default password from environment or generate one"""
        env_password = os.getenv('ECHOFORGE_DB_PASSWORD')
        if env_password:
            return env_password
        
        # Generate default password based on system characteristics
        # In production, this should be user-provided or stored securely
        system_info = f"{os.getlogin()}-{os.uname().nodename}-{self.db_path}"
        default_password = hashlib.sha256(system_info.encode()).hexdigest()[:32]
        
        logger.warning("Using generated default password. Set ECHOFORGE_DB_PASSWORD environment variable for production.")
        return default_password
    
    def _initialize_encryption(self) -> Tuple[str, bytes]:
        """Initialize encryption key and salt"""
        salt_file = f"{self.db_path}.salt"
        
        # Load existing salt or create new one
        if os.path.exists(salt_file):
            with open(salt_file, 'rb') as f:
                salt = f.read()
            logger.debug("Loaded existing encryption salt")
        else:
            salt = secrets.token_bytes(self.config.salt_size)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            logger.debug("Generated new encryption salt")
        
        # Derive database key
        db_key, _ = self.key_manager.generate_database_key(
            self.password.get_value(), salt
        )
        
        return db_key, salt
    
    def _create_encrypted_database(self):
        """Create new encrypted database"""
        logger.info("Creating new encrypted database")
        
        try:
            with self.get_connection() as conn:
                # Test the connection by creating a simple table
                conn.execute("CREATE TABLE IF NOT EXISTS _encryption_test (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO _encryption_test (id) VALUES (1)")
                conn.execute("DROP TABLE _encryption_test")
                
            logger.info("Encrypted database created successfully")
            
        except Exception as e:
            logger.error(f"Error creating encrypted database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get encrypted database connection"""
        with self._connection_lock:
            if SQLCIPHER_AVAILABLE:
                conn = sqlcipher3.connect(self.db_path)
                
                # Set encryption key
                conn.execute(f"PRAGMA key = \"x'{self._db_key}'\"")
                
                # Configure SQLCipher settings
                conn.execute(f"PRAGMA cipher = '{self.config.cipher}'")
                conn.execute(f"PRAGMA kdf_iter = {self.config.kdf_iterations}")
                conn.execute(f"PRAGMA cipher_page_size = {self.config.page_size}")
                
            else:
                # Fallback to standard SQLite (with warning)
                conn = sqlite3.connect(self.db_path)
                logger.warning("Using unencrypted SQLite connection")
            
            # Configure database settings
            conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            conn.execute(f"PRAGMA temp_store = {self.config.temp_store}")
            conn.execute(f"PRAGMA auto_vacuum = {self.config.auto_vacuum}")
            conn.execute(f"PRAGMA mmap_size = {self.config.mmap_size}")
            
            if self.config.secure_delete:
                conn.execute("PRAGMA secure_delete = ON")
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set row factory for easier data access
            conn.row_factory = sqlite3.Row
            
            try:
                # Verify encryption is working
                if SQLCIPHER_AVAILABLE:
                    self._verify_encryption(conn)
                
                yield conn
                
            finally:
                conn.close()
                self._encryption_stats["operations_count"] += 1
    
    def _verify_encryption(self, conn):
        """Verify that encryption is properly enabled"""
        try:
            # This should work with correct key
            cursor = conn.execute("SELECT count(*) FROM sqlite_master")
            cursor.fetchone()
            
            # Verify cipher settings
            cursor = conn.execute("PRAGMA cipher_version")
            cipher_version = cursor.fetchone()
            
            if cipher_version:
                logger.debug(f"SQLCipher version: {cipher_version[0]}")
            else:
                logger.warning("Unable to verify SQLCipher version")
                
        except Exception as e:
            logger.error(f"Encryption verification failed: {e}")
            raise ValueError("Database encryption verification failed")
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change database encryption password.
        
        Args:
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password change successful, False otherwise
        """
        try:
            # Verify old password
            if not self.password.verify(old_password):
                logger.error("Old password verification failed")
                return False
            
            # Generate new key
            new_key, new_salt = self.key_manager.rotate_key(
                old_password, new_password, None
            )
            
            if SQLCIPHER_AVAILABLE:
                with self.get_connection() as conn:
                    # Change the key
                    conn.execute(f"PRAGMA rekey = \"x'{new_key}'\"")
                    
                    # Verify new key works
                    cursor = conn.execute("SELECT count(*) FROM sqlite_master")
                    cursor.fetchone()
            
            # Update stored values
            self._db_key = new_key
            self._salt = new_salt
            self.password = SecureString(new_password)
            
            # Save new salt
            salt_file = f"{self.db_path}.salt"
            with open(salt_file, 'wb') as f:
                f.write(new_salt)
            
            # Update stats
            self._encryption_stats["last_key_rotation"] = datetime.now()
            
            logger.info("Database password changed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error changing database password: {e}")
            return False
    
    def create_encrypted_backup(self, backup_path: str, backup_password: str = None) -> bool:
        """
        Create encrypted backup of the database.
        
        Args:
            backup_path: Path for backup file
            backup_password: Password for backup encryption (uses current if None)
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            if backup_password is None:
                backup_password = self.password.get_value()
            
            # Generate backup encryption key
            backup_key, backup_salt = self.key_manager.generate_database_key(
                backup_password, None
            )
            
            if SQLCIPHER_AVAILABLE:
                with self.get_connection() as conn:
                    # Attach backup database with new key
                    conn.execute(f"ATTACH DATABASE '{backup_path}' AS backup KEY \"x'{backup_key}'\"")
                    
                    # Copy all data to backup
                    conn.execute("SELECT sqlcipher_export('backup')")
                    
                    # Detach backup database
                    conn.execute("DETACH DATABASE backup")
            else:
                # For unencrypted fallback, use file copy
                import shutil
                shutil.copy2(self.db_path, backup_path)
            
            # Save backup salt
            backup_salt_file = f"{backup_path}.salt"
            with open(backup_salt_file, 'wb') as f:
                f.write(backup_salt)
            
            # Create backup metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "source_db": self.db_path,
                "encrypted": SQLCIPHER_AVAILABLE,
                "cipher": self.config.cipher,
                "kdf_iterations": self.config.kdf_iterations
            }
            
            metadata_file = f"{backup_path}.metadata"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Encrypted backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating encrypted backup: {e}")
            return False
    
    def restore_from_encrypted_backup(self, backup_path: str, backup_password: str) -> bool:
        """
        Restore database from encrypted backup.
        
        Args:
            backup_path: Path to backup file
            backup_password: Password for backup decryption
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            # Load backup salt
            backup_salt_file = f"{backup_path}.salt"
            if not os.path.exists(backup_salt_file):
                logger.error(f"Backup salt file not found: {backup_salt_file}")
                return False
            
            with open(backup_salt_file, 'rb') as f:
                backup_salt = f.read()
            
            # Generate backup key
            backup_key, _ = self.key_manager.generate_database_key(
                backup_password, backup_salt
            )
            
            if SQLCIPHER_AVAILABLE:
                # Create temporary connection to backup
                backup_conn = sqlcipher3.connect(backup_path)
                backup_conn.execute(f"PRAGMA key = \"x'{backup_key}'\"")
                
                # Verify backup is readable
                cursor = backup_conn.execute("SELECT count(*) FROM sqlite_master")
                cursor.fetchone()
                backup_conn.close()
                
                # Restore to current database
                with self.get_connection() as conn:
                    # Attach backup database
                    conn.execute(f"ATTACH DATABASE '{backup_path}' AS backup KEY \"x'{backup_key}'\"")
                    
                    # Import from backup
                    conn.execute("SELECT sqlcipher_export('main', 'backup')")
                    
                    # Detach backup database
                    conn.execute("DETACH DATABASE backup")
            else:
                # For unencrypted fallback, use file copy
                import shutil
                shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from encrypted backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from encrypted backup: {e}")
            return False
    
    def export_data(self, export_path: str, tables: List[str] = None, 
                   encrypt_export: bool = True) -> bool:
        """
        Export database data to file.
        
        Args:
            export_path: Path for export file
            tables: List of tables to export (all if None)
            encrypt_export: Whether to encrypt the export file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Get table list
                if tables is None:
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    """)
                    tables = [row[0] for row in cursor.fetchall()]
                
                # Export data
                export_data = {}
                
                for table in tables:
                    cursor = conn.execute(f"SELECT * FROM {table}")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    export_data[table] = {
                        "columns": columns,
                        "rows": [dict(row) for row in rows]
                    }
                
                # Convert to JSON
                json_data = json.dumps(export_data, indent=2, default=str)
                
                if encrypt_export:
                    # Encrypt the export
                    file_key = self.key_manager.generate_file_encryption_key()
                    fernet = Fernet(file_key)
                    encrypted_data = fernet.encrypt(json_data.encode())
                    
                    # Save encrypted data and key
                    with open(export_path, 'wb') as f:
                        f.write(encrypted_data)
                    
                    with open(f"{export_path}.key", 'wb') as f:
                        f.write(file_key)
                    
                    logger.info(f"Encrypted data export created: {export_path}")
                else:
                    # Save unencrypted JSON
                    with open(export_path, 'w') as f:
                        f.write(json_data)
                    
                    logger.info(f"Data export created: {export_path}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get information about database encryption"""
        info = {
            "encrypted": SQLCIPHER_AVAILABLE,
            "cipher": self.config.cipher,
            "kdf_algorithm": self.config.kdf_algorithm,
            "kdf_iterations": self.config.kdf_iterations,
            "key_size": self.config.key_size,
            "database_path": self.db_path,
            "file_size": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
            "stats": self._encryption_stats.copy()
        }
        
        if SQLCIPHER_AVAILABLE:
            try:
                with self.get_connection() as conn:
                    # Get SQLCipher version and settings
                    cursor = conn.execute("PRAGMA cipher_version")
                    cipher_version = cursor.fetchone()
                    if cipher_version:
                        info["cipher_version"] = cipher_version[0]
                    
                    # Get page count and size
                    cursor = conn.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]
                    
                    cursor = conn.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    
                    info["pages"] = page_count
                    info["page_size"] = page_size
                    info["estimated_size"] = page_count * page_size
                    
            except Exception as e:
                logger.error(f"Error getting encryption info: {e}")
        
        return info
    
    def vacuum_encrypted_database(self) -> bool:
        """Vacuum the encrypted database to reclaim space"""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            
            logger.info("Encrypted database vacuum completed")
            return True
            
        except Exception as e:
            logger.error(f"Error vacuuming encrypted database: {e}")
            return False
    
    def check_encryption_integrity(self) -> bool:
        """Check encryption integrity and database accessibility"""
        try:
            with self.get_connection() as conn:
                # Test basic operations
                cursor = conn.execute("SELECT count(*) FROM sqlite_master")
                table_count = cursor.fetchone()[0]
                
                # Verify we can read some data
                if table_count > 0:
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' 
                        LIMIT 1
                    """)
                    test_table = cursor.fetchone()
                    
                    if test_table:
                        cursor = conn.execute(f"SELECT count(*) FROM {test_table[0]}")
                        cursor.fetchone()
                
                logger.info("Encryption integrity check passed")
                return True
                
        except Exception as e:
            logger.error(f"Encryption integrity check failed: {e}")
            return False
    
    def close(self):
        """Close database and clean up resources"""
        try:
            # Clear sensitive data
            if hasattr(self, '_db_key'):
                self._db_key = None
            
            if hasattr(self, 'password'):
                del self.password
            
            # Clear key derivation cache
            self.key_manager._key_derivation_cache.clear()
            
            logger.info("EncryptedDatabase closed and cleaned up")
            
        except Exception as e:
            logger.error(f"Error closing encrypted database: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Convenience functions for compatibility
def create_encrypted_connection(db_path: str, password: str = None) -> EncryptedDatabase:
    """Create encrypted database connection"""
    return EncryptedDatabase(db_path, password)

def verify_sqlcipher_available() -> bool:
    """Check if SQLCipher is available"""
    return SQLCIPHER_AVAILABLE
