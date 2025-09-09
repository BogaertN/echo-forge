import os
import sys
import sqlite3
import shutil
import gzip
import hashlib
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from db import DatabaseManager


class BackupManager:
    """
    Manages database backup, restore, and maintenance operations for EchoForge.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the backup manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config()
        self.db_path = Path(self.config.database.db_path)
        self.backup_dir = Path(self.config.database.backup_directory)
        self.retention_days = self.config.database.backup_retention_days
        self.enable_compression = getattr(self.config.database, 'backup_compression', True)
        
        # Setup logging
        self._setup_logging()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"BackupManager initialized - DB: {self.db_path}, Backup Dir: {self.backup_dir}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Setup file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'backup.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def create_backup(self, backup_name: Optional[str] = None, verify: bool = True) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_name: Optional custom name for backup
            verify: Whether to verify backup integrity
            
        Returns:
            Path to created backup file
            
        Raises:
            BackupError: If backup creation fails
        """
        try:
            # Check if source database exists
            if not self.db_path.exists():
                raise BackupError(f"Source database not found: {self.db_path}")
            
            # Generate backup filename
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"echoforge_backup_{timestamp}"
            
            # Determine backup path
            backup_file = self.backup_dir / f"{backup_name}.db"
            if self.enable_compression:
                backup_file = backup_file.with_suffix('.db.gz')
            
            self.logger.info(f"Creating backup: {backup_file}")
            
            # Create backup with progress tracking
            file_size = self.db_path.stat().st_size
            self._create_backup_with_progress(backup_file, file_size)
            
            # Verify backup if requested
            if verify:
                self.logger.info("Verifying backup integrity...")
                if not self.verify_backup(str(backup_file)):
                    raise BackupError("Backup verification failed")
                self.logger.info("Backup verification successful")
            
            # Create metadata file
            metadata = {
                'backup_file': str(backup_file),
                'source_db': str(self.db_path),
                'created_at': datetime.now().isoformat(),
                'file_size': backup_file.stat().st_size,
                'source_size': file_size,
                'compressed': self.enable_compression,
                'verified': verify,
                'checksum': self._calculate_checksum(backup_file)
            }
            
            metadata_file = backup_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Backup completed successfully: {backup_file}")
            self.logger.info(f"Backup size: {self._format_size(backup_file.stat().st_size)}")
            
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise BackupError(f"Failed to create backup: {e}") from e

    def _create_backup_with_progress(self, backup_file: Path, source_size: int):
        """Create backup with progress reporting."""
        if self.enable_compression:
            self._create_compressed_backup(backup_file, source_size)
        else:
            self._create_simple_backup(backup_file, source_size)

    def _create_simple_backup(self, backup_file: Path, source_size: int):
        """Create uncompressed backup with progress."""
        copied = 0
        chunk_size = 64 * 1024  # 64KB chunks
        
        with open(self.db_path, 'rb') as src, open(backup_file, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                
                dst.write(chunk)
                copied += len(chunk)
                
                # Report progress
                progress = (copied / source_size) * 100
                self._report_progress(progress)

    def _create_compressed_backup(self, backup_file: Path, source_size: int):
        """Create compressed backup with progress."""
        copied = 0
        chunk_size = 64 * 1024  # 64KB chunks
        
        with open(self.db_path, 'rb') as src, gzip.open(backup_file, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                
                dst.write(chunk)
                copied += len(chunk)
                
                # Report progress
                progress = (copied / source_size) * 100
                self._report_progress(progress)

    def _report_progress(self, progress: float):
        """Report backup progress."""
        if progress % 10 < 1:  # Report every 10%
            self.logger.info(f"Backup progress: {progress:.1f}%")

    def restore_backup(self, backup_file: str, target_db: Optional[str] = None, 
                      force: bool = False) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_file: Path to backup file
            target_db: Optional target database path (defaults to original)
            force: Whether to overwrite existing database
            
        Returns:
            True if restore successful
            
        Raises:
            RestoreError: If restore fails
        """
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                raise RestoreError(f"Backup file not found: {backup_file}")
            
            # Determine target database path
            if target_db is None:
                target_db = str(self.db_path)
            target_path = Path(target_db)
            
            # Safety checks
            if target_path.exists() and not force:
                raise RestoreError(f"Target database exists: {target_path}. Use --force to overwrite.")
            
            self.logger.info(f"Restoring backup {backup_file} to {target_path}")
            
            # Verify backup before restore
            if not self.verify_backup(backup_file):
                raise RestoreError("Backup verification failed - cannot restore corrupted backup")
            
            # Create backup of existing database if it exists
            if target_path.exists():
                backup_existing = target_path.with_suffix('.db.pre_restore')
                shutil.copy2(target_path, backup_existing)
                self.logger.info(f"Existing database backed up to: {backup_existing}")
            
            # Restore based on file type
            if backup_path.suffix == '.gz':
                self._restore_compressed_backup(backup_path, target_path)
            else:
                self._restore_simple_backup(backup_path, target_path)
            
            # Verify restored database
            if not self._verify_database_integrity(target_path):
                raise RestoreError("Restored database failed integrity check")
            
            self.logger.info(f"Database restored successfully to: {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}")
            raise RestoreError(f"Failed to restore database: {e}") from e

    def _restore_simple_backup(self, backup_path: Path, target_path: Path):
        """Restore from uncompressed backup."""
        shutil.copy2(backup_path, target_path)

    def _restore_compressed_backup(self, backup_path: Path, target_path: Path):
        """Restore from compressed backup."""
        with gzip.open(backup_path, 'rb') as src, open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    def verify_backup(self, backup_file: str) -> bool:
        """
        Verify backup file integrity.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if backup is valid
        """
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Check metadata if available
            metadata_file = backup_path.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Verify checksum
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != metadata.get('checksum'):
                    self.logger.error("Backup checksum mismatch - file may be corrupted")
                    return False
            
            # Test database file by attempting to read it
            if backup_path.suffix == '.gz':
                return self._verify_compressed_backup(backup_path)
            else:
                return self._verify_database_integrity(backup_path)
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False

    def _verify_compressed_backup(self, backup_path: Path) -> bool:
        """Verify compressed backup by extracting to temp file."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.db') as temp_file:
                with gzip.open(backup_path, 'rb') as src:
                    shutil.copyfileobj(src, temp_file)
                temp_file.flush()
                
                return self._verify_database_integrity(Path(temp_file.name))
                
        except Exception as e:
            self.logger.error(f"Compressed backup verification failed: {e}")
            return False

    def _verify_database_integrity(self, db_path: Path) -> bool:
        """Verify SQLite database integrity."""
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                if result and result[0] == 'ok':
                    return True
                else:
                    self.logger.error(f"Database integrity check failed: {result}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Database integrity check error: {e}")
            return False

    def cleanup_old_backups(self, dry_run: bool = False) -> List[str]:
        """
        Clean up old backup files based on retention policy.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of deleted files
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            deleted_files = []
            
            self.logger.info(f"Cleaning up backups older than {cutoff_date}")
            
            for backup_file in self.backup_dir.glob("echoforge_backup_*.db*"):
                if backup_file.suffix == '.json':
                    continue  # Skip metadata files, they'll be handled with their backup
                
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    if dry_run:
                        self.logger.info(f"Would delete: {backup_file}")
                    else:
                        # Delete backup file
                        backup_file.unlink()
                        deleted_files.append(str(backup_file))
                        
                        # Delete associated metadata file
                        metadata_file = backup_file.with_suffix('.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                            deleted_files.append(str(metadata_file))
                        
                        self.logger.info(f"Deleted old backup: {backup_file}")
            
            if dry_run:
                self.logger.info(f"Dry run completed - {len(deleted_files)} files would be deleted")
            else:
                self.logger.info(f"Cleanup completed - {len(deleted_files)} files deleted")
            
            return deleted_files
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
            return []

    def list_backups(self) -> List[Dict[str, any]]:
        """
        List all available backups with metadata.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("echoforge_backup_*.db*")):
            if backup_file.suffix == '.json':
                continue
            
            backup_info = {
                'file': str(backup_file),
                'size': backup_file.stat().st_size,
                'size_formatted': self._format_size(backup_file.stat().st_size),
                'created': datetime.fromtimestamp(backup_file.stat().st_mtime),
                'compressed': backup_file.suffix == '.gz'
            }
            
            # Load metadata if available
            metadata_file = backup_file.with_suffix('.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    backup_info.update(metadata)
                except Exception as e:
                    self.logger.warning(f"Could not read metadata for {backup_file}: {e}")
            
            backups.append(backup_info)
        
        return backups

    def optimize_database(self) -> bool:
        """
        Optimize the database by running VACUUM and ANALYZE.
        
        Returns:
            True if optimization successful
        """
        try:
            self.logger.info("Starting database optimization...")
            
            # Create backup before optimization
            backup_file = self.create_backup("pre_optimization", verify=True)
            self.logger.info(f"Pre-optimization backup created: {backup_file}")
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get initial database size
                initial_size = self.db_path.stat().st_size
                
                # Run VACUUM to reclaim space
                self.logger.info("Running VACUUM...")
                conn.execute("VACUUM")
                
                # Run ANALYZE to update statistics
                self.logger.info("Running ANALYZE...")
                conn.execute("ANALYZE")
                
                # Get final database size
                final_size = self.db_path.stat().st_size
                space_saved = initial_size - final_size
                
                self.logger.info(f"Database optimization completed")
                self.logger.info(f"Initial size: {self._format_size(initial_size)}")
                self.logger.info(f"Final size: {self._format_size(final_size)}")
                self.logger.info(f"Space saved: {self._format_size(space_saved)}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return False

    def get_database_stats(self) -> Dict[str, any]:
        """
        Get database statistics and health information.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            if not self.db_path.exists():
                return {'error': 'Database file not found'}
            
            # File system stats
            file_stat = self.db_path.stat()
            stats['file_size'] = file_stat.st_size
            stats['file_size_formatted'] = self._format_size(file_stat.st_size)
            stats['last_modified'] = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Database stats
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get table count and row counts
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                stats['table_count'] = len(tables)
                
                table_stats = {}
                total_rows = 0
                
                for table in tables:
                    table_name = table[0]
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                        row_count = cursor.fetchone()[0]
                        table_stats[table_name] = row_count
                        total_rows += row_count
                    except Exception as e:
                        table_stats[table_name] = f"Error: {e}"
                
                stats['tables'] = table_stats
                stats['total_rows'] = total_rows
                
                # Get database page information
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                stats['page_count'] = page_count
                stats['page_size'] = page_size
                stats['calculated_size'] = page_count * page_size
                
                # Get integrity check result
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                stats['integrity'] = integrity
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class BackupError(Exception):
    """Exception raised for backup-related errors."""
    pass


class RestoreError(Exception):
    """Exception raised for restore-related errors."""
    pass


def main():
    """Command-line interface for backup operations."""
    parser = argparse.ArgumentParser(
        description="EchoForge Database Backup and Maintenance Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s backup                     # Create a backup
  %(prog)s backup --name my_backup    # Create named backup
  %(prog)s restore backup.db          # Restore from backup
  %(prog)s list                       # List all backups
  %(prog)s cleanup                    # Clean up old backups
  %(prog)s optimize                   # Optimize database
  %(prog)s stats                      # Show database statistics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--name', help='Custom backup name')
    backup_parser.add_argument('--no-verify', action='store_true', help='Skip backup verification')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database from backup')
    restore_parser.add_argument('backup_file', help='Path to backup file')
    restore_parser.add_argument('--target', help='Target database path')
    restore_parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')
    list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup_file', help='Path to backup file')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize database')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize backup manager
        backup_manager = BackupManager()
        
        # Execute command
        if args.command == 'backup':
            backup_file = backup_manager.create_backup(
                backup_name=args.name,
                verify=not args.no_verify
            )
            print(f"Backup created: {backup_file}")
            
        elif args.command == 'restore':
            success = backup_manager.restore_backup(
                backup_file=args.backup_file,
                target_db=args.target,
                force=args.force
            )
            if success:
                print("Database restored successfully")
            else:
                print("Database restore failed")
                return 1
                
        elif args.command == 'list':
            backups = backup_manager.list_backups()
            if args.json:
                print(json.dumps(backups, indent=2, default=str))
            else:
                if not backups:
                    print("No backups found")
                else:
                    print(f"{'Backup File':<40} {'Size':<10} {'Created':<20} {'Compressed':<10}")
                    print("-" * 80)
                    for backup in backups:
                        compressed = "Yes" if backup['compressed'] else "No"
                        print(f"{Path(backup['file']).name:<40} {backup['size_formatted']:<10} "
                              f"{backup['created'].strftime('%Y-%m-%d %H:%M'):<20} {compressed:<10}")
                              
        elif args.command == 'cleanup':
            deleted = backup_manager.cleanup_old_backups(dry_run=args.dry_run)
            if args.dry_run:
                print(f"Would delete {len(deleted)} backup files")
            else:
                print(f"Deleted {len(deleted)} old backup files")
                
        elif args.command == 'verify':
            if backup_manager.verify_backup(args.backup_file):
                print("Backup verification successful")
            else:
                print("Backup verification failed")
                return 1
                
        elif args.command == 'optimize':
            if backup_manager.optimize_database():
                print("Database optimization completed")
            else:
                print("Database optimization failed")
                return 1
                
        elif args.command == 'stats':
            stats = backup_manager.get_database_stats()
            if args.json:
                print(json.dumps(stats, indent=2, default=str))
            else:
                if 'error' in stats:
                    print(f"Error: {stats['error']}")
                    return 1
                
                print("Database Statistics:")
                print(f"  File size: {stats['file_size_formatted']}")
                print(f"  Last modified: {stats['last_modified']}")
                print(f"  Tables: {stats['table_count']}")
                print(f"  Total rows: {stats['total_rows']}")
                print(f"  Page count: {stats['page_count']}")
                print(f"  Page size: {stats['page_size']} bytes")
                print(f"  Integrity: {stats['integrity']}")
                
                if stats['tables']:
                    print("\nTable row counts:")
                    for table, count in stats['tables'].items():
                        print(f"  {table}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
