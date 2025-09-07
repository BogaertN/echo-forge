import logging
import uuid
import json
from typing import Dict, List, Optional
from datetime import datetime
from encrypted_db import EncryptedDB
from agents import JournalingAssistant
from config import load_config
from utils import validate_metadata

logger = logging.getLogger(__name__)

class JournalManager:
    """
    Manages journaling operations: entry creation, metadata generation,
    search/retrieval, voice journaling, auto-suggestions, streaks/badges integration,
    backup/restore, and privacy controls. Integrates with JournalingAssistant agent.
    """
    
    def __init__(self, db: EncryptedDB):
        self.db = db
        self.config = load_config()
        self.journaling_assistant = JournalingAssistant(self.config['models']['journaling_assistant'])
        
        logger.info("JournalManager initialized")
    
    # Entry Creation Flow
    
    def create_entry(self, content: str, metadata: Dict, session_id: str, 
                     debate_id: Optional[str] = None, user_edits: Optional[str] = None,
                     voice_transcription: bool = False) -> str:
        """
        Create a new journal entry with auto-generated metadata and rephrasing.
        
        Args:
            content: Raw content (text or transcribed voice)
            metadata: Initial metadata (will be enhanced by agent)
            session_id: Associated session ID
            debate_id: Optional linked debate ID
            user_edits: Optional user modifications
            voice_transcription: Flag if content came from voice
        
        Returns:
            Entry ID
        """
        # Rephrase content if needed
        if self.config['journal']['auto_rephrase'] or voice_transcription:
            content = self.journaling_assistant.rephrase_for_journal(content)
        
        # Generate/enhance metadata
        enhanced_metadata = self.journaling_assistant.generate_metadata(
            content,
            context={
                'session_id': session_id,
                'debate_id': debate_id,
                'user_edits': bool(user_edits),
                'voice': voice_transcription
            }
        )
        enhanced_metadata.update(metadata)  # Merge with provided metadata
        
        validate_metadata(enhanced_metadata)  # Utility validation
        
        # Save to DB
        entry_id = self.db.create_journal_entry(
            content=content,
            metadata=enhanced_metadata,
            session_id=session_id,
            debate_id=debate_id,
            user_edits=user_edits
        )
        
        # Auto-suggest tags/connections if enabled
        if self.config['journal']['auto_suggest']:
            self._generate_auto_suggestions(entry_id)
        
        logger.info(f"Journal entry created: {entry_id}")
        return entry_id
    
    def _generate_auto_suggestions(self, entry_id: str):
        """Generate auto-suggestions for tags, weights, connections"""
        entry = self.db.get_journal_entry(entry_id)  # Assume method added to EncryptedDB
        suggestions = self.journaling_assistant.generate_metadata(entry['content'], {'suggest_only': True})
        
        # Update entry with suggestions (as separate field or notify user)
        self.db.update_journal_entry(entry_id, {'auto_suggestions': suggestions})
    
    # Voice Journaling
    
    def create_voice_entry(self, audio_path: str, session_id: str, 
                           debate_id: Optional[str] = None) -> str:
        """Create entry from voice audio using Whisper transcription"""
        transcription = self.journaling_assistant.transcribe_voice(audio_path)
        
        # Generate metadata with voice context
        metadata = {'source': 'voice', 'transcription_length': len(transcription)}
        
        return self.create_entry(
            content=transcription,
            metadata=metadata,
            session_id=session_id,
            debate_id=debate_id,
            voice_transcription=True
        )
    
    # Search and Retrieval
    
    def search_entries(self, query: str = "", tags: List[str] = None, 
                       ghost_loops_only: bool = False, date_range: Optional[Dict] = None,
                       sort_by: str = "created_at DESC", limit: int = 50) -> List[Dict]:
        """Advanced search for journal entries"""
        results = self.db.search_journal(
            query=query,
            tags=tags,
            ghost_loops_only=ghost_loops_only,
            limit=limit
        )
        
        # Apply date range filter if provided
        if date_range:
            start = datetime.fromisoformat(date_range.get('start'))
            end = datetime.fromisoformat(date_range.get('end'))
            results = [r for r in results if start <= datetime.fromisoformat(r['created_at']) <= end]
        
        # Custom sort if needed
        if sort_by == "relevance DESC":
            results.sort(key=lambda r: r['weights'].get('relevance', 0), reverse=True)
        
        return results
    
    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Retrieve single entry"""
        return self.db.get_journal_entry(entry_id)  # Assume implemented in DB
    
    # Update and Edit
    
    def update_entry(self, entry_id: str, updates: Dict) -> bool:
        """Update entry content or metadata"""
        if 'content' in updates:
            # Re-generate metadata if content changes
            new_metadata = self.journaling_assistant.generate_metadata(updates['content'])
            updates['metadata'] = new_metadata
        
        success = self.db.update_journal_entry(entry_id, updates)
        if success:
            logger.info(f"Journal entry updated: {entry_id}")
        return success
    
    def close_ghost_loop(self, entry_id: str, resolution: str) -> bool:
        """Close a ghost loop entry"""
        updates = {
            'ghost_loop': False,
            'ghost_loop_reason': '',  # Clear reason
            'resolution': resolution  # Add resolution field, assume added to schema
        }
        success = self.update_entry(entry_id, updates)
        if success:
            # Award gamification points/badges
            self.gamification.award_ghost_closure()  # Assume method in GamificationEngine
        return success
    
    # Gamification Integration (hooks)
    
    def update_streaks(self):
        """Update journaling streaks (called periodically or on create)"""
        self.gamification.update_streaks()
    
    def generate_weekly_report(self) -> Dict:
        """Generate weekly clarity snapshot"""
        return self.gamification.generate_weekly_report()
    
    # Privacy and Data Handling
    
    def export_entries(self, format: str = 'json', encrypted: bool = True) -> str:
        """Export journal data"""
        all_entries = self.search_entries(limit=999999)
        data = json.dumps(all_entries, indent=2)
        
        if encrypted:
            # Simple encryption example; use proper crypto in production
            encrypted_data = self._encrypt_data(data)
            file_path = f"journal_export_{datetime.now().strftime('%Y%m%d')}.enc"
            with open(file_path, 'w') as f:
                f.write(encrypted_data)
        else:
            file_path = f"journal_export_{datetime.now().strftime('%Y%m%d')}.json"
            with open(file_path, 'w') as f:
                f.write(data)
        
        logger.info(f"Journal exported to {file_path}")
        return file_path
    
    def _encrypt_data(self, data: str) -> str:
        """Placeholder for data encryption"""
        # Use cryptography library in production
        return data  # TODO: Implement proper encryption
    
    def backup_journal(self) -> str:
        """Backup journal data"""
        return self.db.backup_database()
    
    def restore_journal(self, backup_path: str):
        """Restore from backup"""
        self.db.restore_database(backup_path)
        logger.info("Journal restored")
    
    # Advanced Features
    
    def auto_tag_entries(self, batch_size: int = 50):
        """Auto-tag untagged entries using agent"""
        untagged = self.search_entries(tags=[])
        for entry in untagged[:batch_size]:
            new_metadata = self.journaling_assistant.generate_metadata(entry['content'])
            self.update_entry(entry['id'], {'tags': new_metadata['tags']})
        
        logger.info(f"Auto-tagged {len(untagged[:batch_size])} entries")
    
    def analyze_journal_trends(self) -> Dict:
        """Analyze trends in journal (e.g., emotion over time)"""
        entries = self.search_entries(limit=1000)
        emotion_trend = []
        for entry in entries:
            emotion_trend.append({
                'date': entry['created_at'][:10],
                'emotion': entry['weights'].get('emotion', 0)
            })
        
        # Simple average by date
        from collections import defaultdict
        avg_emotion = defaultdict(list)
        for e in emotion_trend:
            avg_emotion[e['date']].append(e['emotion'])
        
        trend_data = {date: sum(values)/len(values) for date, values in avg_emotion.items()}
        
        return {'emotion_trend': trend_data}
