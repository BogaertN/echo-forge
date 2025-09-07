import logging
import json
from typing import Dict, List
from datetime import datetime, timedelta
from encrypted_db import EncryptedDB
from config import load_config
from utils import send_notification  # Assume utility for user notifications

logger = logging.getLogger(__name__)

class GamificationEngine:
    """
    Handles gamification features: journaling streaks, badges for ghost loop closure/consistency/depth,
    progress dashboards, clarity metrics, weekly reports, and motivation without distraction.
    Integrates with DB for persistence and journaling for triggers.
    """
    
    def __init__(self, db: EncryptedDB):
        self.db = db
        self.config = load_config()
        self.badges = self._define_badges()
        
        logger.info("GamificationEngine initialized")
    
    def _define_badges(self) -> Dict:
        """Define available badges and their criteria"""
        return {
            'consistency': {
                'description': 'Achieved a 7-day journaling streak',
                'criteria': {'streak': 7}
            },
            'ghost_hunter': {
                'description': 'Closed 5 ghost loops',
                'criteria': {'ghost_closures': 5}
            },
            'deep_thinker': {
                'description': 'Journal entry with depth score > 8',
                'criteria': {'depth_score': 8}
            },
            'clarity_master': {
                'description': 'Weekly average clarity > 7',
                'criteria': {'weekly_clarity': 7}
            }
            # Extendable for more badges
        }
    
    # Streaks and Reports
    
    def update_streaks(self):
        """Update journaling streaks based on recent activity"""
        last_date = self.db.get_last_journal_date()  # Assume DB method
        current_streak = self.db.get_current_streak()
        
        today = datetime.now().date()
        if last_date:
            last_date = datetime.fromisoformat(last_date).date()
            if (today - last_date).days == 1:
                new_streak = current_streak + 1
                self.db.update_streak(new_streak, today.isoformat())
                self._check_badge('consistency', new_streak)
            elif (today - last_date).days > 1:
                self.db.update_streak(1, today.isoformat())
        
        logger.info(f"Streaks updated: current={new_streak}")
    
    def generate_weekly_report(self) -> Dict:
        """Generate weekly clarity snapshot and metrics"""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        entries = self.db.get_entries_since(week_ago)  # Assume DB method
        
        if not entries:
            return {'message': 'No entries this week'}
        
        clarity_scores = [e['weights'].get('clarity', 0) for e in entries]
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        
        report = {
            'entries_count': len(entries),
            'avg_clarity': round(avg_clarity, 2),
            'ghost_closures': self.db.count_ghost_closures(week_ago),
            'streak': self.db.get_current_streak(),
            'insights': self._generate_insights(avg_clarity, len(entries)),
            'generated_at': datetime.now().isoformat()
        }
        
        self.db.save_weekly_report(report)
        self._check_badge('clarity_master', avg_clarity)
        
        # Send gentle notification if configured
        if self.config['gamification']['notify_reports']:
            send_notification("Your weekly clarity report is ready!")
        
        logger.info("Weekly report generated")
        return report
    
    def _generate_insights(self, avg_clarity: float, entry_count: int) -> List[str]:
        """Generate motivational insights"""
        insights = []
        if entry_count > 5:
            insights.append("Great consistency this week!")
        if avg_clarity > 7:
            insights.append("Your reflections show high clarity—keep it up!")
        elif avg_clarity < 5:
            insights.append("Consider deeper reflections for clearer insights.")
        return insights
    
    # Badges
    
    def award_badge(self, badge_key: str):
        """Award a badge if not already earned"""
        current_badges = self.db.get_earned_badges()  # Assume list from DB
        if badge_key not in current_badges:
            current_badges.append(badge_key)
            self.db.update_badges(current_badges)
            send_notification(f"Badge earned: {self.badges[badge_key]['description']}")
            logger.info(f"Badge awarded: {badge_key}")
    
    def _check_badge(self, badge_key: str, value: float or int):
        """Check if badge criteria met"""
        criteria = self.badges.get(badge_key, {}).get('criteria', {})
        for key, threshold in criteria.items():
            if value >= threshold:
                self.award_badge(badge_key)
                break
    
    def award_ghost_closure(self):
        """Award on ghost loop closure"""
        closures = self.db.get_total_ghost_closures()  # Assume DB count
        self._check_badge('ghost_hunter', closures)
    
    def award_depth(self, entry_id: str):
        """Check depth for new/updated entry"""
        entry = self.db.get_journal_entry(entry_id)
        depth = entry['weights'].get('depth', 0)  # Assume depth in weights
        self._check_badge('deep_thinker', depth)
    
    # Progress Dashboards and Metrics
    
    def get_progress_dashboard(self) -> Dict:
        """Get user progress dashboard data"""
        stats = self.db.get_gamification_stats()
        recent_reports = self.db.get_recent_weekly_reports(4)  # Last 4 weeks
        
        dashboard = {
            'current_streak': stats['streak_count'],
            'earned_badges': stats['badges'],
            'clarity_trend': [r['avg_clarity'] for r in recent_reports],
            'total_entries': self.db.get_total_entries(),
            'ghost_loops_open': len(self.db.find_ghost_loops()),
            'clarity_metrics': self._compute_clarity_metrics(),
            'motivation_tip': self._get_motivation_tip(stats['streak_count'])
        }
        
        return dashboard
    
    def _compute_clarity_metrics(self) -> Dict:
        """Compute overall clarity metrics"""
        all_entries = self.db.get_all_entries()  # Limited if needed
        clarities = [e['weights'].get('clarity', 0) for e in all_entries]
        if clarities:
            avg = sum(clarities) / len(clarities)
            max_clarity = max(clarities)
            return {'average': round(avg, 2), 'highest': max_clarity}
        return {'average': 0, 'highest': 0}
    
    def _get_motivation_tip(self, streak: int) -> str:
        """Get gentle motivation tip without distraction"""
        if streak == 0:
            return "Start a streak today with a quick reflection."
        elif streak < 3:
            return "Building momentum—keep going!"
        else:
            return "Impressive streak! Your consistency is paying off."
    
    # Integration Hooks
    
    def update_on_entry(self):
        """Hook called on new journal entry"""
        self.update_streaks()
        # Other updates as needed
    
    def generate_report_if_due(self):
        """Check and generate weekly report if due"""
        last_report = self.db.get_last_report_date()
        if (datetime.now() - datetime.fromisoformat(last_report)).days >= 7:
            self.generate_weekly_report()
