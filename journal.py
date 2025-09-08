import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import hashlib
import statistics

from utils import (
    extract_key_concepts, calculate_similarity, generate_uuid,
    count_words, estimate_reading_time, clean_text, extract_hashtags
)

logger = logging.getLogger(__name__)

@dataclass
class JournalEntry:
    """Journal entry data structure"""
    id: str
    session_id: str
    debate_id: Optional[str]
    title: str
    content: str
    summary: str
    insights: List[str]
    personal_reflections: str
    action_items: List[str]
    tags: List[str]
    mood_rating: Optional[int]  # 1-10 scale
    complexity_rating: Optional[int]  # 1-10 scale
    satisfaction_rating: Optional[int]  # 1-10 scale
    word_count: int
    reading_time_minutes: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JournalEntry':
        """Create from dictionary"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

@dataclass
class SearchQuery:
    """Journal search query structure"""
    text: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    mood_range: Optional[Tuple[int, int]] = None
    complexity_range: Optional[Tuple[int, int]] = None
    satisfaction_range: Optional[Tuple[int, int]] = None
    has_debate: Optional[bool] = None
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None
    sort_by: str = "created_at"  # created_at, updated_at, word_count, title
    sort_order: str = "desc"  # asc, desc
    limit: int = 20
    offset: int = 0

@dataclass
class GamificationStats:
    """User gamification statistics"""
    session_id: str
    total_debates: int
    total_journal_entries: int
    total_questions_clarified: int
    current_streak_days: int
    longest_streak_days: int
    last_activity_date: Optional[date]
    total_words_written: int
    total_insights_generated: int
    level: int
    experience_points: int
    badges_earned: List[str]
    achievements: List[str]
    preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class Badge:
    """Badge definition"""
    id: str
    name: str
    description: str
    category: str
    icon: Optional[str]
    requirements: Dict[str, Any]
    points_reward: int
    rarity: str  # common, rare, epic, legendary
    created_at: datetime

class JournalAnalytics:
    """Analytics and insights for journal entries"""
    
    def __init__(self, entries: List[JournalEntry]):
        self.entries = entries
    
    def get_writing_patterns(self) -> Dict[str, Any]:
        """Analyze writing patterns over time"""
        if not self.entries:
            return {}
        
        # Group entries by date
        daily_counts = defaultdict(int)
        daily_words = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for entry in self.entries:
            entry_date = entry.created_at.date()
            entry_hour = entry.created_at.hour
            
            daily_counts[entry_date] += 1
            daily_words[entry_date] += entry.word_count
            hourly_distribution[entry_hour] += 1
        
        # Calculate statistics
        word_counts = [entry.word_count for entry in self.entries]
        
        return {
            "total_entries": len(self.entries),
            "total_words": sum(word_counts),
            "average_words_per_entry": statistics.mean(word_counts) if word_counts else 0,
            "median_words_per_entry": statistics.median(word_counts) if word_counts else 0,
            "most_productive_day": max(daily_counts.items(), key=lambda x: x[1])[0].isoformat() if daily_counts else None,
            "most_productive_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            "daily_average": statistics.mean(daily_counts.values()) if daily_counts else 0,
            "writing_frequency": len(daily_counts),  # Number of days with entries
            "longest_entry": max(word_counts) if word_counts else 0,
            "shortest_entry": min(word_counts) if word_counts else 0
        }
    
    def get_mood_trends(self) -> Dict[str, Any]:
        """Analyze mood trends over time"""
        mood_entries = [e for e in self.entries if e.mood_rating is not None]
        
        if not mood_entries:
            return {"has_data": False}
        
        moods = [e.mood_rating for e in mood_entries]
        
        # Group by date for trend analysis
        daily_moods = defaultdict(list)
        for entry in mood_entries:
            daily_moods[entry.created_at.date()].append(entry.mood_rating)
        
        # Calculate daily averages
        daily_mood_averages = {
            date: statistics.mean(moods)
            for date, moods in daily_moods.items()
        }
        
        return {
            "has_data": True,
            "average_mood": statistics.mean(moods),
            "median_mood": statistics.median(moods),
            "mood_range": (min(moods), max(moods)),
            "mood_distribution": dict(Counter(moods)),
            "trend_data": [
                {"date": date.isoformat(), "mood": mood}
                for date, mood in sorted(daily_mood_averages.items())
            ],
            "entries_with_mood": len(mood_entries),
            "total_entries": len(self.entries)
        }
    
    def get_topic_analysis(self) -> Dict[str, Any]:
        """Analyze topics and themes in journal entries"""
        # Extract all concepts and tags
        all_concepts = []
        all_tags = []
        
        for entry in self.entries:
            # Extract concepts from content
            concepts = extract_key_concepts(entry.content + " " + entry.title)
            all_concepts.extend(concepts)
            all_tags.extend(entry.tags)
        
        # Count frequency
        concept_frequency = Counter(all_concepts)
        tag_frequency = Counter(all_tags)
        
        return {
            "most_common_concepts": concept_frequency.most_common(20),
            "most_common_tags": tag_frequency.most_common(10),
            "unique_concepts": len(concept_frequency),
            "unique_tags": len(tag_frequency),
            "total_concepts": len(all_concepts),
            "total_tags": len(all_tags)
        }
    
    def get_productivity_insights(self) -> Dict[str, Any]:
        """Generate productivity insights and recommendations"""
        patterns = self.get_writing_patterns()
        mood_trends = self.get_mood_trends()
        
        insights = []
        recommendations = []
        
        # Analyze writing frequency
        if patterns.get("writing_frequency", 0) > 20:  # Regular writer
            insights.append("You maintain a consistent journaling habit.")
            recommendations.append("Consider setting weekly reflection goals to deepen insights.")
        elif patterns.get("writing_frequency", 0) > 5:
            insights.append("You journal regularly but could benefit from more consistency.")
            recommendations.append("Try setting a daily reminder to maintain momentum.")
        else:
            insights.append("You're just getting started with journaling.")
            recommendations.append("Start with short, daily entries to build the habit.")
        
        # Analyze word count patterns
        avg_words = patterns.get("average_words_per_entry", 0)
        if avg_words > 500:
            insights.append("You tend to write detailed, comprehensive entries.")
            recommendations.append("Consider summarizing key insights at the end of longer entries.")
        elif avg_words > 200:
            insights.append("You write substantial entries with good detail.")
        else:
            insights.append("Your entries are concise and focused.")
            recommendations.append("Consider expanding on key insights for deeper reflection.")
        
        # Analyze mood data
        if mood_trends.get("has_data"):
            avg_mood = mood_trends.get("average_mood", 0)
            if avg_mood > 7:
                insights.append("Your journal reflects generally positive experiences.")
            elif avg_mood < 4:
                insights.append("Consider using journaling to explore challenging experiences.")
                recommendations.append("Try gratitude journaling or positive reflection exercises.")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "productivity_score": min(100, patterns.get("writing_frequency", 0) * 5),
            "consistency_rating": self._calculate_consistency_rating(),
            "growth_areas": self._identify_growth_areas()
        }
    
    def _calculate_consistency_rating(self) -> str:
        """Calculate consistency rating based on writing patterns"""
        if not self.entries:
            return "starting"
        
        # Calculate days between entries
        dates = sorted([e.created_at.date() for e in self.entries])
        if len(dates) < 2:
            return "starting"
        
        # Calculate gaps between entries
        gaps = []
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            gaps.append(gap)
        
        avg_gap = statistics.mean(gaps)
        
        if avg_gap <= 2:
            return "excellent"
        elif avg_gap <= 5:
            return "good"
        elif avg_gap <= 10:
            return "fair"
        else:
            return "needs_improvement"
    
    def _identify_growth_areas(self) -> List[str]:
        """Identify areas for improvement"""
        areas = []
        
        # Check for mood tracking
        mood_entries = [e for e in self.entries if e.mood_rating is not None]
        if len(mood_entries) / len(self.entries) < 0.5:
            areas.append("mood_tracking")
        
        # Check for action items
        action_entries = [e for e in self.entries if e.action_items]
        if len(action_entries) / len(self.entries) < 0.3:
            areas.append("actionable_insights")
        
        # Check for tags
        tagged_entries = [e for e in self.entries if e.tags]
        if len(tagged_entries) / len(self.entries) < 0.5:
            areas.append("categorization")
        
        # Check for insights
        insight_entries = [e for e in self.entries if e.insights]
        if len(insight_entries) / len(self.entries) < 0.4:
            areas.append("reflection_depth")
        
        return areas

class BadgeSystem:
    """Gamification badge system"""
    
    def __init__(self, db):
        self.db = db
    
    async def check_and_award_badges(self, session_id: str, stats: GamificationStats) -> List[str]:
        """Check and award new badges to user"""
        with self.db.get_connection() as conn:
            # Get all available badges
            cursor = conn.execute("SELECT * FROM badges")
            all_badges = [dict(row) for row in cursor.fetchall()]
            
            # Get user's current badges
            cursor = conn.execute("""
                SELECT badge_id FROM user_badges 
                WHERE session_id = ?
            """, (session_id,))
            earned_badge_ids = {row[0] for row in cursor.fetchall()}
            
            newly_earned = []
            
            for badge_dict in all_badges:
                badge_id = badge_dict["id"]
                
                if badge_id in earned_badge_ids:
                    continue  # Already earned
                
                # Parse requirements
                try:
                    requirements = json.loads(badge_dict["requirements"])
                except (json.JSONDecodeError, TypeError):
                    continue
                
                # Check if requirements are met
                if self._check_badge_requirements(requirements, stats):
                    # Award badge
                    await self._award_badge(session_id, badge_id, conn)
                    newly_earned.append(badge_id)
                    
                    logger.info(f"Badge '{badge_dict['name']}' awarded to session {session_id}")
            
            return newly_earned
    
    def _check_badge_requirements(self, requirements: Dict[str, Any], stats: GamificationStats) -> bool:
        """Check if badge requirements are met"""
        for req_key, req_value in requirements.items():
            if req_key == "debates_completed" and stats.total_debates < req_value:
                return False
            elif req_key == "journal_entries" and stats.total_journal_entries < req_value:
                return False
            elif req_key == "streak_days" and stats.current_streak_days < req_value:
                return False
            elif req_key == "words_written" and stats.total_words_written < req_value:
                return False
            elif req_key == "insights_generated" and stats.total_insights_generated < req_value:
                return False
            elif req_key == "questions_clarified" and stats.total_questions_clarified < req_value:
                return False
            elif req_key == "level" and stats.level < req_value:
                return False
        
        return True
    
    async def _award_badge(self, session_id: str, badge_id: str, conn):
        """Award badge to user"""
        conn.execute("""
            INSERT INTO user_badges (id, session_id, badge_id, context)
            VALUES (?, ?, ?, ?)
        """, (
            generate_uuid(),
            session_id,
            badge_id,
            "automatic_award"
        ))
        conn.commit()

class JournalManager:
    """
    Comprehensive journal management system for EchoForge.
    
    Handles journal entries, search, gamification, and analytics.
    """
    
    def __init__(self, db):
        self.db = db
        self.badge_system = BadgeSystem(db)
        self._search_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        logger.info("JournalManager initialized")
    
    async def create_entry(self, session_id: str, title: str, content: str,
                          insights: List[str] = None, personal_reflections: str = "",
                          action_items: List[str] = None, tags: List[str] = None,
                          mood_rating: Optional[int] = None,
                          complexity_rating: Optional[int] = None,
                          satisfaction_rating: Optional[int] = None,
                          debate_id: Optional[str] = None,
                          question: str = "", debate_summary: str = "",
                          metadata: Dict[str, Any] = None) -> str:
        """
        Create a new journal entry.
        
        Args:
            session_id: User session ID
            title: Entry title
            content: Main content
            insights: List of key insights
            personal_reflections: Personal thoughts and reflections
            action_items: List of actionable items
            tags: List of tags/categories
            mood_rating: Mood rating (1-10)
            complexity_rating: Complexity rating (1-10)
            satisfaction_rating: Satisfaction rating (1-10)
            debate_id: Associated debate ID
            question: Original question that led to this entry
            debate_summary: Summary of the debate
            metadata: Additional metadata
            
        Returns:
            Entry ID
        """
        try:
            # Generate entry ID
            entry_id = generate_uuid()
            
            # Process and validate inputs
            title = clean_text(title) or f"Journal Entry {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            content = clean_text(content)
            
            if not content:
                raise ValueError("Content is required for journal entry")
            
            # Extract metadata
            word_count = count_words(content)
            reading_time = estimate_reading_time(content)
            
            # Auto-extract tags from content if not provided
            if tags is None:
                tags = []
            
            # Extract hashtags from content
            hashtags = extract_hashtags(content)
            tags.extend(hashtags)
            
            # Auto-extract key concepts if insights not provided
            if insights is None:
                insights = extract_key_concepts(content + " " + title)[:10]  # Top 10 concepts
            
            # Generate summary if not provided
            summary = self._generate_summary(content, title)
            
            # Validate ratings
            mood_rating = self._validate_rating(mood_rating)
            complexity_rating = self._validate_rating(complexity_rating)
            satisfaction_rating = self._validate_rating(satisfaction_rating)
            
            # Prepare metadata
            entry_metadata = {
                "source": "manual",
                "has_debate": debate_id is not None,
                "original_question": question,
                "debate_summary": debate_summary,
                "auto_generated_insights": len(insights) if insights else 0,
                "extraction_timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Save to database
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO journal_entries (
                        id, session_id, debate_id, title, content, summary,
                        insights, personal_reflections, action_items, tags,
                        mood_rating, complexity_rating, satisfaction_rating,
                        word_count, reading_time_minutes, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id, session_id, debate_id, title, content, summary,
                    json.dumps(insights or []),
                    personal_reflections,
                    json.dumps(action_items or []),
                    json.dumps(tags),
                    mood_rating, complexity_rating, satisfaction_rating,
                    word_count, reading_time, json.dumps(entry_metadata)
                ))
                conn.commit()
            
            # Update gamification stats
            await self._update_gamification_stats(session_id, {
                "journal_entries": 1,
                "words_written": word_count,
                "insights_generated": len(insights) if insights else 0
            })
            
            # Clear search cache
            self._clear_search_cache(session_id)
            
            logger.info(f"Journal entry created: {entry_id} ({word_count} words)")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error creating journal entry: {e}")
            raise
    
    async def get_entry(self, entry_id: str, session_id: str) -> Optional[JournalEntry]:
        """Get specific journal entry"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM journal_entries 
                    WHERE id = ? AND session_id = ?
                """, (entry_id, session_id))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return self._row_to_entry(row)
                
        except Exception as e:
            logger.error(f"Error getting journal entry {entry_id}: {e}")
            return None
    
    async def update_entry(self, entry_id: str, session_id: str, 
                          updates: Dict[str, Any]) -> bool:
        """Update existing journal entry"""
        try:
            # Get current entry
            current_entry = await self.get_entry(entry_id, session_id)
            if not current_entry:
                return False
            
            # Prepare update fields
            update_fields = []
            update_values = []
            
            allowed_fields = {
                "title", "content", "summary", "insights", "personal_reflections",
                "action_items", "tags", "mood_rating", "complexity_rating",
                "satisfaction_rating", "metadata"
            }
            
            for field, value in updates.items():
                if field in allowed_fields:
                    if field in ["insights", "action_items", "tags", "metadata"]:
                        # JSON fields
                        update_fields.append(f"{field} = ?")
                        update_values.append(json.dumps(value))
                    else:
                        update_fields.append(f"{field} = ?")
                        update_values.append(value)
            
            # Update word count and reading time if content changed
            if "content" in updates:
                new_content = updates["content"]
                word_count = count_words(new_content)
                reading_time = estimate_reading_time(new_content)
                
                update_fields.extend(["word_count = ?", "reading_time_minutes = ?"])
                update_values.extend([word_count, reading_time])
            
            # Add updated timestamp
            update_fields.append("updated_at = ?")
            update_values.append(datetime.now())
            
            # Add WHERE clause values
            update_values.extend([entry_id, session_id])
            
            # Execute update
            with self.db.get_connection() as conn:
                conn.execute(f"""
                    UPDATE journal_entries 
                    SET {', '.join(update_fields)}
                    WHERE id = ? AND session_id = ?
                """, update_values)
                conn.commit()
            
            # Clear search cache
            self._clear_search_cache(session_id)
            
            logger.info(f"Journal entry updated: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating journal entry {entry_id}: {e}")
            return False
    
    async def delete_entry(self, entry_id: str, session_id: str) -> bool:
        """Delete journal entry"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM journal_entries 
                    WHERE id = ? AND session_id = ?
                """, (entry_id, session_id))
                
                if cursor.rowcount == 0:
                    return False
                
                conn.commit()
            
            # Clear search cache
            self._clear_search_cache(session_id)
            
            logger.info(f"Journal entry deleted: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting journal entry {entry_id}: {e}")
            return False
    
    async def search_entries(self, session_id: str, query: SearchQuery) -> Dict[str, Any]:
        """
        Search journal entries with advanced filtering.
        
        Args:
            session_id: User session ID
            query: Search query parameters
            
        Returns:
            Dictionary with entries, total count, and metadata
        """
        try:
            # Check cache
            cache_key = self._get_search_cache_key(session_id, query)
            if cache_key in self._search_cache:
                cache_entry = self._search_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self._cache_timeout:
                    return cache_entry["result"]
            
            # Build SQL query
            where_conditions = ["session_id = ?"]
            query_params = [session_id]
            
            # Text search
            if query.text:
                where_conditions.append("""
                    (title LIKE ? OR content LIKE ? OR summary LIKE ? 
                     OR personal_reflections LIKE ?)
                """)
                search_text = f"%{query.text}%"
                query_params.extend([search_text, search_text, search_text, search_text])
            
            # Tag filtering
            if query.tags:
                tag_conditions = []
                for tag in query.tags:
                    tag_conditions.append("tags LIKE ?")
                    query_params.append(f"%{tag}%")
                where_conditions.append(f"({' OR '.join(tag_conditions)})")
            
            # Date range filtering
            if query.date_from:
                where_conditions.append("date(created_at) >= ?")
                query_params.append(query.date_from.isoformat())
            
            if query.date_to:
                where_conditions.append("date(created_at) <= ?")
                query_params.append(query.date_to.isoformat())
            
            # Rating range filtering
            if query.mood_range:
                where_conditions.append("mood_rating BETWEEN ? AND ?")
                query_params.extend(query.mood_range)
            
            if query.complexity_range:
                where_conditions.append("complexity_rating BETWEEN ? AND ?")
                query_params.extend(query.complexity_range)
            
            if query.satisfaction_range:
                where_conditions.append("satisfaction_rating BETWEEN ? AND ?")
                query_params.extend(query.satisfaction_range)
            
            # Debate association filtering
            if query.has_debate is not None:
                if query.has_debate:
                    where_conditions.append("debate_id IS NOT NULL")
                else:
                    where_conditions.append("debate_id IS NULL")
            
            # Word count filtering
            if query.min_word_count:
                where_conditions.append("word_count >= ?")
                query_params.append(query.min_word_count)
            
            if query.max_word_count:
                where_conditions.append("word_count <= ?")
                query_params.append(query.max_word_count)
            
            # Build complete query
            base_sql = f"""
                FROM journal_entries 
                WHERE {' AND '.join(where_conditions)}
            """
            
            # Get total count
            with self.db.get_connection() as conn:
                count_cursor = conn.execute(f"SELECT COUNT(*) {base_sql}", query_params)
                total_count = count_cursor.fetchone()[0]
                
                # Get entries with sorting and pagination
                order_clause = f"ORDER BY {query.sort_by} {query.sort_order.upper()}"
                limit_clause = f"LIMIT {query.limit} OFFSET {query.offset}"
                
                entries_cursor = conn.execute(f"""
                    SELECT * {base_sql} {order_clause} {limit_clause}
                """, query_params)
                
                entries = [self._row_to_entry(row) for row in entries_cursor.fetchall()]
            
            # Prepare result
            result = {
                "entries": [entry.to_dict() for entry in entries],
                "total_count": total_count,
                "page_size": query.limit,
                "page_number": (query.offset // query.limit) + 1,
                "total_pages": (total_count + query.limit - 1) // query.limit,
                "has_more": (query.offset + query.limit) < total_count,
                "search_metadata": {
                    "query_text": query.text,
                    "filters_applied": len([f for f in [
                        query.tags, query.date_from, query.date_to,
                        query.mood_range, query.complexity_range, query.satisfaction_range,
                        query.has_debate, query.min_word_count, query.max_word_count
                    ] if f is not None and f != []]),
                    "search_time": time.time()
                }
            }
            
            # Cache result
            self._search_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching journal entries: {e}")
            return {
                "entries": [],
                "total_count": 0,
                "error": str(e)
            }
    
    async def get_entries(self, session_id: str, limit: int = 20, offset: int = 0,
                         search_query: str = None) -> List[Dict[str, Any]]:
        """Get journal entries with simple pagination"""
        query = SearchQuery(
            text=search_query,
            limit=limit,
            offset=offset,
            sort_by="created_at",
            sort_order="desc"
        )
        
        result = await self.search_entries(session_id, query)
        return result.get("entries", [])
    
    async def get_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for user's journal"""
        try:
            # Get all entries for user
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM journal_entries 
                    WHERE session_id = ?
                    ORDER BY created_at
                """, (session_id,))
                
                entries = [self._row_to_entry(row) for row in cursor.fetchall()]
            
            if not entries:
                return {"has_data": False, "message": "No journal entries found"}
            
            # Generate analytics
            analytics = JournalAnalytics(entries)
            
            return {
                "has_data": True,
                "writing_patterns": analytics.get_writing_patterns(),
                "mood_trends": analytics.get_mood_trends(),
                "topic_analysis": analytics.get_topic_analysis(),
                "productivity_insights": analytics.get_productivity_insights(),
                "summary": {
                    "total_entries": len(entries),
                    "date_range": {
                        "start": entries[0].created_at.isoformat(),
                        "end": entries[-1].created_at.isoformat()
                    },
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {"has_data": False, "error": str(e)}
    
    async def get_gamification_stats(self, session_id: str) -> GamificationStats:
        """Get gamification statistics for user"""
        try:
            with self.db.get_connection() as conn:
                # Get or create user stats
                cursor = conn.execute("""
                    SELECT * FROM user_stats WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                
                if row:
                    stats = GamificationStats(
                        session_id=row["session_id"],
                        total_debates=row["total_debates"],
                        total_journal_entries=row["total_journal_entries"],
                        total_questions_clarified=row["total_questions_clarified"],
                        current_streak_days=row["current_streak_days"],
                        longest_streak_days=row["longest_streak_days"],
                        last_activity_date=date.fromisoformat(row["last_activity_date"]) if row["last_activity_date"] else None,
                        total_words_written=row["total_words_written"],
                        total_insights_generated=row["total_insights_generated"],
                        level=row["level"],
                        experience_points=row["experience_points"],
                        badges_earned=json.loads(row["badges_earned"]),
                        achievements=json.loads(row["achievements"]),
                        preferences=json.loads(row["preferences"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"])
                    )
                else:
                    # Create new stats
                    stats = GamificationStats(
                        session_id=session_id,
                        total_debates=0,
                        total_journal_entries=0,
                        total_questions_clarified=0,
                        current_streak_days=0,
                        longest_streak_days=0,
                        last_activity_date=None,
                        total_words_written=0,
                        total_insights_generated=0,
                        level=1,
                        experience_points=0,
                        badges_earned=[],
                        achievements=[],
                        preferences={},
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    # Save to database
                    await self._save_gamification_stats(stats)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting gamification stats: {e}")
            # Return default stats on error
            return GamificationStats(
                session_id=session_id,
                total_debates=0, total_journal_entries=0, total_questions_clarified=0,
                current_streak_days=0, longest_streak_days=0, last_activity_date=None,
                total_words_written=0, total_insights_generated=0, level=1,
                experience_points=0, badges_earned=[], achievements=[],
                preferences={}, created_at=datetime.now(), updated_at=datetime.now()
            )
    
    async def _update_gamification_stats(self, session_id: str, updates: Dict[str, int]):
        """Update gamification statistics"""
        try:
            stats = await self.get_gamification_stats(session_id)
            
            # Update stats
            if "journal_entries" in updates:
                stats.total_journal_entries += updates["journal_entries"]
            if "words_written" in updates:
                stats.total_words_written += updates["words_written"]
            if "insights_generated" in updates:
                stats.total_insights_generated += updates["insights_generated"]
            if "debates" in updates:
                stats.total_debates += updates["debates"]
            if "questions_clarified" in updates:
                stats.total_questions_clarified += updates["questions_clarified"]
            
            # Update activity streak
            today = date.today()
            if stats.last_activity_date != today:
                if stats.last_activity_date == today - timedelta(days=1):
                    # Consecutive day
                    stats.current_streak_days += 1
                else:
                    # Streak broken or first activity
                    stats.current_streak_days = 1
                
                stats.last_activity_date = today
                
                # Update longest streak
                if stats.current_streak_days > stats.longest_streak_days:
                    stats.longest_streak_days = stats.current_streak_days
            
            # Calculate experience points and level
            base_points = (
                stats.total_journal_entries * 10 +
                stats.total_words_written // 100 +
                stats.total_insights_generated * 5 +
                stats.total_debates * 50 +
                stats.total_questions_clarified * 20 +
                stats.current_streak_days * 25
            )
            
            stats.experience_points = base_points
            stats.level = min(100, 1 + stats.experience_points // 1000)
            stats.updated_at = datetime.now()
            
            # Save updated stats
            await self._save_gamification_stats(stats)
            
            # Check for new badges
            newly_earned = await self.badge_system.check_and_award_badges(session_id, stats)
            
            if newly_earned:
                # Update badges in stats
                stats.badges_earned.extend(newly_earned)
                await self._save_gamification_stats(stats)
            
        except Exception as e:
            logger.error(f"Error updating gamification stats: {e}")
    
    async def _save_gamification_stats(self, stats: GamificationStats):
        """Save gamification statistics to database"""
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_stats (
                    session_id, total_debates, total_journal_entries,
                    total_questions_clarified, current_streak_days, longest_streak_days,
                    last_activity_date, total_words_written, total_insights_generated,
                    level, experience_points, badges_earned, achievements,
                    preferences, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.session_id, stats.total_debates, stats.total_journal_entries,
                stats.total_questions_clarified, stats.current_streak_days,
                stats.longest_streak_days,
                stats.last_activity_date.isoformat() if stats.last_activity_date else None,
                stats.total_words_written, stats.total_insights_generated,
                stats.level, stats.experience_points,
                json.dumps(stats.badges_earned), json.dumps(stats.achievements),
                json.dumps(stats.preferences),
                stats.created_at.isoformat(), stats.updated_at.isoformat()
            ))
            conn.commit()
    
    def _row_to_entry(self, row) -> JournalEntry:
        """Convert database row to JournalEntry object"""
        return JournalEntry(
            id=row["id"],
            session_id=row["session_id"],
            debate_id=row["debate_id"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            insights=json.loads(row["insights"]) if row["insights"] else [],
            personal_reflections=row["personal_reflections"] or "",
            action_items=json.loads(row["action_items"]) if row["action_items"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            mood_rating=row["mood_rating"],
            complexity_rating=row["complexity_rating"],
            satisfaction_rating=row["satisfaction_rating"],
            word_count=row["word_count"],
            reading_time_minutes=row["reading_time_minutes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )
    
    def _generate_summary(self, content: str, title: str) -> str:
        """Generate automatic summary of journal entry"""
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return title
        
        # Return first sentence or title if content is short
        if len(sentences) == 1 or len(content) < 200:
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
        
        # For longer content, take first and a middle sentence
        summary_parts = [sentences[0]]
        if len(sentences) > 2:
            summary_parts.append(sentences[len(sentences) // 2])
        
        summary = " ".join(summary_parts)
        return summary[:300] + "..." if len(summary) > 300 else summary
    
    def _validate_rating(self, rating: Optional[int]) -> Optional[int]:
        """Validate rating is within 1-10 range"""
        if rating is None:
            return None
        
        try:
            rating = int(rating)
            return max(1, min(10, rating))
        except (ValueError, TypeError):
            return None
    
    def _get_search_cache_key(self, session_id: str, query: SearchQuery) -> str:
        """Generate cache key for search query"""
        query_str = f"{session_id}:{query.text}:{query.tags}:{query.date_from}:{query.date_to}:{query.sort_by}:{query.sort_order}:{query.limit}:{query.offset}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _clear_search_cache(self, session_id: str):
        """Clear search cache for session"""
        keys_to_remove = [key for key in self._search_cache.keys() if key.startswith(session_id)]
        for key in keys_to_remove:
            del self._search_cache[key]
    
    async def export_entries(self, session_id: str, format: str = "json") -> Dict[str, Any]:
        """Export all journal entries for a session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM journal_entries 
                    WHERE session_id = ?
                    ORDER BY created_at
                """, (session_id,))
                
                entries = [self._row_to_entry(row).to_dict() for row in cursor.fetchall()]
            
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "total_entries": len(entries),
                "format_version": "1.0",
                "entries": entries
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting entries: {e}")
            raise
