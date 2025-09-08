import asyncio
import functools
import hashlib
import html
import json
import logging
import os
import re
import secrets
import string
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
import threading
import psutil
import platform

# External libraries for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("Warning: Advanced NLP features not available. Install nltk and scikit-learn for full functionality.")

# === LOGGING CONFIGURATION ===

def setup_logging(log_level: str = "INFO", log_file: str = "data/echo_forge.log") -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance with consistent configuration"""
    return logging.getLogger(name)

# === TEXT PROCESSING UTILITIES ===

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Strip and return
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Normalize text for similarity comparisons.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_key_concepts(text: str, max_concepts: int = 20) -> List[str]:
    """
    Extract key concepts from text using multiple techniques.
    
    Args:
        text: Input text
        max_concepts: Maximum number of concepts to return
        
    Returns:
        List of key concepts
    """
    if not text or len(text.strip()) < 10:
        return []
    
    concepts = set()
    
    # Method 1: Simple keyword extraction
    simple_concepts = _extract_simple_keywords(text, max_concepts // 2)
    concepts.update(simple_concepts)
    
    # Method 2: Advanced NLP if available
    if NLP_AVAILABLE:
        nlp_concepts = _extract_nlp_concepts(text, max_concepts // 2)
        concepts.update(nlp_concepts)
    
    # Method 3: Noun phrase extraction
    noun_phrases = extract_noun_phrases(text)
    concepts.update(noun_phrases[:max_concepts // 4])
    
    # Filter and clean concepts
    filtered_concepts = []
    stop_words = get_stop_words()
    
    for concept in concepts:
        concept = concept.lower().strip()
        if (len(concept) >= 3 and 
            concept not in stop_words and 
            not concept.isdigit() and
            len(concept.split()) <= 3):  # Max 3 words
            filtered_concepts.append(concept)
    
    return filtered_concepts[:max_concepts]

def _extract_simple_keywords(text: str, max_keywords: int) -> List[str]:
    """Extract keywords using simple frequency analysis"""
    # Clean text
    cleaned = normalize_text(text)
    words = cleaned.split()
    
    # Filter words
    stop_words = get_stop_words()
    filtered_words = [
        word for word in words 
        if len(word) >= 3 and word not in stop_words
    ]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, count in word_freq.most_common(max_keywords)]

def _extract_nlp_concepts(text: str, max_concepts: int) -> List[str]:
    """Extract concepts using NLTK and TF-IDF"""
    try:
        # Tokenize and tag
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns and noun phrases
        concepts = []
        for word, pos in tagged:
            if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(word) >= 3:
                concepts.append(word)
        
        # Use TF-IDF for scoring
        if len(concepts) > 1:
            vectorizer = TfidfVectorizer(
                max_features=max_concepts,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top scoring terms
                top_indices = tfidf_scores.argsort()[-max_concepts:][::-1]
                top_concepts = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                
                return top_concepts
            except:
                pass
        
        return concepts[:max_concepts]
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"NLP concept extraction failed: {e}")
        return []

def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases from text.
    
    Args:
        text: Input text
        
    Returns:
        List of noun phrases
    """
    if not NLP_AVAILABLE:
        # Simple fallback: consecutive capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        phrases = re.findall(pattern, text)
        return [phrase.lower() for phrase in phrases if len(phrase.split()) >= 2]
    
    try:
        # Use NLTK for proper noun phrase extraction
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Simple noun phrase pattern: (DT)?(JJ)*(NN)+
        phrases = []
        current_phrase = []
        
        for word, pos in tagged:
            if pos in ['DT', 'JJ', 'NN', 'NNS', 'NNP', 'NNPS']:
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 2:
                    phrase = ' '.join(current_phrase).lower()
                    phrases.append(phrase)
                current_phrase = []
        
        # Don't forget the last phrase
        if len(current_phrase) >= 2:
            phrase = ' '.join(current_phrase).lower()
            phrases.append(phrase)
        
        return phrases
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Noun phrase extraction failed: {e}")
        return []

def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.
    
    Args:
        text: Input text
        
    Returns:
        List of hashtags (without #)
    """
    pattern = r'#(\w+)'
    hashtags = re.findall(pattern, text, re.IGNORECASE)
    return [tag.lower() for tag in hashtags]

def get_stop_words() -> Set[str]:
    """Get comprehensive stop words list"""
    default_stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'would', 'this', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their',
        'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
        'more', 'very', 'when', 'come', 'its', 'now', 'over', 'think',
        'also', 'back', 'after', 'use', 'year', 'work', 'first', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
        'most', 'us', 'can', 'could', 'should', 'just', 'being', 'get',
        'got', 'going', 'go', 'know', 'well', 'see', 'here', 'there',
        'where', 'who', 'why', 'yes', 'no', 'not', 'only', 'other',
        'than', 'such', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'among', 'through', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once'
    }
    
    if NLP_AVAILABLE:
        try:
            nltk_stops = set(stopwords.words('english'))
            return default_stop_words.union(nltk_stops)
        except:
            pass
    
    return default_stop_words

def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    if not text:
        return 0
    
    # Clean text and split by whitespace
    cleaned = clean_text(text)
    words = cleaned.split()
    
    # Filter out very short "words" (likely punctuation)
    meaningful_words = [word for word in words if len(word) >= 1]
    
    return len(meaningful_words)

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time in minutes.
    
    Args:
        text: Input text
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = count_words(text)
    if word_count == 0:
        return 0
    
    minutes = max(1, round(word_count / words_per_minute))
    return minutes

# === SIMILARITY AND ANALYSIS ===

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if norm1 == norm2:
        return 1.0
    
    # Method 1: Simple word overlap
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard_similarity = len(intersection) / len(union) if union else 0.0
    
    # Method 2: TF-IDF similarity if available
    if NLP_AVAILABLE and len(norm1) > 10 and len(norm2) > 10:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([norm1, norm2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Combine Jaccard and cosine similarity
            return (jaccard_similarity + cosine_sim) / 2.0
        except:
            pass
    
    return jaccard_similarity

def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """
    Analyze text complexity metrics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of complexity metrics
    """
    if not text:
        return {"complexity_score": 0.0, "metrics": {}}
    
    # Basic metrics
    word_count = count_words(text)
    char_count = len(text)
    sentence_count = len(sent_tokenize(text)) if NLP_AVAILABLE else len(re.split(r'[.!?]+', text))
    
    if sentence_count == 0:
        sentence_count = 1
    
    # Calculate metrics
    avg_words_per_sentence = word_count / sentence_count
    avg_chars_per_word = char_count / word_count if word_count > 0 else 0
    
    # Vocabulary diversity (unique words / total words)
    words = normalize_text(text).split()
    unique_words = len(set(words))
    vocab_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Simple complexity score (0-10)
    # Based on average sentence length, word length, and vocabulary diversity
    sentence_complexity = min(10, avg_words_per_sentence / 2)  # 20 words = max complexity
    word_complexity = min(10, avg_chars_per_word)  # 10 chars = max complexity
    vocab_complexity = vocab_diversity * 10
    
    complexity_score = (sentence_complexity + word_complexity + vocab_complexity) / 3
    
    return {
        "complexity_score": complexity_score,
        "metrics": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_chars_per_word": avg_chars_per_word,
            "vocabulary_diversity": vocab_diversity,
            "unique_words": unique_words
        }
    }

# === ID AND UUID GENERATION ===

def generate_uuid() -> str:
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def generate_session_id() -> str:
    """Generate a session ID"""
    return generate_uuid()

def generate_short_id(length: int = 8) -> str:
    """
    Generate a short alphanumeric ID.
    
    Args:
        length: Length of the ID
        
    Returns:
        Short ID string
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def validate_uuid(uuid_str: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_str: UUID string to validate
        
    Returns:
        True if valid UUID, False otherwise
    """
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False

# === CONTENT HASHING AND SECURITY ===

def hash_content(content: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of content.
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hex digest of hash
    """
    if not content:
        return ""
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(content.encode('utf-8'))
    return hash_obj.hexdigest()

def generate_salt(length: int = 16) -> bytes:
    """Generate cryptographic salt"""
    return secrets.token_bytes(length)

def secure_compare(a: str, b: str) -> bool:
    """Secure string comparison to prevent timing attacks"""
    import hmac
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    # Ensure not empty
    if not filename.strip():
        filename = "unnamed_file"
    
    return filename.strip()

# === PERFORMANCE MONITORING ===

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        logger = get_logger(__name__)
        if exc_type is None:
            logger.debug(f"Operation '{self.operation_name}' completed in {self.duration:.3f}s")
        else:
            logger.warning(f"Operation '{self.operation_name}' failed after {self.duration:.3f}s")

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.debug(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.warning(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.debug(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.warning(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def get_system_info() -> Dict[str, Any]:
    """Get system information and performance metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
            "disk_percent": (disk.used / disk.total) * 100,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e), "timestamp": datetime.now()}

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now()}

# === CACHING UTILITIES ===

class TTLCache:
    """Time-To-Live cache implementation"""
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        self.access_order = {}
        self.access_counter = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.default_ttl:
                self._delete_key(key)
                return None
            
            # Update access order
            self.access_counter += 1
            self.access_order[key] = self.access_counter
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cached value"""
        with self._lock:
            # Clean up if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_counter += 1
            self.access_order[key] = self.access_counter
    
    def delete(self, key: str) -> bool:
        """Delete cached value"""
        with self._lock:
            if key in self.cache:
                self._delete_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached values"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_order.clear()
            self.access_counter = 0
    
    def cleanup(self) -> int:
        """Clean up expired entries and return count removed"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.default_ttl
            ]
            
            for key in expired_keys:
                self._delete_key(key)
            
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "hit_ratio": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
            }
    
    def _delete_key(self, key: str) -> None:
        """Delete key from all internal structures"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_order.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry"""
        if not self.access_order:
            return
        
        oldest_key = min(self.access_order.items(), key=lambda x: x[1])[0]
        self._delete_key(oldest_key)

def cache_result(ttl: int = 3600, max_size: int = 100):
    """Decorator to cache function results"""
    cache = TTLCache(default_ttl=ttl, max_size=max_size)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hash_content(key_data)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        wrapper.cache_cleanup = cache.cleanup
        
        return wrapper
    
    return decorator

# === DATE AND TIME UTILITIES ===

def get_utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime as string"""
    return dt.strftime(format_str)

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse datetime from string"""
    return datetime.strptime(dt_str, format_str)

def time_ago(dt: datetime) -> str:
    """
    Get human-readable time difference.
    
    Args:
        dt: Datetime to compare with now
        
    Returns:
        Human-readable time difference
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now - dt
    
    if diff.days > 365:
        return f"{diff.days // 365} year{'s' if diff.days // 365 != 1 else ''} ago"
    elif diff.days > 30:
        return f"{diff.days // 30} month{'s' if diff.days // 30 != 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"

def is_business_hours(dt: datetime = None, 
                     start_hour: int = 9, 
                     end_hour: int = 17,
                     weekdays_only: bool = True) -> bool:
    """
    Check if datetime is within business hours.
    
    Args:
        dt: Datetime to check (default: now)
        start_hour: Business start hour (24-hour format)
        end_hour: Business end hour (24-hour format)
        weekdays_only: Only consider weekdays as business days
        
    Returns:
        True if within business hours
    """
    if dt is None:
        dt = datetime.now()
    
    # Check weekday
    if weekdays_only and dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check hour
    return start_hour <= dt.hour < end_hour

# === CONFIGURATION UTILITIES ===

def load_config_from_env(prefix: str = "ECHOFORGE_") -> Dict[str, str]:
    """
    Load configuration from environment variables.
    
    Args:
        prefix: Environment variable prefix
        
    Returns:
        Dictionary of configuration values
    """
    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    
    return config

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, "").lower()
    return value in ["true", "1", "yes", "on"] if value else default

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float value from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def validate_environment() -> bool:
    """
    Validate that required environment is properly set up.
    
    Returns:
        True if environment is valid
    """
    logger = get_logger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required directories
    required_dirs = ["data", "configs", "logs"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Created directory: {dir_name}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_name}: {e}")
                return False
    
    # Check disk space (at least 1GB free)
    try:
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1.0:
            logger.warning(f"Low disk space: {free_gb:.1f}GB free")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    # Check memory (at least 1GB available)
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 1.0:
            logger.warning(f"Low memory: {available_gb:.1f}GB available")
    except Exception as e:
        logger.warning(f"Could not check memory: {e}")
    
    logger.info("Environment validation completed successfully")
    return True

# === ERROR HANDLING UTILITIES ===

def handle_exception(exc: Exception, context: str = "") -> Dict[str, Any]:
    """
    Handle exception and return standardized error information.
    
    Args:
        exc: Exception instance
        context: Additional context information
        
    Returns:
        Dictionary with error details
    """
    import traceback
    
    error_info = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "traceback": traceback.format_exc()
    }
    
    logger = get_logger(__name__)
    logger.error(f"Exception in {context}: {exc}", exc_info=True)
    
    return error_info

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump JSON with fallback"""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default

# === FILE SYSTEM UTILITIES ===

def ensure_directory(path: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def safe_file_write(filepath: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', backup: bool = True) -> bool:
    """
    Safely write file with optional backup.
    
    Args:
        filepath: File path
        content: Content to write
        encoding: File encoding
        backup: Create backup if file exists
        
    Returns:
        True if write successful
    """
    try:
        filepath = Path(filepath)
        
        # Create backup if requested and file exists
        if backup and filepath.exists():
            backup_path = filepath.with_suffix(f"{filepath.suffix}.backup")
            filepath.rename(backup_path)
        
        # Ensure directory exists
        ensure_directory(filepath.parent)
        
        # Write file
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to write file {filepath}: {e}")
        return False

def safe_file_read(filepath: Union[str, Path], 
                  encoding: str = 'utf-8', 
                  default: str = "") -> str:
    """
    Safely read file with fallback.
    
    Args:
        filepath: File path
        encoding: File encoding
        default: Default value if read fails
        
    Returns:
        File content or default value
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to read file {filepath}: {e}")
        return default

# === VALIDATION UTILITIES ===

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))

def validate_json(json_str: str) -> bool:
    """Validate JSON format"""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def is_safe_string(text: str, max_length: int = 1000) -> bool:
    """
    Check if string is safe (no dangerous content).
    
    Args:
        text: String to check
        max_length: Maximum allowed length
        
    Returns:
        True if string is safe
    """
    if not isinstance(text, str):
        return False
    
    if len(text) > max_length:
        return False
    
    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript protocol
        r'vbscript:',      # VBScript protocol
        r'on\w+\s*=',      # Event handlers
        r'expression\s*\(', # CSS expressions
        r'import\s+',      # Import statements
        r'eval\s*\(',      # Eval calls
        r'exec\s*\(',      # Exec calls
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

# === MATHEMATICAL UTILITIES ===

def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def normalize_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """Normalize score to 0-1 range"""
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)

def calculate_percentage(part: Union[int, float], total: Union[int, float]) -> float:
    """Calculate percentage with division by zero protection"""
    if total == 0:
        return 0.0
    return (part / total) * 100.0

def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average"""
    if not values or window_size <= 0:
        return []
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_values = values[start_idx:i+1]
        result.append(sum(window_values) / len(window_values))
    
    return result

# === SINGLETON UTILITIES ===

def singleton(cls):
    """Decorator to make a class a singleton"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

# === INITIALIZATION ===

def initialize_nltk():
    """Initialize NLTK data if available"""
    if not NLP_AVAILABLE:
        return
    
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        required_data = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words']
        
        for data_name in required_data:
            try:
                nltk.download(data_name, quiet=True)
            except:
                pass
        
        logger = get_logger(__name__)
        logger.info("NLTK initialized successfully")
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"NLTK initialization failed: {e}")

# Initialize NLTK on import
if NLP_AVAILABLE:
    initialize_nltk()
