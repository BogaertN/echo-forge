import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import aiohttp
import re
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categories of available tools"""
    WEB_SEARCH = "web_search"
    FACT_CHECK = "fact_check"
    KNOWLEDGE_BASE = "knowledge_base"
    TRANSLATION = "translation"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    COMPUTATION = "computation"

class ToolStatus(Enum):
    """Tool operational status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class ToolConfig:
    """Configuration for external tools"""
    name: str
    category: ToolCategory
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit: int = 100  # requests per hour
    timeout: int = 10  # seconds
    enabled: bool = True
    cost_per_request: float = 0.0
    privacy_level: str = "high"  # high, medium, low
    fallback_tools: List[str] = field(default_factory=list)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        return result

@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    response_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SearchResult:
    """Web search result structure"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactCheckResult:
    """Fact-checking result structure"""
    claim: str
    verdict: str  # true, false, mixed, unknown
    confidence: float
    sources: List[Dict[str, Any]]
    explanation: str
    fact_checker: str
    checked_at: datetime

class RateLimiter:
    """Rate limiting for tool usage"""
    
    def __init__(self, requests_per_hour: int):
        self.requests_per_hour = requests_per_hour
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def can_proceed(self) -> bool:
        """Check if request can proceed under rate limit"""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 hour
            self.requests = [req_time for req_time in self.requests if now - req_time < 3600]
            
            if len(self.requests) < self.requests_per_hour:
                self.requests.append(now)
                return True
            return False
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get when rate limit resets"""
        if not self.requests:
            return None
        
        oldest_request = min(self.requests)
        reset_time = oldest_request + 3600  # 1 hour later
        return datetime.fromtimestamp(reset_time)

class ToolCache:
    """Caching system for tool results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
    
    def get_cache_key(self, tool_name: str, query: str, params: Dict = None) -> str:
        """Generate cache key for query"""
        params_str = json.dumps(params or {}, sort_keys=True)
        key_data = f"{tool_name}:{query}:{params_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return None
        
        # Update access time
        self.access_times[cache_key] = time.time()
        return entry["data"]
    
    def set(self, cache_key: str, data: Any):
        """Set cached result"""
        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self.access_times[cache_key] = time.time()
    
    def _evict_oldest(self):
        """Evict least recently used entries"""
        if not self.access_times:
            return
        
        # Find oldest entry
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

class WebSearchTool:
    """Web search tool using DuckDuckGo and other privacy-focused engines"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.session = None
    
    async def search(self, query: str, max_results: int = 10, 
                    region: str = "us-en") -> List[SearchResult]:
        """
        Perform web search using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            region: Search region
            
        Returns:
            List of search results
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
            
            # DuckDuckGo instant answer API
            search_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Search API returned status {response.status}")
                
                data = await response.json()
                
                results = []
                
                # Process instant answer
                if data.get("Answer"):
                    results.append(SearchResult(
                        title="Instant Answer",
                        url=data.get("AbstractURL", ""),
                        snippet=data["Answer"],
                        source="DuckDuckGo",
                        relevance_score=1.0
                    ))
                
                # Process abstract
                if data.get("Abstract") and len(results) < max_results:
                    results.append(SearchResult(
                        title=data.get("Heading", "Abstract"),
                        url=data.get("AbstractURL", ""),
                        snippet=data["Abstract"],
                        source=data.get("AbstractSource", "DuckDuckGo"),
                        relevance_score=0.9
                    ))
                
                # Process related topics
                for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(SearchResult(
                            title=topic.get("Text", "")[:100],
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            source="DuckDuckGo",
                            relevance_score=0.7
                        ))
                
                # If we don't have enough results, try HTML scraping (fallback)
                if len(results) < max_results // 2:
                    html_results = await self._search_html(query, max_results - len(results))
                    results.extend(html_results)
                
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    async def _search_html(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback HTML search for DuckDuckGo"""
        try:
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with self.session.get(search_url, params=params, headers=headers) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
                
                # Simple HTML parsing for search results
                results = []
                
                # Extract search result snippets (simplified)
                import re
                
                # Find result blocks
                result_pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
                snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'
                
                urls_titles = re.findall(result_pattern, html_content)
                snippets = re.findall(snippet_pattern, html_content)
                
                for i, (url, title) in enumerate(urls_titles[:max_results]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    
                    results.append(SearchResult(
                        title=title.strip(),
                        url=url,
                        snippet=snippet.strip(),
                        source="DuckDuckGo",
                        relevance_score=0.6 - (i * 0.05)
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Error in HTML search fallback: {e}")
            return []
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class FactCheckTool:
    """Fact-checking tool using multiple verification sources"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.session = None
        
        # Known fact-checking patterns
        self.fact_check_patterns = {
            "snopes.com": r"snopes\.com/fact-check/([^/]+)",
            "factcheck.org": r"factcheck\.org/([^/]+)",
            "politifact.com": r"politifact\.com/factchecks/([^/]+)",
            "reuters.com": r"reuters\.com/fact-check/([^/]+)"
        }
    
    async def verify_claim(self, claim: str) -> FactCheckResult:
        """
        Verify a factual claim using multiple sources.
        
        Args:
            claim: Claim to verify
            
        Returns:
            Fact-checking result
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
            
            # Search for fact-checks related to the claim
            search_query = f'"{claim}" site:snopes.com OR site:factcheck.org OR site:politifact.com'
            
            # Use web search to find fact-check articles
            web_searcher = WebSearchTool(ToolConfig(
                name="web_search_factcheck",
                category=ToolCategory.WEB_SEARCH,
                timeout=self.config.timeout
            ))
            
            search_results = await web_searcher.search(search_query, max_results=5)
            
            fact_check_sources = []
            verdicts = []
            
            for result in search_results:
                if any(domain in result.url for domain in self.fact_check_patterns.keys()):
                    # Analyze the fact-check result
                    verdict = await self._analyze_fact_check_page(result.url, result.snippet)
                    if verdict:
                        fact_check_sources.append({
                            "url": result.url,
                            "title": result.title,
                            "snippet": result.snippet,
                            "verdict": verdict["verdict"],
                            "confidence": verdict["confidence"]
                        })
                        verdicts.append(verdict["verdict"])
            
            # Determine overall verdict
            overall_verdict = self._determine_overall_verdict(verdicts)
            confidence = self._calculate_confidence(verdicts, len(fact_check_sources))
            
            explanation = self._generate_explanation(claim, fact_check_sources, overall_verdict)
            
            await web_searcher.close()
            
            return FactCheckResult(
                claim=claim,
                verdict=overall_verdict,
                confidence=confidence,
                sources=fact_check_sources,
                explanation=explanation,
                fact_checker="multi_source",
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in fact checking: {e}")
            return FactCheckResult(
                claim=claim,
                verdict="unknown",
                confidence=0.0,
                sources=[],
                explanation=f"Error during fact-checking: {str(e)}",
                fact_checker="error",
                checked_at=datetime.now()
            )
    
    async def _analyze_fact_check_page(self, url: str, snippet: str) -> Optional[Dict[str, Any]]:
        """Analyze fact-check page content"""
        try:
            # Simple keyword-based analysis of snippet
            snippet_lower = snippet.lower()
            
            if any(word in snippet_lower for word in ["false", "incorrect", "debunked", "myth"]):
                return {"verdict": "false", "confidence": 0.8}
            elif any(word in snippet_lower for word in ["true", "correct", "confirmed", "verified"]):
                return {"verdict": "true", "confidence": 0.8}
            elif any(word in snippet_lower for word in ["mixed", "partly", "partially", "misleading"]):
                return {"verdict": "mixed", "confidence": 0.7}
            else:
                return {"verdict": "unknown", "confidence": 0.3}
                
        except Exception as e:
            logger.error(f"Error analyzing fact-check page: {e}")
            return None
    
    def _determine_overall_verdict(self, verdicts: List[str]) -> str:
        """Determine overall verdict from multiple sources"""
        if not verdicts:
            return "unknown"
        
        verdict_counts = {}
        for verdict in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Return most common verdict
        return max(verdict_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_confidence(self, verdicts: List[str], source_count: int) -> float:
        """Calculate confidence based on agreement and source count"""
        if not verdicts:
            return 0.0
        
        # Agreement factor (how much sources agree)
        agreement_factor = verdicts.count(self._determine_overall_verdict(verdicts)) / len(verdicts)
        
        # Source count factor
        source_factor = min(1.0, source_count / 3.0)  # Max confidence with 3+ sources
        
        return min(1.0, agreement_factor * source_factor)
    
    def _generate_explanation(self, claim: str, sources: List[Dict], verdict: str) -> str:
        """Generate explanation for fact-check result"""
        if not sources:
            return f"No reliable fact-checking sources found for the claim: '{claim}'"
        
        source_names = [source.get("title", "Unknown") for source in sources[:3]]
        source_list = ", ".join(source_names)
        
        verdict_text = {
            "true": "appears to be true",
            "false": "appears to be false",
            "mixed": "contains mixed or misleading information",
            "unknown": "could not be definitively verified"
        }.get(verdict, "has an unclear truth value")
        
        return f"Based on {len(sources)} fact-checking source(s) including {source_list}, this claim {verdict_text}."
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class ComputationTool:
    """Tool for mathematical and computational tasks"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    async def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate mathematical expressions safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Calculation result
        """
        try:
            # Import math modules for safe evaluation
            import math
            import statistics
            
            # Allowed functions and constants
            allowed_names = {
                # Math functions
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow,
                
                # Math module
                "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "asin": math.asin, "acos": math.acos, "atan": math.atan,
                "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
                "pi": math.pi, "e": math.e,
                
                # Statistics
                "mean": statistics.mean, "median": statistics.median,
                "mode": statistics.mode, "stdev": statistics.stdev,
                
                # Basic operations (handled by Python)
                "__builtins__": {}
            }
            
            # Clean expression (remove dangerous functions)
            cleaned_expr = self._clean_expression(expression)
            
            # Evaluate safely
            result = eval(cleaned_expr, {"__builtins__": {}}, allowed_names)
            
            return {
                "success": True,
                "result": result,
                "expression": cleaned_expr,
                "type": type(result).__name__
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and validate mathematical expression"""
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r"import\s+", r"exec\s*\(", r"eval\s*\(", r"open\s*\(",
            r"file\s*\(", r"input\s*\(", r"raw_input\s*\(",
            r"__.*__", r"globals\s*\(", r"locals\s*\(",
            r"getattr\s*\(", r"setattr\s*\(", r"hasattr\s*\(",
            r"delattr\s*\(", r"dir\s*\(", r"vars\s*\("
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous pattern detected: {pattern}")
        
        # Replace common mathematical notation
        expression = expression.replace("^", "**")  # Exponentiation
        expression = expression.replace("÷", "/")    # Division
        expression = expression.replace("×", "*")    # Multiplication
        
        return expression

class ToolManager:
    """
    Central manager for all external tool integrations.
    
    Handles tool routing, rate limiting, caching, and monitoring.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.tools = {}
        self.rate_limiters = {}
        self.cache = ToolCache()
        
        # Tool configurations
        self.tool_configs = self._load_tool_configs()
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_requests": 0,
            "cost_total": 0.0,
            "tools_used": {},
            "last_reset": datetime.now()
        }
        
        # Initialize tools
        self._initialize_tools()
        
        logger.info("ToolManager initialized")
    
    def _load_tool_configs(self) -> Dict[str, ToolConfig]:
        """Load tool configurations from environment and defaults"""
        configs = {}
        
        # Web search configuration
        configs["duckduckgo"] = ToolConfig(
            name="duckduckgo",
            category=ToolCategory.WEB_SEARCH,
            base_url="https://api.duckduckgo.com/",
            rate_limit=100,
            timeout=10,
            enabled=True,
            privacy_level="high"
        )
        
        # Fact-checking configuration
        configs["fact_check"] = ToolConfig(
            name="fact_check",
            category=ToolCategory.FACT_CHECK,
            rate_limit=50,
            timeout=15,
            enabled=True,
            privacy_level="high"
        )
        
        # Computation tool
        configs["computation"] = ToolConfig(
            name="computation",
            category=ToolCategory.COMPUTATION,
            rate_limit=1000,
            timeout=5,
            enabled=True,
            privacy_level="high"
        )
        
        # Override with environment variables
        for tool_name, config in configs.items():
            env_prefix = f"ECHOFORGE_{tool_name.upper()}"
            
            if os.getenv(f"{env_prefix}_API_KEY"):
                config.api_key = os.getenv(f"{env_prefix}_API_KEY")
            
            if os.getenv(f"{env_prefix}_ENABLED"):
                config.enabled = os.getenv(f"{env_prefix}_ENABLED").lower() == "true"
            
            if os.getenv(f"{env_prefix}_RATE_LIMIT"):
                try:
                    config.rate_limit = int(os.getenv(f"{env_prefix}_RATE_LIMIT"))
                except ValueError:
                    pass
        
        return configs
    
    def _initialize_tools(self):
        """Initialize tool instances and rate limiters"""
        for tool_name, config in self.tool_configs.items():
            if not config.enabled:
                continue
            
            # Initialize rate limiter
            self.rate_limiters[tool_name] = RateLimiter(config.rate_limit)
            
            # Initialize tool instance
            try:
                if config.category == ToolCategory.WEB_SEARCH:
                    self.tools[tool_name] = WebSearchTool(config)
                elif config.category == ToolCategory.FACT_CHECK:
                    self.tools[tool_name] = FactCheckTool(config)
                elif config.category == ToolCategory.COMPUTATION:
                    self.tools[tool_name] = ComputationTool(config)
                
                logger.info(f"Initialized tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_name}: {e}")
    
    def is_enabled(self) -> bool:
        """Check if any tools are enabled"""
        return len(self.tools) > 0
    
    async def web_search(self, query: str, max_results: int = 10,
                        agent_id: str = None, session_id: str = None) -> List[SearchResult]:
        """
        Perform web search with rate limiting and caching.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            agent_id: ID of requesting agent
            session_id: Session ID
            
        Returns:
            List of search results
        """
        try:
            tool_name = "duckduckgo"
            
            # Check if tool is available
            if tool_name not in self.tools:
                logger.warning(f"Web search tool {tool_name} not available")
                return []
            
            # Check rate limit
            if not await self.rate_limiters[tool_name].can_proceed():
                logger.warning(f"Rate limit exceeded for {tool_name}")
                return []
            
            # Check cache
            cache_key = self.cache.get_cache_key(tool_name, query, {"max_results": max_results})
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self._update_usage_stats(tool_name, True, True, 0.0)
                logger.info(f"Cache hit for web search: {query}")
                return cached_result
            
            # Perform search
            start_time = time.time()
            tool = self.tools[tool_name]
            results = await tool.search(query, max_results)
            response_time = time.time() - start_time
            
            # Cache results
            self.cache.set(cache_key, results)
            
            # Update stats
            success = len(results) > 0
            cost = self.tool_configs[tool_name].cost_per_request
            self._update_usage_stats(tool_name, success, False, cost, response_time)
            
            # Log usage to database
            if self.db and session_id:
                await self._log_tool_usage(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_action="web_search",
                    input_data={"query": query, "max_results": max_results},
                    output_data={"result_count": len(results)},
                    success=success,
                    response_time=response_time,
                    cost=cost
                )
            
            logger.info(f"Web search completed: {query} -> {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            self._update_usage_stats(tool_name, False, False, 0.0)
            return []
    
    async def fact_check(self, claim: str, agent_id: str = None,
                        session_id: str = None) -> FactCheckResult:
        """
        Perform fact-checking with rate limiting and caching.
        
        Args:
            claim: Claim to verify
            agent_id: ID of requesting agent
            session_id: Session ID
            
        Returns:
            Fact-checking result
        """
        try:
            tool_name = "fact_check"
            
            # Check if tool is available
            if tool_name not in self.tools:
                logger.warning(f"Fact-check tool {tool_name} not available")
                return FactCheckResult(
                    claim=claim,
                    verdict="unknown",
                    confidence=0.0,
                    sources=[],
                    explanation="Fact-checking tool not available",
                    fact_checker="unavailable",
                    checked_at=datetime.now()
                )
            
            # Check rate limit
            if not await self.rate_limiters[tool_name].can_proceed():
                logger.warning(f"Rate limit exceeded for {tool_name}")
                return FactCheckResult(
                    claim=claim,
                    verdict="unknown",
                    confidence=0.0,
                    sources=[],
                    explanation="Rate limit exceeded for fact-checking",
                    fact_checker="rate_limited",
                    checked_at=datetime.now()
                )
            
            # Check cache
            cache_key = self.cache.get_cache_key(tool_name, claim)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self._update_usage_stats(tool_name, True, True, 0.0)
                logger.info(f"Cache hit for fact check: {claim}")
                return cached_result
            
            # Perform fact check
            start_time = time.time()
            tool = self.tools[tool_name]
            result = await tool.verify_claim(claim)
            response_time = time.time() - start_time
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Update stats
            success = result.verdict != "unknown"
            cost = self.tool_configs[tool_name].cost_per_request
            self._update_usage_stats(tool_name, success, False, cost, response_time)
            
            # Log usage to database
            if self.db and session_id:
                await self._log_tool_usage(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_action="fact_check",
                    input_data={"claim": claim},
                    output_data={"verdict": result.verdict, "confidence": result.confidence},
                    success=success,
                    response_time=response_time,
                    cost=cost
                )
            
            logger.info(f"Fact check completed: {claim} -> {result.verdict}")
            return result
            
        except Exception as e:
            logger.error(f"Error in fact checking: {e}")
            self._update_usage_stats(tool_name, False, False, 0.0)
            return FactCheckResult(
                claim=claim,
                verdict="unknown",
                confidence=0.0,
                sources=[],
                explanation=f"Error during fact-checking: {str(e)}",
                fact_checker="error",
                checked_at=datetime.now()
            )
    
    async def calculate(self, expression: str, agent_id: str = None,
                       session_id: str = None) -> Dict[str, Any]:
        """
        Perform mathematical calculation.
        
        Args:
            expression: Mathematical expression
            agent_id: ID of requesting agent
            session_id: Session ID
            
        Returns:
            Calculation result
        """
        try:
            tool_name = "computation"
            
            if tool_name not in self.tools:
                return {"success": False, "error": "Computation tool not available"}
            
            # Check rate limit
            if not await self.rate_limiters[tool_name].can_proceed():
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Check cache
            cache_key = self.cache.get_cache_key(tool_name, expression)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self._update_usage_stats(tool_name, True, True, 0.0)
                return cached_result
            
            # Perform calculation
            start_time = time.time()
            tool = self.tools[tool_name]
            result = await tool.calculate(expression)
            response_time = time.time() - start_time
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Update stats
            success = result.get("success", False)
            self._update_usage_stats(tool_name, success, False, 0.0, response_time)
            
            # Log usage to database
            if self.db and session_id:
                await self._log_tool_usage(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_action="calculate",
                    input_data={"expression": expression},
                    output_data=result,
                    success=success,
                    response_time=response_time
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in calculation: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_usage_stats(self, tool_name: str, success: bool, cache_hit: bool,
                           cost: float, response_time: float = 0.0):
        """Update usage statistics"""
        self.usage_stats["total_requests"] += 1
        
        if success:
            self.usage_stats["successful_requests"] += 1
        else:
            self.usage_stats["failed_requests"] += 1
        
        if cache_hit:
            self.usage_stats["cached_requests"] += 1
        
        self.usage_stats["cost_total"] += cost
        
        if tool_name not in self.usage_stats["tools_used"]:
            self.usage_stats["tools_used"][tool_name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "cache_hits": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0,
                "response_times": []
            }
        
        tool_stats = self.usage_stats["tools_used"][tool_name]
        tool_stats["requests"] += 1
        
        if success:
            tool_stats["successes"] += 1
        else:
            tool_stats["failures"] += 1
        
        if cache_hit:
            tool_stats["cache_hits"] += 1
        
        tool_stats["total_cost"] += cost
        
        if response_time > 0:
            tool_stats["response_times"].append(response_time)
            if len(tool_stats["response_times"]) > 100:  # Keep last 100 measurements
                tool_stats["response_times"] = tool_stats["response_times"][-100:]
            
            tool_stats["avg_response_time"] = sum(tool_stats["response_times"]) / len(tool_stats["response_times"])
    
    async def _log_tool_usage(self, session_id: str, tool_name: str, tool_action: str,
                             input_data: Dict, output_data: Dict, success: bool,
                             response_time: float, cost: float = 0.0):
        """Log tool usage to database"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO tool_usage (
                        id, session_id, tool_name, tool_action, input_data,
                        output_data, success, response_time, cost
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session_id,
                    tool_name,
                    tool_action,
                    json.dumps(input_data),
                    json.dumps(output_data),
                    success,
                    response_time,
                    cost
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging tool usage: {e}")
    
    def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tools"""
        status = {}
        
        for tool_name, config in self.tool_configs.items():
            tool_status = {
                "name": tool_name,
                "category": config.category.value,
                "enabled": config.enabled,
                "available": tool_name in self.tools,
                "rate_limit": config.rate_limit,
                "privacy_level": config.privacy_level
            }
            
            # Add rate limit status
            if tool_name in self.rate_limiters:
                rate_limiter = self.rate_limiters[tool_name]
                reset_time = rate_limiter.get_reset_time()
                tool_status["rate_limit_reset"] = reset_time.isoformat() if reset_time else None
            
            # Add usage stats
            if tool_name in self.usage_stats["tools_used"]:
                tool_stats = self.usage_stats["tools_used"][tool_name]
                tool_status["usage"] = {
                    "requests": tool_stats["requests"],
                    "success_rate": tool_stats["successes"] / tool_stats["requests"] if tool_stats["requests"] > 0 else 0,
                    "cache_hit_rate": tool_stats["cache_hits"] / tool_stats["requests"] if tool_stats["requests"] > 0 else 0,
                    "avg_response_time": tool_stats["avg_response_time"],
                    "total_cost": tool_stats["total_cost"]
                }
            
            status[tool_name] = tool_status
        
        return status
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get overall usage statistics"""
        return {
            **self.usage_stats,
            "cache_hit_rate": self.usage_stats["cached_requests"] / self.usage_stats["total_requests"] if self.usage_stats["total_requests"] > 0 else 0,
            "success_rate": self.usage_stats["successful_requests"] / self.usage_stats["total_requests"] if self.usage_stats["total_requests"] > 0 else 0,
            "uptime_hours": (datetime.now() - self.usage_stats["last_reset"]).total_seconds() / 3600
        }
    
    async def cleanup(self):
        """Clean up tool resources"""
        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                try:
                    await tool.close()
                except Exception as e:
                    logger.error(f"Error closing tool: {e}")
        
        logger.info("ToolManager cleanup completed")

# Convenience functions for external use
async def create_tool_manager(db=None) -> ToolManager:
    """Create and initialize tool manager"""
    return ToolManager(db)

def get_available_tools() -> List[str]:
    """Get list of available tool categories"""
    return [category.value for category in ToolCategory]
