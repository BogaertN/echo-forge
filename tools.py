import logging
import asyncio
import json
import requests
from typing import Dict, Any, Optional, Callable, List
from config import load_config
from utils import log_tool_use, validate_tool_input
from connection_manager import ConnectionManager  # For notifications

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages optional tool use protocols: enablement per-agent/session, web search,
    fact-check, API plugins, logging/network calls, security/user control,
    extensions/plug-in hooks. Offline-first, with explicit user opt-in.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled_tools = self.config.get('tools', {})
        self.session_settings: Dict[str, Dict] = {}  # Per-session enablement
        self.agent_settings: Dict[str, Dict] = {}   # Per-agent enablement
        self.plugins: Dict[str, Callable] = {}      # Extension hooks
        self.connection_manager = ConnectionManager()  # For user notifications
        
        # Built-in tools
        self._register_builtin_tools()
        
        logger.info("ToolManager initialized with {} tools".format(len(self.enabled_tools)))
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        self.plugins['web_search'] = self._web_search
        self.plugins['fact_check'] = self._fact_check
        # Add more built-ins as needed, e.g., 'scraper', 'api_call'
    
    # Enablement and Control
    
    def enable_tool(self, tool_name: str, session_id: Optional[str] = None, agent_id: Optional[str] = None):
        """Enable tool for session or agent"""
        if tool_name not in self.enabled_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        if session_id:
            if session_id not in self.session_settings:
                self.session_settings[session_id] = {}
            self.session_settings[session_id][tool_name] = True
        elif agent_id:
            if agent_id not in self.agent_settings:
                self.agent_settings[agent_id] = {}
            self.agent_settings[agent_id][tool_name] = True
        else:
            # Global enable (admin only, assume user control)
            self.enabled_tools[tool_name] = True
        
        self._notify_user(f"Tool enabled: {tool_name}", session_id)
        logger.info(f"Tool {tool_name} enabled for {'session ' + session_id if session_id else 'agent ' + agent_id if agent_id else 'global'}")
    
    def disable_tool(self, tool_name: str, session_id: Optional[str] = None, agent_id: Optional[str] = None):
        """Disable tool"""
        if session_id and session_id in self.session_settings:
            self.session_settings[session_id].pop(tool_name, None)
        elif agent_id and agent_id in self.agent_settings:
            self.agent_settings[agent_id].pop(tool_name, None)
        else:
            self.enabled_tools.pop(tool_name, None)
        
        self._notify_user(f"Tool disabled: {tool_name}", session_id)
        logger.info(f"Tool {tool_name} disabled")
    
    def is_tool_enabled(self, tool_name: str, session_id: Optional[str] = None, agent_id: Optional[str] = None) -> bool:
        """Check if tool is enabled for context"""
        if session_id and session_id in self.session_settings and tool_name in self.session_settings[session_id]:
            return self.session_settings[session_id][tool_name]
        if agent_id and agent_id in self.agent_settings and tool_name in self.agent_settings[agent_id]:
            return self.agent_settings[agent_id][tool_name]
        return tool_name in self.enabled_tools and self.enabled_tools[tool_name]
    
    # Tool Execution
    
    async def execute_tool(self, tool_name: str, session_id: str, agent_id: str, **kwargs) -> Any:
        """Execute tool if enabled, with logging and security"""
        if not self.is_tool_enabled(tool_name, session_id, agent_id):
            raise PermissionError(f"Tool {tool_name} not enabled for this context")
        
        validate_tool_input(tool_name, kwargs)  # Security validation
        
        if tool_name in self.plugins:
            result = await self.plugins[tool_name](**kwargs)
            self._log_tool_use(tool_name, session_id, agent_id, kwargs, result)
            self._notify_user(f"Tool used: {tool_name}", session_id)
            return result
        raise ValueError(f"Tool not found: {tool_name}")
    
    def _log_tool_use(self, tool_name: str, session_id: str, agent_id: str, inputs: Dict, output: Any):
        """Log tool usage for audit"""
        log_entry = {
            'tool': tool_name,
            'session_id': session_id,
            'agent_id': agent_id,
            'inputs': inputs,
            'output': str(output)[:500],  # Truncate for log
            'timestamp': datetime.now().isoformat()
        }
        log_tool_use(log_entry)  # Utility function
        self.db.log_tool_use(log_entry)  # Assume DB method for audit trails
    
    # Built-in Tools
    
    async def _web_search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform web search (placeholder for real API, e.g., Serper or Bing)"""
        # Security: Rate limit, user confirmation if needed
        if not self.config['tools']['web_search_api_key']:
            raise ValueError("Web search API key not configured")
        
        try:
            response = requests.get(
                "https://api.search.example.com/search",  # Replace with real API
                params={'q': query, 'num': num_results},
                headers={'API-Key': self.config['tools']['web_search_api_key']}
            )
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    async def _fact_check(self, claim: str) -> Dict:
        """Fact-check a claim using web search or API"""
        search_results = await self._web_search(f"fact check: {claim}", 5)
        # Analyze results with agent or simple logic
        verification = {
            'claim': claim,
            'sources': [r['url'] for r in search_results],
            'status': 'verified' if any('true' in r['snippet'].lower() for r in search_results) else 'unverified'
        }
        return verification
    
    # Extensions and Plugins
    
    def register_plugin(self, name: str, func: Callable):
        """Register a new plugin/tool"""
        if name in self.plugins:
            raise ValueError(f"Plugin already exists: {name}")
        self.plugins[name] = func
        self.enabled_tools[name] = False  # Disabled by default
        logger.info(f"Plugin registered: {name}")
    
    def load_plugins(self):
        """Load community plugins from config or directory"""
        for plugin_config in self.config.get('plugins', []):
            # Dynamic import example
            module = __import__(plugin_config['module'])
            func = getattr(module, plugin_config['function'])
            self.register_plugin(plugin_config['name'], func)
    
    # Security and User Control
    
    def _notify_user(self, message: str, session_id: Optional[str]):
        """Notify user of tool events via WebSocket"""
        if session_id:
            asyncio.run(self.connection_manager.send_to_session(session_id, {
                'type': 'tool_notification',
                'message': message
            }))
    
    def get_tool_logs(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get audit logs for tool use"""
        return self.db.get_tool_logs(session_id)  # Assume DB method
    
    def clear_tool_logs(self):
        """Clear tool usage logs (user-initiated)"""
        self.db.clear_tool_logs()
        logger.warning("Tool logs cleared")
