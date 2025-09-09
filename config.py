import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import secrets
from enum import Enum


class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    # SQLite settings
    db_path: str = "data/echoforge.db"
    encrypted_db_path: str = "data/echoforge_encrypted.db"
    enable_encryption: bool = True
    encryption_key_file: str = "data/.encryption_key"
    
    # Connection settings
    connection_timeout: int = 30
    max_connections: int = 10
    pool_size: int = 5
    
    # Performance settings
    enable_wal_mode: bool = True
    cache_size: int = 2000
    synchronous_mode: str = "NORMAL"
    temp_store: str = "MEMORY"
    
    # Backup settings
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_directory: str = "data/backups"
    
    # Maintenance settings
    auto_vacuum: bool = True
    analyze_frequency_hours: int = 168  # Weekly


@dataclass
class LLMConfig:
    """LLM (Language Model) configuration settings."""
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 1.0
    
    # Default models
    default_model: str = "llama3.1"
    fast_model: str = "llama3.1:8b"
    powerful_model: str = "llama3.1:70b"
    
    # Model-specific settings
    model_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "llama3.1": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,
            "context_length": 8192
        },
        "llama3.1:8b": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 1024,
            "context_length": 4096
        },
        "llama3.1:70b": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 4096,
            "context_length": 16384
        }
    })
    
    # Performance settings
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 5
    
    # Fallback settings
    enable_fallback_models: bool = True
    fallback_model_chain: List[str] = field(default_factory=lambda: ["llama3.1", "llama3.1:8b"])


@dataclass
class WebConfig:
    """Web server configuration settings."""
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # WebSocket settings
    websocket_timeout: int = 60
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    max_websocket_connections: int = 100
    
    # Security settings
    secret_key: str = ""
    session_timeout_minutes: int = 720  # 12 hours
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 15
    
    # Static files
    static_directory: str = "frontend/static"
    templates_directory: str = "frontend/templates"
    upload_directory: str = "data/uploads"
    max_upload_size: int = 5 * 1024 * 1024  # 5MB


@dataclass
class AgentConfig:
    """Agent system configuration settings."""
    # General agent settings
    default_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Agent-specific timeouts
    clarifier_timeout: int = 45
    proponent_timeout: int = 90
    opponent_timeout: int = 90
    synthesizer_timeout: int = 120
    auditor_timeout: int = 75
    journal_assistant_timeout: int = 60
    specialist_timeout: int = 75
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    max_conversation_history: int = 50
    context_compression_threshold: int = 4000
    
    # Agent behavior settings
    enable_tools: bool = True
    enable_web_search: bool = True
    enable_fact_checking: bool = True
    enable_specialists: bool = True
    max_specialists_per_debate: int = 2
    
    # Quality control
    min_response_length: int = 50
    max_response_length: int = 4000
    enable_content_filtering: bool = True
    enable_quality_checks: bool = True


@dataclass
class DebateConfig:
    """Debate system configuration settings."""
    # Debate flow settings
    max_clarification_rounds: int = 5
    max_debate_rounds: int = 10
    max_specialist_calls: int = 3
    
    # Time limits
    clarification_timeout_minutes: int = 15
    debate_timeout_minutes: int = 45
    synthesis_timeout_minutes: int = 10
    
    # Quality thresholds
    min_argument_quality_score: float = 3.0
    fallacy_warning_threshold: int = 2
    auto_auditor_trigger_threshold: float = 2.5
    
    # Auto-completion settings
    enable_auto_completion: bool = True
    auto_complete_on_convergence: bool = True
    convergence_threshold: float = 0.8
    auto_complete_on_repetition: bool = True
    repetition_threshold: int = 3
    
    # Ghost loop detection
    enable_ghost_loop_detection: bool = True
    similarity_threshold: float = 0.85
    max_similar_responses: int = 2


@dataclass
class JournalConfig:
    """Journal system configuration settings."""
    # Entry settings
    max_entry_length: int = 10000
    min_entry_length: int = 10
    auto_save_interval_seconds: int = 30
    
    # Search settings
    enable_full_text_search: bool = True
    search_results_limit: int = 50
    search_highlight_enabled: bool = True
    
    # Analytics settings
    enable_analytics: bool = True
    analytics_update_interval_hours: int = 24
    mood_tracking_enabled: bool = True
    
    # Gamification settings
    enable_gamification: bool = True
    points_per_entry: int = 10
    bonus_points_long_entry: int = 5
    streak_bonus_multiplier: float = 1.5
    
    # Export settings
    enable_export: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "markdown", "pdf"])
    max_export_entries: int = 1000


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Encryption settings
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    
    # Session security
    secure_cookies: bool = True
    cookie_samesite: str = "Lax"
    session_regenerate_interval: int = 3600  # 1 hour
    
    # Content security
    enable_content_sanitization: bool = True
    max_input_length: int = 10000
    blocked_file_extensions: List[str] = field(default_factory=lambda: [".exe", ".bat", ".cmd", ".com", ".scr"])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    global_rate_limit: int = 1000
    per_ip_rate_limit: int = 100
    rate_limit_window_seconds: int = 3600
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_file: str = "logs/audit.log"
    log_sensitive_operations: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Memory management
    max_memory_usage_mb: int = 1024
    gc_threshold_mb: int = 512
    enable_memory_monitoring: bool = True
    
    # Caching
    enable_redis_cache: bool = False
    redis_url: str = "redis://localhost:6379"
    cache_default_ttl: int = 3600
    max_cache_size_mb: int = 256
    
    # Background tasks
    enable_background_tasks: bool = True
    task_queue_size: int = 100
    max_concurrent_tasks: int = 10
    task_timeout_seconds: int = 300
    
    # Database optimization
    enable_query_optimization: bool = True
    query_timeout_seconds: int = 30
    connection_pool_size: int = 10
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60
    performance_log_file: str = "logs/performance.log"


@dataclass
class EchoForgeConfig:
    """Main EchoForge configuration container."""
    environment: Environment = Environment.DEVELOPMENT
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    web: WebConfig = field(default_factory=WebConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file: str = "logs/echoforge.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Application settings
    app_name: str = "EchoForge"
    app_version: str = "1.0.0"
    data_directory: str = "data"
    logs_directory: str = "logs"
    temp_directory: str = "temp"
    
    # Feature flags
    enable_voice_transcription: bool = True
    enable_resonance_mapping: bool = True
    enable_gamification: bool = True
    enable_export_features: bool = True
    enable_analytics: bool = True


class ConfigurationManager:
    """
    Manages loading, validation, and dynamic updates of configuration settings.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.env_file = env_file or ".env"
        self.config = EchoForgeConfig()
        self._config_validators = {}
        self._setup_validators()
        
    def load_configuration(self) -> EchoForgeConfig:
        """Load configuration from environment variables and config files."""
        try:
            # Load from .env file if it exists
            self._load_env_file()
            
            # Load from JSON config file if it exists
            self._load_config_file()
            
            # Override with environment variables
            self._load_environment_variables()
            
            # Validate configuration
            self._validate_configuration()
            
            # Ensure directories exist
            self._ensure_directories()
            
            # Generate security keys if needed
            self._ensure_security_keys()
            
            logging.info(f"Configuration loaded for {self.config.environment.value} environment")
            return self.config
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            # Return default configuration as fallback
            return EchoForgeConfig()
    
    def _load_env_file(self):
        """Load environment variables from .env file."""
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def _load_config_file(self):
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    self._apply_config_data(config_data)
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to config object."""
        for section, values in config_data.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Environment
        env_name = os.getenv('ECHOFORGE_ENV', 'development').lower()
        if env_name in [e.value for e in Environment]:
            self.config.environment = Environment(env_name)
        
        # Database settings
        self.config.database.db_path = os.getenv('DB_PATH', self.config.database.db_path)
        self.config.database.enable_encryption = self._get_bool_env('DB_ENABLE_ENCRYPTION', self.config.database.enable_encryption)
        
        # LLM settings
        self.config.llm.ollama_base_url = os.getenv('OLLAMA_BASE_URL', self.config.llm.ollama_base_url)
        self.config.llm.default_model = os.getenv('DEFAULT_MODEL', self.config.llm.default_model)
        
        # Web settings
        self.config.web.host = os.getenv('HOST', self.config.web.host)
        self.config.web.port = self._get_int_env('PORT', self.config.web.port)
        self.config.web.debug = self._get_bool_env('DEBUG', self.config.web.debug)
        self.config.web.secret_key = os.getenv('SECRET_KEY', self.config.web.secret_key)
        
        # Security settings
        self.config.security.encryption_algorithm = os.getenv('ENCRYPTION_ALGORITHM', self.config.security.encryption_algorithm)
        
        # Performance settings
        self.config.performance.max_memory_usage_mb = self._get_int_env('MAX_MEMORY_MB', self.config.performance.max_memory_usage_mb)
        
        # Feature flags
        self.config.enable_voice_transcription = self._get_bool_env('ENABLE_VOICE', self.config.enable_voice_transcription)
        self.config.enable_resonance_mapping = self._get_bool_env('ENABLE_RESONANCE', self.config.enable_resonance_mapping)
        
        # Logging
        log_level_str = os.getenv('LOG_LEVEL', self.config.log_level.value)
        if log_level_str in [l.value for l in LogLevel]:
            self.config.log_level = LogLevel(log_level_str)
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _setup_validators(self):
        """Setup configuration validators."""
        self._config_validators = {
            'web.port': lambda x: 1 <= x <= 65535,
            'database.connection_timeout': lambda x: x > 0,
            'llm.ollama_timeout': lambda x: x > 0,
            'agents.max_retries': lambda x: x >= 0,
            'debate.max_debate_rounds': lambda x: x > 0,
            'security.key_derivation_iterations': lambda x: x >= 1000,
        }
    
    def _validate_configuration(self):
        """Validate configuration values."""
        errors = []
        
        # Validate using registered validators
        for path, validator in self._config_validators.items():
            try:
                value = self._get_config_value_by_path(path)
                if not validator(value):
                    errors.append(f"Invalid value for {path}: {value}")
            except Exception as e:
                errors.append(f"Error validating {path}: {e}")
        
        # Additional validation
        if not self.config.web.secret_key and self.config.environment == Environment.PRODUCTION:
            errors.append("SECRET_KEY must be set in production environment")
        
        if self.config.llm.ollama_timeout <= 0:
            errors.append("LLM timeout must be positive")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    def _get_config_value_by_path(self, path: str) -> Any:
        """Get configuration value by dot-separated path."""
        parts = path.split('.')
        value = self.config
        for part in parts:
            value = getattr(value, part)
        return value
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.config.data_directory,
            self.config.logs_directory,
            self.config.temp_directory,
            self.config.web.upload_directory,
            self.config.database.backup_directory,
            os.path.dirname(self.config.database.db_path),
            os.path.dirname(self.config.database.encrypted_db_path),
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _ensure_security_keys(self):
        """Generate security keys if they don't exist."""
        # Generate secret key if not set
        if not self.config.web.secret_key:
            self.config.web.secret_key = secrets.token_urlsafe(32)
            logging.info("Generated new secret key")
        
        # Ensure encryption key file exists
        key_file = Path(self.config.database.encryption_key_file)
        if not key_file.exists():
            key_file.parent.mkdir(parents=True, exist_ok=True)
            encryption_key = secrets.token_urlsafe(32)
            with open(key_file, 'w') as f:
                f.write(encryption_key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            logging.info("Generated new encryption key")
    
    def save_configuration(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        try:
            save_path = file_path or self.config_file
            config_dict = self._config_to_dict()
            
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logging.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        
        return dataclass_to_dict(self.config)
    
    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            self._apply_config_data(updates)
            self._validate_configuration()
            logging.info("Configuration updated successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to update configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "environment": self.config.environment.value,
            "app_version": self.config.app_version,
            "database_encryption": self.config.database.enable_encryption,
            "default_model": self.config.llm.default_model,
            "web_host": self.config.web.host,
            "web_port": self.config.web.port,
            "debug_mode": self.config.web.debug,
            "features": {
                "voice_transcription": self.config.enable_voice_transcription,
                "resonance_mapping": self.config.enable_resonance_mapping,
                "gamification": self.config.enable_gamification,
                "analytics": self.config.enable_analytics,
            }
        }


# Global configuration instance
_config_manager = None
_config = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_config() -> EchoForgeConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = get_config_manager().load_configuration()
    return _config


def reload_config() -> EchoForgeConfig:
    """Reload configuration from files and environment."""
    global _config, _config_manager
    _config_manager = ConfigurationManager()
    _config = _config_manager.load_configuration()
    return _config


def update_config(updates: Dict[str, Any]) -> bool:
    """Update global configuration."""
    return get_config_manager().update_configuration(updates)


def save_config(file_path: Optional[str] = None) -> bool:
    """Save global configuration."""
    return get_config_manager().save_configuration(file_path)


# Development convenience functions
def is_development() -> bool:
    """Check if running in development mode."""
    return get_config().environment == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production mode."""
    return get_config().environment == Environment.PRODUCTION


def is_testing() -> bool:
    """Check if running in testing mode."""
    return get_config().environment == Environment.TESTING


# Feature flag convenience functions
def voice_transcription_enabled() -> bool:
    """Check if voice transcription is enabled."""
    return get_config().enable_voice_transcription


def resonance_mapping_enabled() -> bool:
    """Check if resonance mapping is enabled."""
    return get_config().enable_resonance_mapping


def gamification_enabled() -> bool:
    """Check if gamification is enabled."""
    return get_config().enable_gamification


def analytics_enabled() -> bool:
    """Check if analytics are enabled."""
    return get_config().enable_analytics
