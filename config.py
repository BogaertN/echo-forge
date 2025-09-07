import json
import os
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/echo_forge_config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file or environment variables.
    Provides defaults for all EchoForge components: models, agents, DB, tools, etc.
    Supports override via env vars for security (e.g., DB passphrase).
    """
    default_config = {
        'models': {
            'clarifier': 'gemma:2b',
            'proponent': 'llama3:8b',
            'opponent': 'llama3:8b',
            'decider': 'phi3:mini',
            'auditor': 'qwen2:1.5b',
            'specialist': 'tinyllama:1.1b',
            'journaling_assistant': 'phi3:mini'
        },
        'agent_routing': {
            'default_model': 'llama3:8b',
            'model_swapping': True,  # Enable dynamic model upgrade/swap
            'upgrade_strategy': 'performance_based'  # 'performance_based', 'user_preference', 'auto'
        },
        'db': {
            'path': 'data/journal.db',
            'passphrase': os.environ.get('ECHOFORGE_DB_PASSPHRASE', 'default_secure_passphrase')  # Override via env
        },
        'tools': {
            'web_search': False,
            'fact_check': False,
            'api_plugins': [],  # List of enabled plugins
            'network_logging': True,
            'user_confirmation_required': True  # For network calls
        },
        'journal': {
            'auto_rephrase': True,
            'auto_suggest': True,
            'backup_frequency': 'weekly'  # 'daily', 'weekly', 'manual'
        },
        'resonance': {
            'link_threshold': 0.7,  # Similarity threshold for auto-linking
            'prune_interval': 'monthly'  # Auto-prune weak edges
        },
        'gamification': {
            'notify_reports': True,
            'motivation_tips': True
        },
        'deployment': {
            'platform': 'local',  # 'local', 'cloud'
            'hardware_constraints': {
                'min_ram_gb': 8,
                'min_gpu_vram_gb': 4
            },
            'auto_optimize': True  # Auto-optimize for hardware
        },
        'security': {
            'encryption_level': 'high',
            'audit_logs': True
        },
        'plugins': []  # Community plugins configs
    }
    
    # Load from file if exists
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
            logger.info(f"Configuration loaded from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid config file: {str(e)}")
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
    
    # Override with env vars
    for key, value in os.environ.items():
        if key.startswith('ECHOFORGE_'):
            config_key = key[len('ECHOFORGE_'):].lower()
            default_config[config_key] = value
    
    # Validate config
    _validate_config(default_config)
    
    return default_config

def _validate_config(config: Dict[str, Any]):
    """Validate configuration values"""
    required_keys = ['models', 'db', 'tools']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Model validation
    for model in config['models'].values():
        if not isinstance(model, str) or ':' not in model:
            raise ValueError(f"Invalid model format: {model}")
    
    # DB passphrase check
    if len(config['db']['passphrase']) < 12:
        logger.warning("DB passphrase is short; recommend stronger passphrase")
    
    logger.debug("Configuration validated")

def save_config(config: Dict[str, Any], config_path: str = "configs/echo_forge_config.json"):
    """Save current configuration to file"""
    Path(config_path).parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

def get_model_for_agent(agent_type: str) -> str:
    """Get model for specific agent from config"""
    config = load_config()
    return config['models'].get(agent_type, config['agent_routing']['default_model'])
