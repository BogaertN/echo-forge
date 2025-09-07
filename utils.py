import logging
import json
import os
import subprocess
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from agents import JournalingAssistant  # For advanced utils needing agent
from config import load_config
from encrypted_db import EncryptedDB  # For backup hooks

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup application-wide logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=os.path.join('logs', 'echoforge.log'),
        filemode='a'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
    logger.info("Logging setup complete")

def log_error(error: Exception, context: Dict = None):
    """Log errors with context"""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg += f" | Context: {json.dumps(context)}"
    logger.error(error_msg)
    
    # Optional: Save to DB audit log
    if load_config()['security']['audit_logs']:
        db = EncryptedDB()  # Assume singleton or injected
        db.log_error(error_msg)  # Assume DB method

def generate_timestamp(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Generate formatted timestamp"""
    return datetime.now().strftime(format)

def validate_metadata(metadata: Dict) -> bool:
    """Validate journal metadata structure"""
    required_keys = ['title', 'tags', 'weights', 'ghost_loop', 'summary']
    if not all(key in metadata for key in required_keys):
        raise ValueError("Invalid metadata: missing required keys")
    
    if not isinstance(metadata['tags'], list):
        raise ValueError("Tags must be a list")
    
    weights = metadata['weights']
    if not all(key in weights for key in ['relevance', 'emotion', 'priority']):
        raise ValueError("Weights missing required fields")
    if not all(1 <= weights[key] <= 10 for key in weights):
        raise ValueError("Weights must be between 1 and 10")
    
    return True

def calculate_edge_strength(summary1: str, summary2: str) -> float:
    """Calculate similarity strength for resonance edges (placeholder for NLP)"""
    # TODO: Use embedding similarity (e.g., via sentence-transformers)
    # For now, simple word overlap
    words1 = set(summary1.lower().split())
    words2 = set(summary2.lower().split())
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    return overlap / total if total > 0 else 0.0

def send_notification(message: str, session_id: Optional[str] = None):
    """Send user notification (via WebSocket or console)"""
    from connection_manager import ConnectionManager
    cm = ConnectionManager()
    if session_id:
        asyncio.run(cm.send_system_notification(message, "info", [session_id]))
    else:
        logger.info(f"Notification: {message}")

def download_model(model_name: str) -> bool:
    """Download/install model via Ollama"""
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        logger.info(f"Model downloaded: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        log_error(e)
        return False

def swap_model(agent_type: str, new_model: str):
    """Swap model for agent"""
    config = load_config()
    config['models'][agent_type] = new_model
    save_config(config)  # Assume from config.py
    logger.info(f"Model swapped for {agent_type}: {new_model}")

def backup_system(backup_path: str = None) -> str:
    """Backup DB and configs"""
    db = EncryptedDB()
    db_backup = db.backup_database(backup_path)
    
    config_path = "configs/"
    shutil.copytree(config_path, os.path.join(backup_path or 'backup', 'configs'))
    
    logger.info("System backup complete")
    return db_backup

def restore_system(backup_path: str):
    """Restore from backup"""
    db = EncryptedDB()
    db.restore_database(os.path.join(backup_path, 'journal.db'))
    
    shutil.copytree(os.path.join(backup_path, 'configs'), 'configs/', dirs_exist_ok=True)
    
    logger.info("System restore complete")

def check_hardware_constraints() -> Dict:
    """Check hardware against constraints"""
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    # GPU check placeholder
    gpu_vram = 0  # TODO: Use GPUtil or similar
    
    config = load_config()
    constraints = config['deployment']['hardware_constraints']
    
    return {
        'ram_sufficient': ram_gb >= constraints['min_ram_gb'],
        'gpu_sufficient': gpu_vram >= constraints['min_gpu_vram_gb'],
        'details': {'ram_gb': ram_gb, 'gpu_vram_gb': gpu_vram}
    }

def optimize_for_hardware():
    """Auto-optimize config for hardware"""
    hardware = check_hardware_constraints()
    if not hardware['ram_sufficient']:
        config = load_config()
        # Downgrade models
        for key in config['models']:
            config['models'][key] = 'tinyllama:1.1b'  # Light model
        save_config(config)
        logger.warning("Config optimized for low RAM")

# Add more utils as referenced, e.g., validate_tool_input, log_tool_use
def validate_tool_input(tool_name: str, inputs: Dict) -> bool:
    """Validate tool inputs (placeholder)"""
    # TODO: Per-tool validation
    return True

def log_tool_use(log_entry: Dict):
    """Log tool use to file"""
    log_path = os.path.join('logs', 'tool_usage.json')
    with open(log_path, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')
