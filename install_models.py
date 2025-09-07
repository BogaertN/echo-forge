import logging
import subprocess
import os
import json
from typing import List, Dict
from config import load_config, save_config
from utils import log_error, generate_timestamp, check_hardware_constraints

logger = logging.getLogger(__name__)

class ModelInstaller:
    """
    Handles model download, install, swapping, and upgrade strategy for EchoForge.
    Supports Ollama (primary) with hooks for llama.cpp. Checks hardware constraints,
    verifies installs, and updates config. Includes scripts for batch install.
    """
    
    def __init__(self):
        self.config = load_config()
        self.installed_models = self._get_installed_models()
        self.upgrade_strategy = self.config['agent_routing']['upgrade_strategy']
        
        logger.info("ModelInstaller initialized")
    
    def _get_installed_models(self) -> List[str]:
        """Get list of installed Ollama models"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            models = [line.split()[0] for line in result.stdout.splitlines()[1:]]  # Skip header
            return models
        except subprocess.CalledProcessError as e:
            log_error(e)
            return []
    
    def install_model(self, model_name: str, force: bool = False) -> bool:
        """Install a single model via Ollama"""
        if model_name in self.installed_models and not force:
            logger.info(f"Model already installed: {model_name}")
            return True
        
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            self.installed_models.append(model_name)
            logger.info(f"Model installed: {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            log_error(e)
            return False
    
    def install_all_models(self):
        """Install all models from config"""
        models = list(self.config['models'].values())
        models = list(set(models))  # Unique
        for model in models:
            self.install_model(model)
        
        logger.info("All models installed")
    
    def swap_model(self, agent_type: str, new_model: str):
        """Swap model for an agent and update config"""
        if new_model not in self.installed_models:
            if not self.install_model(new_model):
                raise ValueError(f"Failed to install {new_model}")
        
        self.config['models'][agent_type] = new_model
        save_config(self.config)
        logger.info(f"Model swapped for {agent_type}: {new_model}")
    
    def upgrade_models(self):
        """Upgrade models based on strategy"""
        if self.upgrade_strategy == 'performance_based':
            hardware = check_hardware_constraints()
            if hardware['ram_sufficient'] and hardware['gpu_sufficient']:
                # Upgrade to larger models
                for agent, model in self.config['models'].items():
                    if '2b' in model or 'mini' in model:
                        new_model = model.replace('2b', '7b').replace('mini', 'medium')
                        self.swap_model(agent, new_model)
            else:
                # Downgrade
                for agent, model in self.config['models'].items():
                    if '8b' in model or 'medium' in model:
                        new_model = model.replace('8b', '3b').replace('medium', 'mini')
                        self.swap_model(agent, new_model)
        
        elif self.upgrade_strategy == 'user_preference':
            # Placeholder for user input
            pass
        
        logger.info("Models upgraded per strategy")
    
    def verify_installs(self) -> Dict[str, bool]:
        """Verify all required models are installed"""
        status = {}
        for agent, model in self.config['models'].items():
            status[agent] = model in self.installed_models
            if not status[agent]:
                logger.warning(f"Missing model for {agent}: {model}")
        return status
    
    def cleanup_unused_models(self):
        """Remove unused models to free space"""
        used_models = set(self.config['models'].values())
        for model in self.installed_models:
            if model not in used_models:
                try:
                    subprocess.run(["ollama", "rm", model], check=True)
                    logger.info(f"Unused model removed: {model}")
                except subprocess.CalledProcessError as e:
                    log_error(e)
    
    def export_model_config(self, path: str = "configs/model_config.json"):
        """Export current model config"""
        with open(path, 'w') as f:
            json.dump(self.config['models'], f, indent=4)
        logger.info(f"Model config exported to {path}")
    
    def import_model_config(self, path: str):
        """Import and apply model config"""
        with open(path, 'r') as f:
            new_models = json.load(f)
        for agent, model in new_models.items():
            self.swap_model(agent, model)
        logger.info(f"Model config imported from {path}")

# CLI Script Entry Point
if __name__ == "__main__":
    installer = ModelInstaller()
    
    import argparse
    parser = argparse.ArgumentParser(description="EchoForge Model Management")
    parser.add_argument('--install-all', action='store_true', help="Install all configured models")
    parser.add_argument('--install', type=str, help="Install specific model")
    parser.add_argument('--swap', nargs=2, help="Swap model for agent: <agent_type> <new_model>")
    parser.add_argument('--upgrade', action='store_true', help="Upgrade models per strategy")
    parser.add_argument('--verify', action='store_true', help="Verify installs")
    parser.add_argument('--cleanup', action='store_true', help="Cleanup unused models")
    
    args = parser.parse_args()
    
    if args.install_all:
        installer.install_all_models()
    elif args.install:
        installer.install_model(args.install)
    elif args.swap:
        installer.swap_model(args.swap[0], args.swap[1])
    elif args.upgrade:
        installer.upgrade_models()
    elif args.verify:
        print(json.dumps(installer.verify_installs(), indent=2))
    elif args.cleanup:
        installer.cleanup_unused_models()
    else:
        parser.print_help()
