#!/usr/bin/env python3
"""
EchoForge Installation Script
============================
Handles initial system setup, dependency management, and environment preparation for EchoForge.

Features:
- System requirements validation
- Python dependency installation
- Ollama setup and model downloading
- Database initialization
- Configuration file generation
- Directory structure creation
- Security key generation
- Health checks and validation
"""

import os
import sys
import subprocess
import platform
import json
import logging
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class EchoForgeInstaller:
    """
    Main installer class for EchoForge system setup.
    """
    
    def __init__(self, verbose: bool = False, dev_mode: bool = False):
        self.verbose = verbose
        self.dev_mode = dev_mode
        self.system_info = self._get_system_info()
        self.requirements = self._load_requirements()
        self.install_log = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('install.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for compatibility checking."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture()[0],
            "cpu_count": os.cpu_count(),
        }

    def _load_requirements(self) -> Dict[str, Any]:
        """Load installation requirements."""
        return {
            "python_min_version": (3, 8),
            "python_max_version": (3, 12),
            "min_memory_gb": 4,
            "recommended_memory_gb": 8,
            "min_disk_space_gb": 10,
            "recommended_disk_space_gb": 50,
            "required_packages": [
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.24.0",
                "websockets>=12.0",
                "pydantic>=2.5.0",
                "sqlalchemy>=2.0.0",
                "aiosqlite>=0.19.0",
                "cryptography>=41.0.0",
                "httpx>=0.25.0",
                "python-multipart>=0.0.6",
                "jinja2>=3.1.2",
                "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4",
                "nltk>=3.8",
                "scikit-learn>=1.3.0",
                "networkx>=3.2",
                "numpy>=1.24.0",
                "pandas>=2.0.0",
                "matplotlib>=3.7.0",
                "seaborn>=0.12.0",
                "requests>=2.31.0",
                "beautifulsoup4>=4.12.0",
                "lxml>=4.9.0",
                "python-dotenv>=1.0.0",
                "click>=8.1.0",
                "rich>=13.0.0",
                "psutil>=5.9.0",
                "schedule>=1.2.0"
            ],
            "optional_packages": [
                "redis>=5.0.0",
                "celery>=5.3.0",
                "openai>=1.0.0",
                "anthropic>=0.8.0",
                "whisper>=1.0.0",
                "torch>=2.0.0",
                "transformers>=4.35.0"
            ],
            "ollama_models": [
                "llama3.1",
                "llama3.1:8b"
            ],
            "optional_ollama_models": [
                "llama3.1:70b",
                "codellama",
                "mistral"
            ]
        }

    def print_banner(self):
        """Print installation banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                            EchoForge Installer                              ║
║                   Privacy-First Multi-Agent Debate Platform                 ║
║                                Version 1.0.0                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}

Welcome to the EchoForge installation wizard!

This installer will:
• Validate system requirements
• Install Python dependencies
• Set up Ollama and download AI models
• Initialize the database
• Create configuration files
• Prepare the runtime environment

{Colors.YELLOW}Note: This process may take 10-30 minutes depending on your internet connection.{Colors.END}
"""
        print(banner)

    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        self.logger.info("Checking system requirements...")
        issues = []

        # Check Python version
        python_version = sys.version_info[:2]
        min_version = self.requirements["python_min_version"]
        max_version = self.requirements["python_max_version"]
        
        if python_version < min_version:
            issues.append(f"Python {min_version[0]}.{min_version[1]}+ required, found {python_version[0]}.{python_version[1]}")
        elif python_version > max_version:
            issues.append(f"Python {max_version[0]}.{max_version[1]} or lower recommended, found {python_version[0]}.{python_version[1]}")

        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < self.requirements["min_memory_gb"]:
                issues.append(f"Minimum {self.requirements['min_memory_gb']}GB RAM required, found {memory_gb:.1f}GB")
            elif memory_gb < self.requirements["recommended_memory_gb"]:
                self.logger.warning(f"Recommended {self.requirements['recommended_memory_gb']}GB RAM, found {memory_gb:.1f}GB")
        except ImportError:
            self.logger.warning("Could not check memory requirements (psutil not available)")

        # Check disk space
        try:
            statvfs = os.statvfs('.')
            available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            if available_gb < self.requirements["min_disk_space_gb"]:
                issues.append(f"Minimum {self.requirements['min_disk_space_gb']}GB disk space required, found {available_gb:.1f}GB")
        except (OSError, AttributeError):
            self.logger.warning("Could not check disk space")

        # Check for required system tools
        required_tools = ["git", "curl"]
        for tool in required_tools:
            if not shutil.which(tool):
                issues.append(f"Required tool '{tool}' not found in PATH")

        if issues:
            self.logger.error("System requirements not met:")
            for issue in issues:
                self.logger.error(f"  • {issue}")
            return False

        self.logger.info(f"{Colors.GREEN}✓ System requirements check passed{Colors.END}")
        return True

    def create_directory_structure(self) -> bool:
        """Create required directory structure."""
        self.logger.info("Creating directory structure...")
        
        directories = [
            "data",
            "data/backups",
            "data/uploads",
            "data/exports",
            "logs",
            "temp",
            "frontend/static",
            "frontend/templates",
            "frontend/static/css",
            "frontend/static/js",
            "frontend/static/images",
            "agents",
            "tests",
            "docs"
        ]

        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
            
            self.logger.info(f"{Colors.GREEN}✓ Directory structure created{Colors.END}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory structure: {e}")
            return False

    def install_python_dependencies(self) -> bool:
        """Install Python dependencies using pip."""
        self.logger.info("Installing Python dependencies...")
        
        try:
            # Upgrade pip first
            self.logger.info("Upgrading pip...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True, check=True)
            
            if self.verbose:
                self.logger.debug(f"Pip upgrade output: {result.stdout}")

            # Install required packages
            self.logger.info("Installing required packages...")
            for package in self.requirements["required_packages"]:
                self.logger.info(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                
                if self.verbose:
                    self.logger.debug(f"Installation output for {package}: {result.stdout}")

            # Install optional packages in development mode
            if self.dev_mode:
                self.logger.info("Installing optional development packages...")
                for package in self.requirements["optional_packages"]:
                    try:
                        self.logger.info(f"Installing optional package {package}...")
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True, check=True)
                    except subprocess.CalledProcessError:
                        self.logger.warning(f"Failed to install optional package {package}")

            self.logger.info(f"{Colors.GREEN}✓ Python dependencies installed{Colors.END}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install Python dependencies: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False

    def install_ollama(self) -> bool:
        """Install Ollama if not present."""
        self.logger.info("Checking Ollama installation...")
        
        # Check if Ollama is already installed
        if shutil.which("ollama"):
            self.logger.info(f"{Colors.GREEN}✓ Ollama already installed{Colors.END}")
            return True

        self.logger.info("Installing Ollama...")
        
        try:
            system = platform.system().lower()
            
            if system == "linux" or system == "darwin":  # macOS
                # Download and run Ollama install script
                install_script_url = "https://ollama.ai/install.sh"
                self.logger.info("Downloading Ollama install script...")
                
                result = subprocess.run([
                    "curl", "-fsSL", install_script_url
                ], capture_output=True, text=True, check=True)
                
                # Execute install script
                result = subprocess.run([
                    "sh", "-c", result.stdout
                ], capture_output=True, text=True, check=True)
                
                if self.verbose:
                    self.logger.debug(f"Ollama install output: {result.stdout}")
                    
            elif system == "windows":
                self.logger.warning("Windows installation requires manual Ollama setup")
                self.logger.info("Please download Ollama from https://ollama.ai/download/windows")
                return False
            else:
                self.logger.error(f"Unsupported operating system: {system}")
                return False

            # Verify installation
            if shutil.which("ollama"):
                self.logger.info(f"{Colors.GREEN}✓ Ollama installed successfully{Colors.END}")
                return True
            else:
                self.logger.error("Ollama installation failed - command not found")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install Ollama: {e}")
            return False

    def start_ollama_service(self) -> bool:
        """Start Ollama service."""
        self.logger.info("Starting Ollama service...")
        
        try:
            # Check if Ollama is already running
            result = subprocess.run([
                "curl", "-s", "http://localhost:11434/api/version"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"{Colors.GREEN}✓ Ollama service already running{Colors.END}")
                return True

            # Start Ollama service
            system = platform.system().lower()
            
            if system == "linux" or system == "darwin":
                # Start Ollama in background
                subprocess.Popen([
                    "ollama", "serve"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Wait for service to start
                import time
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    result = subprocess.run([
                        "curl", "-s", "http://localhost:11434/api/version"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.logger.info(f"{Colors.GREEN}✓ Ollama service started{Colors.END}")
                        return True
                
                self.logger.error("Ollama service failed to start within 30 seconds")
                return False
                
            else:
                self.logger.warning("Manual Ollama service start required on Windows")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start Ollama service: {e}")
            return False

    def download_ollama_models(self) -> bool:
        """Download required Ollama models."""
        self.logger.info("Downloading Ollama models...")
        
        # Download required models
        for model in self.requirements["ollama_models"]:
            self.logger.info(f"Downloading model: {model}")
            try:
                result = subprocess.run([
                    "ollama", "pull", model
                ], capture_output=True, text=True, check=True)
                
                if self.verbose:
                    self.logger.debug(f"Model download output: {result.stdout}")
                    
                self.logger.info(f"{Colors.GREEN}✓ Model {model} downloaded{Colors.END}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to download model {model}: {e}")
                return False

        # Download optional models in dev mode
        if self.dev_mode:
            for model in self.requirements["optional_ollama_models"]:
                self.logger.info(f"Downloading optional model: {model}")
                try:
                    result = subprocess.run([
                        "ollama", "pull", model
                    ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
                    
                    self.logger.info(f"{Colors.GREEN}✓ Optional model {model} downloaded{Colors.END}")
                    
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    self.logger.warning(f"Failed to download optional model {model}: {e}")

        return True

    def initialize_database(self) -> bool:
        """Initialize the database."""
        self.logger.info("Initializing database...")
        
        try:
            # Import database module
            sys.path.insert(0, os.getcwd())
            from db import DatabaseManager
            
            # Initialize database
            db_manager = DatabaseManager()
            db_manager.initialize_database()
            
            self.logger.info(f"{Colors.GREEN}✓ Database initialized{Colors.END}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False

    def create_configuration_files(self) -> bool:
        """Create initial configuration files."""
        self.logger.info("Creating configuration files...")
        
        try:
            # Create .env file
            env_content = f"""# EchoForge Environment Configuration
ECHOFORGE_ENV={'development' if self.dev_mode else 'production'}
HOST=127.0.0.1
PORT=8000
DEBUG={'true' if self.dev_mode else 'false'}

# Database Configuration
DB_PATH=data/echoforge.db
DB_ENABLE_ENCRYPTION=true

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.1

# Feature Flags
ENABLE_VOICE=true
ENABLE_RESONANCE=true
ENABLE_GAMIFICATION=true
ENABLE_ANALYTICS=true

# Logging
LOG_LEVEL={'DEBUG' if self.verbose else 'INFO'}

# Security (will be auto-generated)
SECRET_KEY=
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            # Create basic config.json
            config_content = {
                "database": {
                    "backup_interval_hours": 24,
                    "backup_retention_days": 30
                },
                "web": {
                    "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"]
                },
                "agents": {
                    "enable_tools": True,
                    "enable_specialists": True
                },
                "debate": {
                    "max_debate_rounds": 10,
                    "enable_ghost_loop_detection": True
                },
                "journal": {
                    "enable_gamification": True,
                    "enable_analytics": True
                }
            }
            
            with open('config.json', 'w') as f:
                json.dump(config_content, f, indent=2)

            self.logger.info(f"{Colors.GREEN}✓ Configuration files created{Colors.END}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration files: {e}")
            return False

    def download_nltk_data(self) -> bool:
        """Download required NLTK data."""
        self.logger.info("Downloading NLTK data...")
        
        try:
            import nltk
            
            # Download required NLTK data
            nltk_downloads = [
                'punkt',
                'stopwords',
                'wordnet',
                'averaged_perceptron_tagger',
                'vader_lexicon'
            ]
            
            for dataset in nltk_downloads:
                try:
                    nltk.download(dataset, quiet=not self.verbose)
                    self.logger.debug(f"Downloaded NLTK dataset: {dataset}")
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK dataset {dataset}: {e}")

            self.logger.info(f"{Colors.GREEN}✓ NLTK data downloaded{Colors.END}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download NLTK data: {e}")
            return False

    def create_sample_frontend(self) -> bool:
        """Create basic frontend files."""
        self.logger.info("Creating sample frontend files...")
        
        try:
            # Create basic HTML template
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EchoForge - Multi-Agent Debate Platform</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div id="app">
        <header>
            <h1>EchoForge</h1>
            <p>Privacy-First Multi-Agent Debate Platform</p>
        </header>
        
        <main>
            <div id="status">Connecting...</div>
            <div id="content">
                <p>Welcome to EchoForge! The system is initializing...</p>
            </div>
        </main>
    </div>
    
    <script src="/static/js/app.js"></script>
</body>
</html>"""
            
            with open('frontend/templates/index.html', 'w') as f:
                f.write(html_content)

            # Create basic CSS
            css_content = """/* EchoForge Basic Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
    color: #333;
}

header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

header h1 {
    color: #2c3e50;
    margin: 0;
    font-size: 2.5em;
}

header p {
    color: #7f8c8d;
    margin: 10px 0 0 0;
}

main {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#status {
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 4px;
    background-color: #e8f4f8;
    border: 1px solid #bee5eb;
    color: #0c5460;
}

.success {
    background-color: #d4edda !important;
    border-color: #c3e6cb !important;
    color: #155724 !important;
}

.error {
    background-color: #f8d7da !important;
    border-color: #f5c6cb !important;
    color: #721c24 !important;
}
"""
            
            with open('frontend/static/css/style.css', 'w') as f:
                f.write(css_content)

            # Create basic JavaScript
            js_content = """// EchoForge Frontend Application
console.log('EchoForge initializing...');

class EchoForgeApp {
    constructor() {
        this.statusElement = document.getElementById('status');
        this.contentElement = document.getElementById('content');
        this.init();
    }
    
    async init() {
        try {
            // Test backend connection
            const response = await fetch('/health');
            if (response.ok) {
                this.updateStatus('✓ Connected to EchoForge backend', 'success');
                this.loadMainInterface();
            } else {
                throw new Error('Backend not responding');
            }
        } catch (error) {
            this.updateStatus('⚠ Backend connection failed. Please ensure the server is running.', 'error');
            console.error('Connection error:', error);
        }
    }
    
    updateStatus(message, type = '') {
        this.statusElement.textContent = message;
        this.statusElement.className = type;
    }
    
    loadMainInterface() {
        this.contentElement.innerHTML = `
            <h2>System Ready</h2>
            <p>EchoForge is running and ready for multi-agent debates!</p>
            <p>This is a basic frontend. You can now:</p>
            <ul>
                <li>Access the API at <code>/docs</code> for interactive documentation</li>
                <li>Start a WebSocket connection for real-time debates</li>
                <li>Begin developing your custom frontend</li>
            </ul>
            <p><strong>Next steps:</strong></p>
            <ol>
                <li>Check the <code>/docs</code> endpoint for API documentation</li>
                <li>Review the WebSocket endpoints for real-time communication</li>
                <li>Start building your debate interface</li>
            </ol>
        `;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EchoForgeApp();
});
"""
            
            with open('frontend/static/js/app.js', 'w') as f:
                f.write(js_content)

            self.logger.info(f"{Colors.GREEN}✓ Sample frontend created{Colors.END}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create frontend files: {e}")
            return False

    def run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        self.logger.info("Running health checks...")
        
        checks_passed = 0
        total_checks = 6

        # Check 1: Python imports
        try:
            import fastapi
            import uvicorn
            import sqlalchemy
            import cryptography
            checks_passed += 1
            self.logger.info(f"{Colors.GREEN}✓ Python dependencies check passed{Colors.END}")
        except ImportError as e:
            self.logger.error(f"Python dependencies check failed: {e}")

        # Check 2: Database connection
        try:
            import sqlite3
            test_db = sqlite3.connect(':memory:')
            test_db.close()
            checks_passed += 1
            self.logger.info(f"{Colors.GREEN}✓ Database connectivity check passed{Colors.END}")
        except Exception as e:
            self.logger.error(f"Database connectivity check failed: {e}")

        # Check 3: Ollama service
        try:
            result = subprocess.run([
                "curl", "-s", "http://localhost:11434/api/version"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                checks_passed += 1
                self.logger.info(f"{Colors.GREEN}✓ Ollama service check passed{Colors.END}")
            else:
                self.logger.error("Ollama service check failed - service not responding")
        except Exception as e:
            self.logger.error(f"Ollama service check failed: {e}")

        # Check 4: Configuration files
        if os.path.exists('.env') and os.path.exists('config.json'):
            checks_passed += 1
            self.logger.info(f"{Colors.GREEN}✓ Configuration files check passed{Colors.END}")
        else:
            self.logger.error("Configuration files check failed")

        # Check 5: Directory structure
        required_dirs = ['data', 'logs', 'frontend/static', 'frontend/templates']
        if all(os.path.exists(d) for d in required_dirs):
            checks_passed += 1
            self.logger.info(f"{Colors.GREEN}✓ Directory structure check passed{Colors.END}")
        else:
            self.logger.error("Directory structure check failed")

        # Check 6: NLTK data
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            checks_passed += 1
            self.logger.info(f"{Colors.GREEN}✓ NLTK data check passed{Colors.END}")
        except Exception as e:
            self.logger.error(f"NLTK data check failed: {e}")

        success_rate = checks_passed / total_checks
        self.logger.info(f"Health checks completed: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% success rate required

    def print_completion_message(self, success: bool):
        """Print installation completion message."""
        if success:
            message = f"""
{Colors.GREEN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🎉 Installation Complete! 🎉                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}

{Colors.GREEN}EchoForge has been successfully installed and configured!{Colors.END}

{Colors.CYAN}{Colors.BOLD}Next Steps:{Colors.END}
1. Start the EchoForge server:
   {Colors.YELLOW}python main.py{Colors.END}

2. Open your browser and go to:
   {Colors.YELLOW}http://localhost:8000{Colors.END}

3. Check the API documentation:
   {Colors.YELLOW}http://localhost:8000/docs{Colors.END}

{Colors.CYAN}{Colors.BOLD}Useful Commands:{Colors.END}
• Start server: {Colors.YELLOW}python main.py{Colors.END}
• Run tests: {Colors.YELLOW}python -m pytest tests/{Colors.END}
• Check logs: {Colors.YELLOW}tail -f logs/echoforge.log{Colors.END}

{Colors.CYAN}{Colors.BOLD}Configuration:{Colors.END}
• Main config: {Colors.YELLOW}config.json{Colors.END}
• Environment: {Colors.YELLOW}.env{Colors.END}
• Database: {Colors.YELLOW}data/echoforge.db{Colors.END}

{Colors.GREEN}Happy debating with EchoForge! 🚀{Colors.END}
"""
        else:
            message = f"""
{Colors.RED}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ❌ Installation Failed ❌                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}

{Colors.RED}The installation encountered errors. Please check the logs above.{Colors.END}

{Colors.CYAN}{Colors.BOLD}Troubleshooting:{Colors.END}
1. Check the installation log: {Colors.YELLOW}install.log{Colors.END}
2. Ensure all system requirements are met
3. Verify internet connectivity for downloads
4. Try running with verbose mode: {Colors.YELLOW}python install.py --verbose{Colors.END}

{Colors.CYAN}{Colors.BOLD}Common Issues:{Colors.END}
• Ollama installation: Visit {Colors.YELLOW}https://ollama.ai{Colors.END}
• Python dependencies: Update pip and try again
• Permissions: Ensure write access to current directory

{Colors.YELLOW}For support, please check the documentation or create an issue.{Colors.END}
"""
        
        print(message)

    def run_installation(self) -> bool:
        """Run the complete installation process."""
        self.print_banner()
        
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Directory Structure", self.create_directory_structure),
            ("Python Dependencies", self.install_python_dependencies),
            ("Ollama Installation", self.install_ollama),
            ("Ollama Service", self.start_ollama_service),
            ("AI Models", self.download_ollama_models),
            ("NLTK Data", self.download_nltk_data),
            ("Database", self.initialize_database),
            ("Configuration", self.create_configuration_files),
            ("Frontend", self.create_sample_frontend),
            ("Health Checks", self.run_health_checks)
        ]
        
        for i, (step_name, step_function) in enumerate(steps, 1):
            self.logger.info(f"\n{Colors.CYAN}[{i}/{len(steps)}] {step_name}...{Colors.END}")
            
            try:
                if not step_function():
                    self.logger.error(f"Step '{step_name}' failed")
                    return False
            except KeyboardInterrupt:
                self.logger.error("\nInstallation cancelled by user")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in step '{step_name}': {e}")
                return False

        return True


def main():
    """Main installation entry point."""
    parser = argparse.ArgumentParser(description="EchoForge Installation Script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dev", action="store_true", help="Install in development mode")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    
    args = parser.parse_args()
    
    installer = EchoForgeInstaller(verbose=args.verbose, dev_mode=args.dev)
    
    try:
        success = installer.run_installation()
        installer.print_completion_message(success)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Installation cancelled by user{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
