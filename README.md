# EchoForge: Privacy-First LLM Debate and Journaling Platform

## Overview
EchoForge is an innovative, privacy-first platform designed to enhance personal reasoning, critical thinking, and cognitive growth through multi-agent LLM debates and lifelong journaling. Built with open-source, local-first technologies (Ollama, FastAPI, SQLite with SQLCipher), it empowers users to clarify thoughts, engage in structured debates, and maintain an auditable archive of insights—all offline by default. This project is currently in the Minimum Viable Product (MVP) phase, with all core components stubbed out and the UI loading, but awaiting full functionality.

### Vision
EchoForge aims to be a self-hosted, extensible tool for individuals and small teams, offering a Socratic dialogue experience, resonance mapping of ideas, and gamified motivation, all while prioritizing data sovereignty and user control.

## Current Status (MVP Phase)
- **Build State**: As of September 08, 2025, EchoForge is in the MVP development stage. All files and directories are implemented as skeletons or stubs, providing the structural foundation. The UI loads successfully via the FastAPI server, but core flows (debate, clarification, journaling, resonance mapping, gamification) are not yet interactive.
- **Recent Updates**:
  - Introduced `agents/base_agent.py` as the foundational class for all AI agents.
  - Added `agents/stoic_clarifier.py` for Socratic question refinement.
  - Enhanced existing files (`main.py`, `orchestrator.py`, etc.) with advanced feature placeholders.
  - Integrated WebSocket support, encrypted database setup, and tool management stubs.

## File and Folder Structure
- `main.py`: The FastAPI backend server, serving as the entry point and API hub.
- `orchestrator.py`: The core engine coordinating multi-agent debates, workflows, and system logic.
- `db.py`: Manages SQLite database initialization, schemas, and performance indexes.
- `encrypted_db.py`: Provides SQLCipher encryption wrapper for secure data storage.
- `journal.py`: Handles journal entry creation, search, and gamification features.
- `resonance_map.py`: Visualizes and manages knowledge graph connections.
- `tools.py`: Manages integration of external tools (e.g., web search, fact-checking).
- `agents/base_agent.py`: Defines the base class for all AI agents with LLM integration.
- `agents/stoic_clarifier.py`: Implements the Socratic clarification agent.
- `connection_manager.py`: Manages WebSocket connections for real-time communication.
- `frontend/index.html`: The main HTML interface for user interaction.
- `frontend/script.js`: JavaScript logic for frontend interactivity and D3.js visualization.
- `configs/agent_routing.yaml`: Configuration file for agent roles and model assignments.
- `README.md`: This project documentation and overview.

## File Purposes
- `main.py`: Centralizes the FastAPI application, handling WebSocket and HTTP endpoints, voice transcription, and static file serving.
- `orchestrator.py`: Orchestrates agent interactions, debate flows, synthesis, and system coordination.
- `db.py`: Establishes database schemas (journal entries, debate logs, audit logs) with migration support.
- `encrypted_db.py`: Wraps database operations with SQLCipher for encryption and secure key management.
- `journal.py`: Manages journal entries, supports search, and tracks gamification metrics (streaks, badges).
- `resonance_map.py`: Builds and visualizes a knowledge graph using NetworkX and D3.js.
- `tools.py`: Integrates optional tools like web search and fact-checking with rate limiting.
- `agents/base_agent.py`: Provides the base LLM integration and conversation management for all agents.
- `agents/stoic_clarifier.py`: Guides users through Socratic questioning to refine questions.
- `connection_manager.py`: Handles real-time WebSocket session management and broadcasting.
- `frontend/index.html`: Structures the user interface with HTML and CSS.
- `frontend/script.js`: Manages frontend logic, including D3.js for resonance maps and real-time updates.
- `configs/agent_routing.yaml`: Configures agent roles, models, and tool settings.

## Installation
1. **Prerequisites**: Install Python 3.9+, pip, and Git. Ensure `ECHO_FORGE_DATA_DIR` environment variable is set (e.g., `export ECHO_FORGE_DATA_DIR=~/echo_forge_data`).
2. **Clone Repository**: `git clone https://github.com/BogaertN/echo-forge.git && cd echo-forge`.
3. **Install Dependencies**: `pip install fastapi uvicorn sqlcipher3 openai-whisper ollama pyyaml networkx torch requests pydantic`.
4. **Run Server**: `python main.py` (starts on `http://127.0.0.1:8000`).
5. **Access UI**: Open `http://127.0.0.1:8000/` in your browser.

## Architecture
- **Backend**: FastAPI with WebSocket support, using Ollama for local LLM processing.
- **Database**: SQLite with SQLCipher for encrypted, local storage.
- **Agents**: Modular, class-based agents (e.g., `StoicClarifier`, `Proponent`) inheriting from `BaseAgent`.
- **Frontend**: HTML/CSS/JS with D3.js for dynamic resonance maps.
- **Security**: Offline-first, with encrypted DB and no cloud LLM dependencies.

## Contribution Guidelines
- **Fork and Clone**: Fork the repo, clone your fork, and create a feature branch.
- **Code Style**: Follow PEP 8, add docstrings, and include unit tests where possible.
- **Pull Requests**: Submit PRs to the `main` branch with detailed descriptions.
- **Issues**: Report bugs or suggest features via GitHub Issues.

## Roadmap
- **Short-Term**: Implement core flows (question → clarification → debate → journal/save).
- **Mid-Term**: Add voice input, full gamification, and advanced resonance mapping.
- **Long-Term**: Support multi-user mode, cloud sync (optional), and MoE (Mixture of Experts) agents.

## Known Issues and TODOs
- **Blockers**: LLM calls (Ollama) not yet functional; agent communication untested.
- **Design Gaps**: Gamification logic incomplete; resonance map visualization needs refinement.
- **TODOs**: Debug agent logic, connect backend flows, add error handling, document API endpoints.


# License
[To be determined] - Consider MIT or GPL for open-source alignment.

*Last Updated: September 08, 2025, 04:42 AM EDT*
EchoForge is a sophisticated platform that facilitates multi-agent debates using local LLMs, helping users explore complex topics through structured Socratic questioning, argument synthesis, and reflective journaling. Built with privacy-first principles, all processing happens locally on your machine.
🌟 Key Features
Multi-Agent Debate System

Socratic Clarification: AI-guided questioning to refine and deepen your initial thoughts
Structured Debates: Proponent and opponent agents engage in systematic argumentation
Specialist Consultation: Domain experts (science, ethics, economics, history, legal) provide contextual insights
Argument Synthesis: Intelligent combination of perspectives to find common ground
Quality Auditing: Automated detection of logical fallacies and argument weaknesses

Advanced AI Orchestration

Local LLM Integration: Powered by Ollama for complete privacy
Agent Specialization: Each AI agent has distinct roles and expertise
Ghost Loop Detection: Prevents repetitive circular arguments
Dynamic Routing: Intelligent selection of specialists based on content analysis
Performance Monitoring: Real-time tracking of agent effectiveness

Intelligent Journaling

Guided Reflection: AI-assisted processing of debate experiences
Insight Extraction: Automatic identification of learning patterns
Gamification: Points, badges, and progress tracking
Advanced Search: Full-text search with filtering and analytics
Export Options: Multiple formats for data portability

Knowledge Mapping

Resonance Mapping: Visual network of conceptual connections
Dynamic Updates: Real-time graph evolution during debates
Community Detection: Identification of concept clusters
Pathfinding: Discovery of connections between ideas
Graph Analytics: Centrality measures and network insights

Privacy and Security

Local Processing: All AI processing happens on your machine
Encrypted Storage: Optional database encryption with SQLCipher
No Data Collection: No telemetry or external data transmission
Session Isolation: Complete separation between different users/sessions
Secure Configuration: Automatic security key generation

🚀 Quick Start
Prerequisites

Python 3.8-3.12
4GB+ RAM (8GB+ recommended)
10GB+ free disk space (50GB+ recommended for full model collection)
Internet connection (for initial setup only)

Installation

Clone the repository
bashgit clone https://github.com/yourusername/echoforge.git
cd echoforge

Run the installer
bashpython install.py
For development setup:
bashpython install.py --dev --verbose

Start the server
bashpython main.py

Open your browser
http://localhost:8000


Manual Installation
If the automatic installer doesn't work:

Install dependencies
bashpip install -r requirements.txt

Install Ollama
bash# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download/windows

Download AI models
bashollama pull llama3.1
ollama pull llama3.1:8b

Initialize database
bashpython -c "from db import DatabaseManager; DatabaseManager().initialize_database()"

Start Ollama service
bashollama serve

Start EchoForge
bashpython main.py


📖 Usage Guide
Starting a Debate

Navigate to the main interface at http://localhost:8000
Enter your question - anything you want to explore deeply
Clarification Phase - The Socratic Clarifier will help refine your question through guided questioning
Debate Phase - Proponent and Opponent agents will engage in structured argumentation
Specialist Input - Domain experts may be called in based on the topic
Synthesis - The Synthesizer will find common ground and integrate perspectives
Audit - The Auditor will assess argument quality and identify any logical issues
Journaling - The Journal Assistant will guide you through reflection on the experience

API Usage
EchoForge provides a comprehensive REST API and WebSocket interface:
REST API
pythonimport httpx

# Start a new session
response = httpx.post("http://localhost:8000/api/sessions", json={
    "question": "Should artificial intelligence be regulated?",
    "config": {"max_rounds": 8}
})
session_id = response.json()["session_id"]

# Get session status
status = httpx.get(f"http://localhost:8000/api/sessions/{session_id}")
WebSocket Connection
javascriptconst ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send a message
ws.send(JSON.stringify({
    type: "start_debate",
    question: "Should artificial intelligence be regulated?"
}));
Configuration
EchoForge can be configured through environment variables or configuration files:
Environment Variables (.env)
bash# Environment
ECHOFORGE_ENV=development
HOST=127.0.0.1
PORT=8000
DEBUG=true

# Database
DB_PATH=data/echoforge.db
DB_ENABLE_ENCRYPTION=true

# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.1

# Features
ENABLE_VOICE=true
ENABLE_RESONANCE=true
ENABLE_GAMIFICATION=true
Configuration File (config.json)
json{
  "agents": {
    "enable_specialists": true,
    "max_specialists_per_debate": 2,
    "enable_tools": true
  },
  "debate": {
    "max_debate_rounds": 10,
    "enable_ghost_loop_detection": true,
    "auto_complete_on_convergence": true
  },
  "journal": {
    "enable_gamification": true,
    "points_per_entry": 10,
    "enable_analytics": true
  }
}
🏗️ Architecture
System Components
┌─────────────────────────────────────────────────────────────┐
│                        EchoForge                            │
├─────────────────────────────────────────────────────────────┤
│  Frontend (HTML/CSS/JS) ←→ FastAPI Backend ←→ WebSockets   │
├─────────────────────────────────────────────────────────────┤
│                     Orchestrator                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Clarifier   │ │ Proponent   │ │ Opponent    │          │
│  │ Agent       │ │ Agent       │ │ Agent       │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Synthesizer │ │ Auditor     │ │ Journal     │          │
│  │ Agent       │ │ Agent       │ │ Assistant   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────────────────────────────────────┐          │
│  │           Specialist Agents                 │          │
│  │ Science │ Ethics │ Economics │ History │ Legal │        │
│  └─────────────────────────────────────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Database (SQLite) │ Ollama LLMs │ Tools & Utilities      │
└─────────────────────────────────────────────────────────────┘
Agent Hierarchy

BaseAgent - Foundation for all AI agents
StoicClarifier - Socratic questioning and clarification
ProponentAgent - Builds affirmative arguments
OpponentAgent - Builds counter-arguments and rebuttals
SynthesizerAgent - Finds common ground and integration
AuditorAgent - Quality control and fallacy detection
JournalAssistantAgent - Guided reflection and insight extraction
SpecialistAgents - Domain-specific expertise

Data Flow

Question Input → Clarification → Refined Question
Debate Initialization → Agent Assignment → Argument Exchange
Specialist Consultation → Expert Analysis → Integration
Synthesis → Common Ground → Balanced Perspective
Audit → Quality Assessment → Improvement Recommendations
Journaling → Reflection → Insight Extraction → Storage

🛠️ Development
Project Structure
echoforge/
├── main.py                 # FastAPI application entry point
├── orchestrator.py         # Core debate orchestration logic
├── connection_manager.py   # WebSocket connection management
├── db.py                   # Database schema and operations
├── encrypted_db.py         # Encrypted database wrapper
├── journal.py              # Journaling system
├── resonance_map.py        # Knowledge graph management
├── tools.py                # External tool integrations
├── models.py               # Pydantic data models
├── utils.py                # Utility functions
├── config.py               # Configuration management
├── install.py              # Installation script
├── requirements.txt        # Python dependencies
├── agents/                 # AI agent implementations
│   ├── base_agent.py
│   ├── stoic_clarifier.py
│   ├── proponent.py
│   ├── opponent.py
│   ├── synthesizer.py
│   ├── auditor.py
│   ├── journal_assistant.py
│   └── specialist_agents.py
├── frontend/               # Web interface
│   ├── static/
│   └── templates/
├── data/                   # Database and user data
├── logs/                   # Application logs
└── tests/                  # Test suite
Running Tests
bash# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py
Code Quality
bash# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Security scan
bandit -r .
Development Setup

Clone and setup
bashgit clone https://github.com/yourusername/echoforge.git
cd echoforge
python install.py --dev --verbose

Install development tools
bashpip install black isort flake8 mypy pytest pre-commit
pre-commit install

Start development server
bashuvicorn main:app --reload --host 0.0.0.0 --port 8000

Enable debug logging
bashexport LOG_LEVEL=DEBUG
export DEBUG=true


📚 API Documentation
Interactive Documentation
Once the server is running, visit:

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

Key Endpoints
Sessions

POST /api/sessions - Create new debate session
GET /api/sessions/{session_id} - Get session status
DELETE /api/sessions/{session_id} - End session

Debates

POST /api/debates/start - Start debate process
GET /api/debates/{session_id}/status - Get debate status
POST /api/debates/{session_id}/message - Send user message

Journal

GET /api/journal/entries - List journal entries
POST /api/journal/entries - Create journal entry
GET /api/journal/search - Search journal entries

Resonance Map

GET /api/resonance/{session_id} - Get knowledge graph
POST /api/resonance/{session_id}/update - Update graph

Health and Status

GET /health - System health check
GET /api/status - Detailed system status
GET /api/config - Current configuration

WebSocket Events
Client → Server
json{
  "type": "start_debate",
  "question": "Your question here",
  "config": {}
}

{
  "type": "user_response",
  "content": "Your response",
  "session_id": "session_123"
}

{
  "type": "request_specialist",
  "specialist_type": "science",
  "session_id": "session_123"
}
Server → Client
json{
  "type": "agent_response",
  "agent": "clarifier",
  "content": "Agent's response",
  "metadata": {}
}

{
  "type": "debate_update",
  "phase": "main_debate",
  "round": 3,
  "participants": ["proponent", "opponent"]
}

{
  "type": "synthesis_complete",
  "synthesis": "Synthesis content",
  "insights": []
}
🔧 Configuration Reference
Database Configuration
json{
  "database": {
    "db_path": "data/echoforge.db",
    "enable_encryption": true,
    "connection_timeout": 30,
    "auto_backup_enabled": true,
    "backup_interval_hours": 24
  }
}
LLM Configuration
json{
  "llm": {
    "ollama_base_url": "http://localhost:11434",
    "default_model": "llama3.1",
    "model_settings": {
      "llama3.1": {
        "temperature": 0.7,
        "max_tokens": 2048
      }
    }
  }
}
Agent Configuration
json{
  "agents": {
    "enable_tools": true,
    "enable_specialists": true,
    "max_specialists_per_debate": 2,
    "default_timeout": 60,
    "enable_caching": true
  }
}
🎯 Use Cases
Educational Applications

Philosophy Classes: Explore ethical dilemmas through structured debate
Critical Thinking: Develop argumentation and analysis skills
Research Training: Learn to evaluate evidence and construct arguments
Debate Preparation: Practice with AI opponents before human debates

Personal Development

Decision Making: Explore complex personal decisions from multiple angles
Belief Examination: Challenge and refine your own beliefs
Learning Acceleration: Deepen understanding through Socratic questioning
Reflection Practice: Develop metacognitive awareness

Professional Applications

Strategy Development: Explore business decisions from multiple perspectives
Risk Assessment: Identify potential issues through devil's advocate analysis
Policy Analysis: Examine policy proposals from various stakeholder viewpoints
Research Planning: Develop research questions through systematic inquiry

Creative Applications

Character Development: Explore fictional character motivations and conflicts
Plot Development: Work through story conflicts and resolutions
World Building: Develop consistent fictional worlds through logical analysis
Philosophical Fiction: Explore abstract concepts through narrative

🚨 Troubleshooting
Common Issues
Ollama Connection Problems
bash# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Check available models
ollama list
Database Issues
bash# Check database permissions
ls -la data/

# Reset database
rm data/echoforge.db
python -c "from db import DatabaseManager; DatabaseManager().initialize_database()"
Memory Issues
bash# Check system memory
free -h  # Linux
vm_stat  # macOS

# Reduce model size in config
export DEFAULT_MODEL=llama3.1:8b
Port Conflicts
bash# Check what's using port 8000
lsof -i :8000  # Unix
netstat -ano | findstr :8000  # Windows

# Use different port
export PORT=8080
python main.py
Logging and Debugging
Enable Debug Mode
bashexport DEBUG=true
export LOG_LEVEL=DEBUG
python main.py
Check Logs
bash# Application logs
tail -f logs/echoforge.log

# Installation logs
tail -f install.log

# System logs (Linux)
journalctl -u echoforge
Performance Monitoring
bash# Monitor system resources
htop  # Linux/macOS
taskmgr  # Windows

# Monitor specific process
ps aux | grep python
Getting Help

Check the logs for error messages
Review configuration files for typos
Verify system requirements are met
Test individual components (database, Ollama, etc.)
Search existing issues on GitHub
Create a detailed bug report with logs and system info

🤝 Contributing
We welcome contributions to EchoForge! Here's how to get started:
Development Workflow

Fork the repository
Create a feature branch
bashgit checkout -b feature/amazing-feature

Make your changes
Run tests
bashpytest
black .
flake8 .

Commit your changes
bashgit commit -m "Add amazing feature"

Push to your fork
bashgit push origin feature/amazing-feature

Create a Pull Request

Contribution Guidelines

Code Style: Follow Black formatting and PEP 8
Testing: Add tests for new features
Documentation: Update docs for any API changes
Commit Messages: Use clear, descriptive commit messages
Dependencies: Minimize new dependencies

Areas for Contribution

New Agent Types: Implement specialized agents for different domains
Frontend Improvements: Enhance the user interface
Performance Optimization: Improve speed and memory usage
Testing: Expand test coverage
Documentation: Improve guides and examples
Internationalization: Add support for multiple languages

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Ollama for providing excellent local LLM infrastructure
FastAPI for the robust web framework
SQLAlchemy for database management
NetworkX for graph analysis capabilities
The open-source community for countless tools and libraries

📞 Support

GitHub Issues: Report bugs and request features
Discussions: Ask questions and share ideas
Wiki: Community-maintained documentation
Security Issues: Email security@echoforge.dev

🗓️ Roadmap
Version 1.1

 Voice input/output support
 Advanced visualization tools
 Plugin system for custom agents
 Multi-language support

Version 1.2

 Collaborative debates (multiple humans)
 Advanced analytics dashboard
 Export to academic formats
 Integration with external knowledge bases

Version 2.0

 Distributed deployment support
 Mobile application
 Advanced ML models
 Real-time collaboration features


EchoForge: Empowering critical thinking through AI-assisted debate and reflection.
Built with privacy, powered by local AI, designed for human flourishing.
