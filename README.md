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
