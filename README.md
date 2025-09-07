# EchoForge

A private, multi-agent LLM debate and journaling platform for personal reasoning and cognitive growth.

## Overview

EchoForge is an offline-first, privacy-centric application built on open-source large language models. It guides users through question clarification, structured debates between AI agents, insight synthesis, and journaling with resonance mapping for tracking cognitive connections and unresolved "ghost loops."

Key features:
- Socratic clarification of questions
- Configurable multi-agent debates with proponent, opponent, specialists
- Neutral synthesis and journaling
- Resonance maps for visualizing knowledge
- Gentle gamification for motivation
- Voice journaling with local transcription
- Optional tool integration (web search, etc.)
- Full local operation with encryption

## Installation

1. **Prerequisites**
   - Python 3.10+
   - Ollama for LLM management
   - Whisper for voice (included via openai-whisper)
   - SQLCipher for DB encryption
   - Install deps: `pip install fastapi uvicorn sqlcipher3 openai-whisper ollama python-multipart requests psutil`

2. **Setup**
   ```bash
   git clone [your-repo-link]
   cd echoforge
   python install_models.py --install-all
