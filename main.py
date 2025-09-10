import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

import uvicorn
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException, 
    UploadFile, File, Form, Request, Depends, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: Whisper not available, voice features disabled")
    WHISPER_AVAILABLE = False
    whisper = None
import numpy as np
try:
    import soundfile as sf
except ImportError:
    print("soundfile not available")
    sf = None

# Local imports
from orchestrator import EchoForgeOrchestrator
from connection_manager import ConnectionManager
from db import init_database, get_db_connection
from encrypted_db import EncryptedDatabase
from journal import JournalManager
from resonance_map import ResonanceMapManager
from tools import ToolManager
from models import *
from utils import setup_logging, validate_environment, generate_session_id

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EchoForge",
    description="Privacy-first multi-agent LLM debate and journaling platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates for HTML responses
templates = Jinja2Templates(directory="frontend")

# Global managers and state
connection_manager = ConnectionManager()
orchestrator = None
journal_manager = None
resonance_manager = None
tool_manager = None
encrypted_db = None
whisper_model = None

# Session tracking
active_sessions: Dict[str, Dict] = {}
user_preferences: Dict[str, Dict] = {}

# Load Whisper model for voice transcription
def load_whisper_model():
    """Load Whisper model for voice transcription"""
    global whisper_model
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize all EchoForge components on startup"""
    global orchestrator, journal_manager, resonance_manager, tool_manager, encrypted_db
    
    try:
        logger.info("Starting EchoForge backend...")
        
        # Validate environment
        if not validate_environment():
            raise RuntimeError("Environment validation failed")
        
        # Initialize encrypted database
        encrypted_db = EncryptedDatabase("data/echo_forge.db")
        
        # Initialize database schema
        init_database()
        
        # Initialize managers
        journal_manager = JournalManager(encrypted_db)
        resonance_manager = ResonanceMapManager(encrypted_db)
        tool_manager = ToolManager()
        
        # Initialize orchestrator
        orchestrator = EchoForgeOrchestrator(
            db=encrypted_db,
            journal_manager=journal_manager,
            resonance_manager=resonance_manager,
            tool_manager=tool_manager
        )
        
        # Load Whisper model in background
        asyncio.create_task(asyncio.to_thread(load_whisper_model))
        
        logger.info("EchoForge backend started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start EchoForge backend: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down EchoForge backend...")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    # Cleanup active sessions
    for session_id in list(active_sessions.keys()):
        await cleanup_session(session_id)
    
    # Close database connections
    if encrypted_db:
        encrypted_db.close()
    
    logger.info("EchoForge backend shutdown complete")

# === WEBSOCKET ENDPOINTS ===

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, session_id)
    
    # Initialize session if new
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "debate_active": False,
            "current_question": None,
            "debate_history": [],
            "user_preferences": user_preferences.get(session_id, {})
        }
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Update session activity
            active_sessions[session_id]["last_activity"] = datetime.now()
            
            # Route message based on type
            await handle_websocket_message(session_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_websocket_message(session_id: str, message: Dict):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message.get("type")
        payload = message.get("payload", {})
        
        if message_type == "start_clarification":
            await handle_start_clarification(session_id, payload)
        elif message_type == "respond_to_clarification":
            await handle_clarification_response(session_id, payload)
        elif message_type == "start_debate":
            await handle_start_debate(session_id, payload)
        elif message_type == "save_journal":
            await handle_save_journal(session_id, payload)
        elif message_type == "get_resonance_map":
            await handle_get_resonance_map(session_id, payload)
        elif message_type == "update_preferences":
            await handle_update_preferences(session_id, payload)
        elif message_type == "ping":
            await connection_manager.send_message(session_id, {"type": "pong"})
        else:
            logger.warning(f"Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_start_clarification(session_id: str, payload: Dict):
    """Start Socratic clarification process"""
    question = payload.get("question", "").strip()
    if not question:
        await connection_manager.send_error(session_id, "Question is required")
        return
    
    # Update session
    active_sessions[session_id]["current_question"] = question
    active_sessions[session_id]["debate_active"] = False
    
    # Send status update
    await connection_manager.send_message(session_id, {
        "type": "clarification_started",
        "payload": {"question": question}
    })
    
    # Start clarification in background
    asyncio.create_task(run_clarification(session_id, question))

async def run_clarification(session_id: str, question: str):
    """Run clarification process in background"""
    try:
        async for update in orchestrator.start_clarification(question, session_id):
            if session_id in active_sessions:  # Check if session still active
                await connection_manager.send_message(session_id, update)
    except Exception as e:
        logger.error(f"Error in clarification: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_clarification_response(session_id: str, payload: Dict):
    """Handle user response to clarification"""
    response = payload.get("response", "").strip()
    if not response:
        await connection_manager.send_error(session_id, "Response is required")
        return
    
    # Continue clarification in background
    asyncio.create_task(continue_clarification(session_id, response))

async def continue_clarification(session_id: str, response: str):
    """Continue clarification process with user response"""
    try:
        async for update in orchestrator.continue_clarification(response, session_id):
            if session_id in active_sessions:
                await connection_manager.send_message(session_id, update)
    except Exception as e:
        logger.error(f"Error continuing clarification: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_start_debate(session_id: str, payload: Dict):
    """Start multi-agent debate"""
    clarified_question = payload.get("clarified_question", "").strip()
    debate_config = payload.get("config", {})
    
    if not clarified_question:
        await connection_manager.send_error(session_id, "Clarified question is required")
        return
    
    # Update session
    active_sessions[session_id]["debate_active"] = True
    active_sessions[session_id]["current_question"] = clarified_question
    
    # Send status update
    await connection_manager.send_message(session_id, {
        "type": "debate_started",
        "payload": {"question": clarified_question}
    })
    
    # Start debate in background
    asyncio.create_task(run_debate(session_id, clarified_question, debate_config))

async def run_debate(session_id: str, question: str, config: Dict):
    """Run multi-agent debate in background"""
    try:
        async for update in orchestrator.start_debate(question, session_id, config):
            if session_id in active_sessions and active_sessions[session_id]["debate_active"]:
                await connection_manager.send_message(session_id, update)
                
                # Store debate history
                if update.get("type") == "agent_response":
                    active_sessions[session_id]["debate_history"].append(update["payload"])
                    
    except Exception as e:
        logger.error(f"Error in debate: {e}")
        await connection_manager.send_error(session_id, str(e))
    finally:
        if session_id in active_sessions:
            active_sessions[session_id]["debate_active"] = False

async def handle_save_journal(session_id: str, payload: Dict):
    """Save journal entry"""
    try:
        entry_data = payload.get("entry", {})
        
        # Save journal entry
        entry_id = await journal_manager.create_entry(
            title=entry_data.get("title", ""),
            content=entry_data.get("content", ""),
            insights=entry_data.get("insights", []),
            question=entry_data.get("question", ""),
            debate_summary=entry_data.get("debate_summary", ""),
            session_id=session_id
        )
        
        # Update resonance map
        await resonance_manager.update_from_entry(entry_id)
        
        # Send confirmation
        await connection_manager.send_message(session_id, {
            "type": "journal_saved",
            "payload": {"entry_id": entry_id}
        })
        
    except Exception as e:
        logger.error(f"Error saving journal: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_get_resonance_map(session_id: str, payload: Dict):
    """Get resonance map data"""
    try:
        map_data = await resonance_manager.get_map_data(
            filter_params=payload.get("filters", {}),
            session_id=session_id
        )
        
        await connection_manager.send_message(session_id, {
            "type": "resonance_map_data",
            "payload": map_data
        })
        
    except Exception as e:
        logger.error(f"Error getting resonance map: {e}")
        await connection_manager.send_error(session_id, str(e))

async def handle_update_preferences(session_id: str, payload: Dict):
    """Update user preferences"""
    try:
        preferences = payload.get("preferences", {})
        user_preferences[session_id] = preferences
        
        if session_id in active_sessions:
            active_sessions[session_id]["user_preferences"] = preferences
        
        await connection_manager.send_message(session_id, {
            "type": "preferences_updated",
            "payload": {"preferences": preferences}
        })
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        await connection_manager.send_error(session_id, str(e))

# === HTTP ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend HTML"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": encrypted_db is not None,
            "orchestrator": orchestrator is not None,
            "whisper": whisper_model is not None,
            "active_sessions": len(active_sessions)
        }
    }

@app.post("/api/session")
async def create_session():
    """Create a new session"""
    session_id = generate_session_id()
    active_sessions[session_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "debate_active": False,
        "current_question": None,
        "debate_history": [],
        "user_preferences": {}
    }
    
    return {"session_id": session_id}

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "last_activity": session["last_activity"].isoformat(),
        "debate_active": session["debate_active"],
        "current_question": session["current_question"],
        "debate_history_count": len(session["debate_history"])
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    await cleanup_session(session_id)
    return {"message": "Session deleted"}

@app.post("/api/voice/transcribe")
async def transcribe_voice(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Transcribe voice input using Whisper"""
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Voice transcription not available")
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe audio
        result = whisper_model.transcribe(temp_file_path)
        transcription = result["text"].strip()
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Update session activity
        active_sessions[session_id]["last_activity"] = datetime.now()
        
        return {
            "transcription": transcription,
            "confidence": result.get("confidence", 0.0),
            "language": result.get("language", "en")
        }
        
    except Exception as e:
        logger.error(f"Error transcribing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/journal/entries")
async def get_journal_entries(
    session_id: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None
):
    """Get journal entries"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        entries = await journal_manager.get_entries(
            session_id=session_id,
            limit=limit,
            offset=offset,
            search_query=search
        )
        return {"entries": entries}
        
    except Exception as e:
        logger.error(f"Error getting journal entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/journal/entry/{entry_id}")
async def get_journal_entry(entry_id: str, session_id: str):
    """Get specific journal entry"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        entry = await journal_manager.get_entry(entry_id, session_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry
        
    except Exception as e:
        logger.error(f"Error getting journal entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gamification/stats")
async def get_gamification_stats(session_id: str):
    """Get gamification stats for user"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        stats = await journal_manager.get_gamification_stats(session_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting gamification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/resonance/map")
async def get_resonance_map_data(
    session_id: str,
    depth: int = 2,
    min_connections: int = 1
):
    """Get resonance map visualization data"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        map_data = await resonance_manager.get_map_data(
            filter_params={
                "depth": depth,
                "min_connections": min_connections
            },
            session_id=session_id
        )
        return map_data
        
    except Exception as e:
        logger.error(f"Error getting resonance map: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tools/search")
async def search_web(request: WebSearchRequest):
    """Perform web search using tool manager"""
    try:
        if not tool_manager.is_enabled():
            raise HTTPException(status_code=503, detail="Web search not enabled")
        
        results = await tool_manager.web_search(
            query=request.query,
            max_results=request.max_results
        )
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get current system configuration"""
    return {
        "models_available": orchestrator.get_available_models() if orchestrator else [],
        "voice_enabled": whisper_model is not None,
        "tools_enabled": tool_manager.is_enabled() if tool_manager else False,
        "version": "1.0.0"
    }

@app.post("/api/config/models")
async def update_model_config(config: ModelConfig):
    """Update model configuration"""
    try:
        if orchestrator:
            await orchestrator.update_model_config(config.dict())
        return {"message": "Model configuration updated"}
        
    except Exception as e:
        logger.error(f"Error updating model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === UTILITY FUNCTIONS ===

async def cleanup_session(session_id: str):
    """Clean up session data"""
    if session_id in active_sessions:
        # Stop any active debates
        if active_sessions[session_id]["debate_active"]:
            active_sessions[session_id]["debate_active"] = False
        
        # Disconnect WebSocket
        await connection_manager.disconnect(session_id)
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Session {session_id} cleaned up")

@app.middleware("http")
async def session_cleanup_middleware(request: Request, call_next):
    """Middleware to clean up inactive sessions"""
    # Clean up sessions older than 24 hours
    cutoff_time = datetime.now() - timedelta(hours=24)
    inactive_sessions = [
        session_id for session_id, session_data in active_sessions.items()
        if session_data["last_activity"] < cutoff_time
    ]
    
    for session_id in inactive_sessions:
        await cleanup_session(session_id)
    
    response = await call_next(request)
    return response

# === MAIN ===

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True
    )
