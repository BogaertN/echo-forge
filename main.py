from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import uuid
import tempfile
import os
from pathlib import Path
import sqlcipher3
import whisper
import hashlib
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Import EchoForge components
from orchestrator import EchoForgeOrchestrator
from connection_manager import ConnectionManager
from encrypted_db import EncryptedDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting EchoForge server...")
    
    # Initialize global orchestrator
    global orchestrator
    orchestrator = EchoForgeOrchestrator()
    
    yield
    
    # Shutdown
    logger.info("Shutting down EchoForge server...")
    if orchestrator:
        orchestrator.shutdown()

# Initialize FastAPI app
app = FastAPI(
    title="EchoForge Private AI Reasoning",
    description="Local-first multi-agent debate and journaling platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS frontend)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Global variables
orchestrator: EchoForgeOrchestrator = None
connection_manager = ConnectionManager()
whisper_model = None

# Pydantic models for API
class SessionStartRequest(BaseModel):
    initial_question: str

class ClarificationRequest(BaseModel):
    session_id: str
    user_response: str
    conversation_history: List[Dict[str, str]]

class DebateConfigRequest(BaseModel):
    session_id: str
    rounds: int = 3
    tone: str = "neutral"
    specialists: List[str] = []
    enable_tools: bool = False

class JournalEntryRequest(BaseModel):
    session_id: str
    content: str
    user_edits: Optional[str] = None

class VoiceTranscriptionResponse(BaseModel):
    transcription: str
    language: str
    session_id: str
    timestamp: str

# Utility functions
def load_whisper():
    """Lazy load Whisper model"""
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
    return whisper_model

# Main frontend route
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main HTML interface"""
    try:
        with open("frontend/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>EchoForge</title></head>
        <body>
            <h1>EchoForge Backend Running</h1>
            <p>Frontend files not found. Please create frontend/index.html</p>
        </body>
        </html>
        """)

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket connection for real-time debate updates"""
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await connection_manager.send_to_session(session_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # Echo other messages back
                await connection_manager.send_to_session(session_id, {
                    "type": "echo",
                    "data": message,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
        logger.info(f"Client disconnected from session {session_id}")

# API Endpoints

@app.post("/api/session/start")
async def start_session(request: SessionStartRequest):
    """Start a new clarification session"""
    logger.debug(f"Starting session with question: {request.initial_question}")
    try:
        session_id = orchestrator.start_new_session(request.initial_question)
        logger.debug(f"Session created: {session_id}")
        
        # Send WebSocket update
        await connection_manager.send_to_session(session_id, {
            "type": "session_started",
            "session_id": session_id,
            "initial_question": request.initial_question,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "session_id": session_id,
            "status": "success",
            "message": "Session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Session start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clarification/start")
async def start_clarification(request: SessionStartRequest):
    """Begin the clarification process"""
    logger.debug(f"Starting clarification for session {session_id}")
    try:
        session_id = orchestrator.start_new_session(request.initial_question)
        result = orchestrator.clarify_question(session_id, request.initial_question)
        
        if result["status"] == "success":
            # Send WebSocket update
            await connection_manager.send_to_session(session_id, {
                "type": "clarification_started",
                "session_id": session_id,
                "clarifier_question": result["clarifier_question"],
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Clarification failed"))
            
    except Exception as e:
        logger.error(f"Clarification start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clarification/continue")
async def continue_clarification(request: ClarificationRequest):
    """Continue the clarification dialogue"""
    try:
        result = orchestrator.continue_clarification(
            request.session_id,
            request.user_response,
            request.conversation_history
        )
        
        # Send WebSocket update
        await connection_manager.send_to_session(request.session_id, {
            "type": "clarification_continued",
            "status": result["status"],
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Clarification continuation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/debate/start")
async def start_debate(request: DebateConfigRequest):
    """Start a structured debate session"""
    try:
        config = {
            "rounds": request.rounds,
            "tone": request.tone,
            "specialists": request.specialists,
            "enable_tools": request.enable_tools
        }
        
        result = orchestrator.start_debate(request.session_id, config)
        
        if result["status"] == "success":
            # Send WebSocket update
            await connection_manager.send_to_session(request.session_id, {
                "type": "debate_started",
                "debate_id": result["debate_id"],
                "config": config,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Debate start failed"))
            
    except Exception as e:
        logger.error(f"Debate start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/debate/round/{debate_id}")
async def run_debate_round(debate_id: str):
    """Execute one round of debate"""
    try:
        result = orchestrator.run_debate_round(debate_id)
        
        # Send WebSocket update with round results
        session_id = None  # Would extract from debate_id in real implementation
        if session_id:
            await connection_manager.send_to_session(session_id, {
                "type": "debate_round_complete",
                "debate_id": debate_id,
                "round_data": result,
                "timestamp": datetime.now().isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Debate round failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesis/generate")
async def generate_synthesis(session_id: str, debate_id: str, tone: str = "neutral"):
    """Generate synthesis from completed debate"""
    try:
        result = orchestrator.synthesize_debate(session_id, debate_id, tone)
        
        if result["status"] == "success":
            # Send WebSocket update
            await connection_manager.send_to_session(session_id, {
                "type": "synthesis_generated",
                "synthesis": result["synthesis"],
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Synthesis failed"))
            
    except Exception as e:
        logger.error(f"Synthesis generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/journal/create")
async def create_journal_entry(request: JournalEntryRequest):
    """Create a new journal entry"""
    try:
        # Parse synthesis data from content (in real implementation, this would be structured)
        synthesis_data = {"content": request.content}
        
        result = orchestrator.create_journal_entry(
            request.session_id,
            synthesis_data,
            request.user_edits
        )
        
        if result["status"] == "success":
            # Send WebSocket update
            await connection_manager.send_to_session(request.session_id, {
                "type": "journal_entry_created",
                "entry_id": result["entry_id"],
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Journal entry creation failed"))
            
    except Exception as e:
        logger.error(f"Journal entry creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/journal/search")
async def search_journal(query: str = "", limit: int = 10):
    """Search journal entries"""
    try:
        results = orchestrator.search_journal(query)
        return {
            "results": results[:limit],
            "total": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Journal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gamification/stats")
async def get_gamification_stats():
    """Get current gamification statistics"""
    try:
        stats = orchestrator.get_gamification_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Gamification stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...), session_id: str = "default"):
    """Transcribe uploaded voice file using local Whisper"""
    
    # Validate file format
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Transcribe with local Whisper model
        model = load_whisper()
        result = model.transcribe(tmp_file_path)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        transcription_result = VoiceTranscriptionResponse(
            transcription=result["text"],
            language=result.get("language", "unknown"),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Send WebSocket update
        await connection_manager.send_to_session(session_id, {
            "type": "voice_transcribed",
            "transcription": result["text"],
            "timestamp": datetime.now().isoformat()
        })
        
        return transcription_result
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"Voice transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current status of a session"""
    try:
        # This would query the session manager for real status
        return {
            "session_id": session_id,
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Session status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = orchestrator.get_gamification_stats() if orchestrator else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_connections": connection_manager.get_active_count(),
            "database_status": "connected" if orchestrator else "disconnected",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not found",
        "message": "The requested resource was not found",
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("EchoForge server startup complete")
    
    # Ensure data directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("frontend/static", exist_ok=True)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("EchoForge server shutting down")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()

# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    # Development configuration
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
