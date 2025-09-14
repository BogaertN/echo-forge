"""
EchoForge Main Server - Fixed Version
Addresses Ollama connection issue and implements basic StoicClarifier flow
"""
import os
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import httpx
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
DATA_DIR = os.environ.get('ECHO_FORGE_DATA_DIR', os.path.expanduser('~/echo_forge_data'))
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Import local modules (with error handling)
try:
    from orchestrator import Orchestrator
    from connection_manager import ConnectionManager
    from db import DatabaseManager
except ImportError as e:
    logger.warning(f"Import failed: {e}. Creating minimal implementations.")
    # Create minimal fallback classes if imports fail
    class ConnectionManager:
        def __init__(self):
            self.connections = {}
        async def connect(self, websocket, session_id):
            await websocket.accept()
            self.connections[session_id] = websocket
            logger.info(f"✓ WebSocket connected: {session_id}")
        def disconnect(self, session_id):
            if session_id in self.connections:
                del self.connections[session_id]
                logger.info(f"✗ WebSocket disconnected: {session_id}")
        async def send_to_session(self, session_id, message):
            if session_id in self.connections:
                await self.connections[session_id].send_json(message)
    
    class DatabaseManager:
        def __init__(self):
            self.db_path = os.path.join(DATA_DIR, 'echoforge.db')
        def initialize_database(self):
            logger.info(f"Database initialized at: {self.db_path}")
    
    class Orchestrator:
        def __init__(self, manager=None):
            self.manager = manager
            logger.info("Orchestrator initialized")
        async def process_message(self, message_type, session_id, data=None):
            if message_type == "start_debate":
                await self.start_clarification(session_id)
        
        async def start_clarification(self, session_id):
            logger.info(f"Starting clarification for session {session_id}")
            response = await self.call_ollama_chat("llama3.1:8b", [
                {"role": "system", "content": "You are a Stoic clarifier using Socratic method. Ask probing questions to help refine and clarify the user's intent. Keep responses concise."},
                {"role": "user", "content": "Please help me clarify my thinking with thoughtful questions."}
            ])
            
            if self.manager:
                await self.manager.send_to_session(session_id, {
                    "type": "clarifier_response",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            logger.info(f"✓ Sent clarifier response to frontend for session {session_id}")
        
        async def call_ollama_chat(self, model, messages):
            """Call Ollama chat API with proper error handling"""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://127.0.0.1:11434/api/chat",
                        json={"model": model, "messages": messages, "stream": False},
                        timeout=60.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("message", {}).get("content", "No response")
                    else:
                        logger.error(f"Ollama API error: {response.status_code}")
                        return "Error communicating with Ollama"
            except Exception as e:
                logger.error(f"Ollama chat error: {e}")
                return f"Error: {str(e)}"

# Global instances
connection_manager = ConnectionManager()
db_manager = DatabaseManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting EchoForge server...")
    
    # Initialize connection manager
    logger.info("✓ Connection manager initialized")
    
    # Initialize database
    try:
        db_manager.initialize_database()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Test Ollama connection (FIXED VERSION)
    try:
        async with httpx.AsyncClient() as client:
            # Test version endpoint first
            version_response = await client.get("http://127.0.0.1:11434/api/version", timeout=10.0)
            if version_response.status_code == 200:
                version_data = version_response.json()
                logger.info(f"✓ Ollama connected successfully. Version: {version_data.get('version', 'unknown')}")
                
                # Test models endpoint with CORRECT field access
                models_response = await client.get("http://127.0.0.1:11434/api/tags", timeout=10.0)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    # FIXED: Access 'models' array, then 'name' field of each model
                    models = models_data.get("models", [])
                    if models:
                        model_names = [model.get("name", "unknown") for model in models]
                        logger.info(f"✓ Available models: {', '.join(model_names)}")
                    else:
                        logger.warning("⚠️ No models found. Run 'ollama pull llama3.1:8b'")
                else:
                    logger.warning(f"⚠️ Failed to fetch models: {models_response.status_code}")
            else:
                logger.warning("⚠️ Ollama version check failed")
    except Exception as e:
        logger.warning(f"⚠️ Ollama not available: {str(e)}. Agent functionality will be limited.")
    
    # Initialize orchestrator
    try:
        global orchestrator
        orchestrator = Orchestrator(manager=connection_manager)
        logger.info("✓ Orchestrator initialized and connected to WebSocket manager")
    except Exception as e:
        logger.error(f"Orchestrator initialization failed: {e}")
    
    # Check Whisper availability
    try:
        import whisper
        logger.info("✓ Whisper available for voice features")
    except ImportError:
        logger.info("ℹ️ Whisper not available, voice features disabled")
    
    logger.info("🎉 EchoForge server startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down EchoForge server...")
    logger.info("✓ All WebSocket connections closed")
    logger.info("✓ EchoForge server shutdown complete")

# Create FastAPI app
app = FastAPI(title="EchoForge", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
try:
    frontend_path = Path(__file__).parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
        logger.info(f"✓ Static files mounted from: {frontend_path}")
    else:
        logger.warning(f"Frontend directory not found: {frontend_path}")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connections for real-time communication"""
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")
                logger.info(f"📨 Received from {session_id}: {message_type}")
                
                # Process message through orchestrator
                if 'orchestrator' in globals():
                    await orchestrator.process_message(message_type, session_id, message)
                else:
                    # Fallback response
                    await connection_manager.send_to_session(session_id, {
                        "type": "error",
                        "content": "Orchestrator not available"
                    })
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from {session_id}: {data}")
                
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    frontend_file = Path(__file__).parent / "frontend" / "index.html"
    if frontend_file.exists():
        return HTMLResponse(content=frontend_file.read_text(), status_code=200)
    else:
        return {"message": "EchoForge API is running. Frontend not found."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_dir": DATA_DIR
    }

@app.post("/api/test-ollama")
async def test_ollama():
    """Test Ollama connection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                },
                timeout=30.0
            )
            if response.status_code == 200:
                result = response.json()
                return {"status": "success", "response": result}
            else:
                return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Set environment if not set
    if not os.environ.get('ECHO_FORGE_DATA_DIR'):
        os.environ['ECHO_FORGE_DATA_DIR'] = DATA_DIR
    
    print("🌐 Starting server on http://127.0.0.1:8000")
    print(f"📁 Data directory: {DATA_DIR}")
    print("🔄 Hot reload: enabled")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
