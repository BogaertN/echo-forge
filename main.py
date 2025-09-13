#!/usr/bin/env python3
"""
EchoForge - Main FastAPI application server.
A privacy-first, multi-agent LLM debate and journaling platform.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import EchoForge components
try:
    from orchestrator import Orchestrator
    from connection_manager import ConnectionManager
    from db import DatabaseManager
    from agents.base_agent import AgentConfig
except ImportError as e:
    logging.error(f"Failed to import EchoForge components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
connection_manager: Optional[ConnectionManager] = None
orchestrator: Optional[Orchestrator] = None
db_manager: Optional[DatabaseManager] = None

async def check_ollama_connection() -> bool:
    """Check if Ollama is available and working."""
    try:
        # Test direct HTTP connection first
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            logger.info(f"✓ Ollama connected successfully. Version: {version_info.get('version', 'unknown')}")
            
            # Test ollama Python client
            try:
                import ollama
                models = ollama.list()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"✓ Available models: {', '.join(available_models) if available_models else 'None'}")
                
                if not available_models:
                    logger.warning("⚠️  No models found. You may need to download models with 'ollama pull'")
                
                return True
            except ImportError:
                logger.warning("⚠️  Ollama Python client not available. Install with: pip install ollama")
                return False
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"✗ Ollama HTTP connection failed: {e}")
    except Exception as e:
        logger.warning(f"✗ Ollama connection check failed: {e}")
    
    return False

async def initialize_whisper():
    """Initialize Whisper for voice features."""
    try:
        import whisper
        model = whisper.load_model("base")
        logger.info("✓ Whisper model loaded successfully")
        return model
    except ImportError:
        logger.info("ℹ️  Whisper not available, voice features disabled")
        return None
    except Exception as e:
        logger.warning(f"⚠️  Failed to load Whisper model: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting EchoForge server...")
    
    global connection_manager, orchestrator, db_manager
    
    try:
        # Initialize connection manager
        connection_manager = ConnectionManager()
        logger.info("✓ Connection manager initialized")
        
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        logger.info("✓ Database initialized")
        
        # Check Ollama connection
        ollama_available = await check_ollama_connection()
        if not ollama_available:
            logger.warning("⚠️  Ollama not available. Agent functionality will be limited.")
        
        # Initialize orchestrator and CONNECT to connection manager
        orchestrator = Orchestrator()
        orchestrator.set_connection_manager(connection_manager)
        logger.info("✓ Orchestrator initialized and connected to WebSocket manager")
        
        # Initialize Whisper (optional)
        whisper_model = await initialize_whisper()
        app.state.whisper_model = whisper_model
        
        # Set application state
        app.state.ollama_available = ollama_available
        app.state.connection_manager = connection_manager
        app.state.orchestrator = orchestrator
        app.state.db_manager = db_manager
        
        logger.info("🎉 EchoForge server startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize EchoForge: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down EchoForge server...")
    
    if connection_manager:
        await connection_manager.disconnect_all()
        logger.info("✓ All WebSocket connections closed")
    
    logger.info("✓ EchoForge server shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="EchoForge",
    description="A privacy-first, multi-agent LLM debate and journaling platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = project_root / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"✓ Static files mounted from: {frontend_dir}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "ollama_available": getattr(app.state, 'ollama_available', False),
        "whisper_available": getattr(app.state, 'whisper_model', None) is not None
    }

# System status endpoint
@app.get("/api/status")
async def get_system_status():
    """Get detailed system status."""
    try:
        # Database stats
        db_stats = {}
        if db_manager:
            db_stats = db_manager.get_database_stats()
        
        # Connection stats
        connection_stats = {"active_connections": 0}
        if connection_manager:
            connection_stats["active_connections"] = len(connection_manager.active_connections)
        
        return {
            "status": "running",
            "components": {
                "database": {"status": "connected", "stats": db_stats},
                "ollama": {"status": "connected" if getattr(app.state, 'ollama_available', False) else "disconnected"},
                "whisper": {"status": "available" if getattr(app.state, 'whisper_model', None) else "unavailable"},
                "websocket": {"status": "active", "stats": connection_stats}
            },
            "environment": {
                "data_dir": os.getenv('ECHO_FORGE_DATA_DIR', './data'),
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

# Main page
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML page."""
    html_file = project_root / "frontend" / "index.html"
    
    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Fallback HTML if frontend/index.html doesn't exist
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>EchoForge</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status { padding: 20px; margin: 20px 0; border-radius: 5px; }
        .status.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .button { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 EchoForge</h1>
        <div class="status success">
            <strong>✓ Server is running!</strong><br>
            EchoForge is up and ready to help you explore ideas through AI-powered debates.
        </div>
        
        <div class="status warning">
            <strong>⚠️ Frontend not found</strong><br>
            The main frontend interface (frontend/index.html) is not available. 
            This is a basic status page.
        </div>
        
        <h3>Available Endpoints:</h3>
        <a href="/health" class="button">Health Check</a>
        <a href="/api/status" class="button">System Status</a>
        <a href="/docs" class="button">API Documentation</a>
        
        <h3>Quick Test:</h3>
        <p>You can test the WebSocket connection by opening the browser developer console and running:</p>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
const ws = new WebSocket('ws://localhost:8000/ws/test_session');
ws.onopen = () => console.log('✓ WebSocket connected');
ws.onmessage = (event) => console.log('Received:', JSON.parse(event.data));
        </pre>
    </div>
</body>
</html>
        """)

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connections for real-time communication."""
    global connection_manager, orchestrator
    
    if not connection_manager:
        await websocket.close(code=1011, reason="Server not properly initialized")
        return
    
    try:
        await connection_manager.connect(websocket, session_id)
        logger.info(f"✓ WebSocket connected: {session_id}")
        
        # Send welcome message
        await connection_manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to EchoForge! Ready to start your debate journey."
        }, session_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            logger.info(f"📨 Received from {session_id}: {data.get('type', 'unknown')}")
            
            # Process message through orchestrator
            if orchestrator:
                try:
                    await orchestrator.handle_user_message(data, session_id)
                    
                except Exception as e:
                    logger.error(f"Error in orchestrator processing: {e}")
                    await connection_manager.send_personal_message({
                        "type": "error",
                        "message": f"Error processing your request: {str(e)}"
                    }, session_id)
            else:
                logger.error("❌ Orchestrator not available")
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": "Server orchestrator not available"
                }, session_id)
            
            # Echo back for testing
            if data.get('type') == 'ping':
                await connection_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": data.get('timestamp')
                }, session_id)
                
    except WebSocketDisconnect:
        logger.info(f"✗ WebSocket disconnected: {session_id}")
        if connection_manager:
            connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        if connection_manager:
            connection_manager.disconnect(session_id)

# API Routes
@app.post("/api/sessions")
async def create_session(request: Request):
    """Create a new debate session."""
    try:
        data = await request.json()
        question = data.get('question', '')
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # Create session in database
        if db_manager:
            success = db_manager.create_debate_session(session_id, question)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create session")
        
        return {
            "session_id": session_id,
            "question": question,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    try:
        if db_manager:
            messages = db_manager.get_session_messages(session_id)
            return {
                "session_id": session_id,
                "message_count": len(messages),
                "messages": messages[-10:]  # Last 10 messages
            }
        else:
            return {"session_id": session_id, "message_count": 0, "messages": []}
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")

def main():
    """Main function to run the server."""
    # Set up environment
    if not os.getenv('ECHO_FORGE_DATA_DIR'):
        data_dir = Path.cwd() / "data"
        data_dir.mkdir(exist_ok=True)
        os.environ['ECHO_FORGE_DATA_DIR'] = str(data_dir)
        logger.info(f"✓ Data directory set to: {data_dir}")
    
    # Configuration
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))
    reload = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🌐 Starting server on http://{host}:{port}")
    logger.info(f"📁 Data directory: {os.getenv('ECHO_FORGE_DATA_DIR')}")
    logger.info(f"🔄 Hot reload: {'enabled' if reload else 'disabled'}")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
