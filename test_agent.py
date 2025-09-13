#!/usr/bin/env python3
"""
Test script for EchoForge agents.
Tests individual agent functionality and connections.
Run from project root: python test_agent.py
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import EchoForge components
try:
    from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
    from agents.stoic_clarifier import StoicClarifier
    from db import DatabaseManager
except ImportError as e:
    logger.error(f"Failed to import EchoForge components: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)

def check_ollama_service():
    """Check if Ollama service is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            logger.info(f"✓ Ollama service running, version: {version_info.get('version', 'unknown')}")
            return True
        else:
            logger.warning(f"✗ Ollama service responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"✗ Ollama service not accessible: {e}")
        return False

def check_ollama_models():
    """Check available Ollama models."""
    try:
        import ollama
        models_response = ollama.list()
        models = [model['name'] for model in models_response.get('models', [])]
        
        if models:
            logger.info(f"✓ Available models: {', '.join(models)}")
            return models
        else:
            logger.warning("✗ No models found in Ollama")
            return []
    except Exception as e:
        logger.warning(f"✗ Failed to get Ollama models: {e}")
        return []

def test_database():
    """Test database functionality."""
    logger.info("Testing database...")
    try:
        # Initialize database manager
        db = DatabaseManager()
        db.initialize_database()
        
        # Get stats
        stats = db.get_database_stats()
        logger.info(f"✓ Database working. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        return False

async def test_base_agent():
    """Test base agent functionality."""
    logger.info("Testing base agent...")
    
    try:
        # Create agent with configuration
        config = AgentConfig(
            model="llama3.1:8b",
            temperature=0.7,
            session_id="test_session_base"
        )
        
        agent = BaseAgent(config)
        logger.info(f"✓ Created {agent.agent_type} (ID: {agent.agent_id[:8]}...)")
        
        # Test connection
        connection_ok = await agent.test_connection()
        if connection_ok:
            logger.info("✓ Agent connection test passed")
        else:
            logger.warning("✗ Agent connection test failed")
            return False
        
        # Test simple response
        test_prompt = "Respond with exactly: 'Agent test successful'"
        response = await agent.generate_response(test_prompt)
        
        logger.info(f"✓ Agent response: {response.content[:100]}...")
        logger.info(f"  Confidence: {response.confidence}")
        logger.info(f"  Agent ID: {response.agent_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Base agent test failed: {e}")
        return False

async def test_stoic_clarifier():
    """Test StoicClarifier agent specifically."""
    logger.info("Testing StoicClarifier agent...")
    
    try:
        # Create StoicClarifier agent
        config = AgentConfig(
            model="llama3.1:8b",
            temperature=0.7,
            session_id="test_session_clarifier"
        )
        
        clarifier = StoicClarifier(config)
        logger.info(f"✓ Created {clarifier.agent_type} (ID: {clarifier.agent_id[:8]}...)")
        
        # Test clarification method
        test_question = "What should I do with my life?"
        logger.info(f"  Testing with question: '{test_question}'")
        
        response = await clarifier.clarify_question(test_question)
        logger.info(f"✓ Clarifier response: {response.content[:150]}...")
        logger.info(f"  Confidence: {response.confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ StoicClarifier test failed: {e}")
        return False

async def test_agent_conversation():
    """Test a multi-turn conversation with an agent."""
    logger.info("Testing multi-turn conversation...")
    
    try:
        config = AgentConfig(
            model="llama3.1:8b",
            temperature=0.7,
            session_id="test_conversation"
        )
        
        agent = BaseAgent(config)
        
        # Multi-turn conversation
        questions = [
            "Hello, what's your role?",
            "Can you help me think through problems?",
            "What was my first question?"
        ]
        
        for i, question in enumerate(questions, 1):
            logger.info(f"  Turn {i}: {question}")
            response = await agent.generate_response(question)
            logger.info(f"  Response {i}: {response.content[:100]}...")
        
        # Check conversation history
        history_summary = agent.get_conversation_summary()
        logger.info(f"✓ Conversation history: {len(agent.conversation_history)} messages")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Conversation test failed: {e}")
        return False

def print_system_info():
    """Print system information."""
    logger.info("EchoForge Agent Testing")
    logger.info("=" * 50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Project root: {project_root}")
    
    # Check environment variables
    data_dir = os.getenv('ECHO_FORGE_DATA_DIR', 'Not set')
    logger.info(f"ECHO_FORGE_DATA_DIR: {data_dir}")

async def main():
    """Main testing function."""
    print_system_info()
    
    # Track test results
    test_results = {}
    
    # Test 1: Ollama service
    logger.info("\n1. Testing Ollama service...")
    test_results['ollama_service'] = check_ollama_service()
    
    # Test 2: Ollama models
    logger.info("\n2. Testing Ollama models...")
    available_models = check_ollama_models()
    test_results['ollama_models'] = len(available_models) > 0
    
    # Test 3: Database
    logger.info("\n3. Testing database...")
    test_results['database'] = test_database()
    
    # Test 4: Base agent
    logger.info("\n4. Testing base agent...")
    test_results['base_agent'] = await test_base_agent()
    
    # Test 5: StoicClarifier agent
    logger.info("\n5. Testing StoicClarifier agent...")
    test_results['stoic_clarifier'] = await test_stoic_clarifier()
    
    # Test 6: Multi-turn conversation
    logger.info("\n6. Testing multi-turn conversation...")
    test_results['conversation'] = await test_agent_conversation()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! EchoForge is ready to go.")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Check logs above for details.")
        return False

if __name__ == "__main__":
    # Ensure we have the required environment
    if not os.path.exists('agents'):
        logger.error("❌ 'agents' directory not found. Are you running from the project root?")
        sys.exit(1)
    
    try:
        # Run the tests
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        sys.exit(1)
