import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import DebateOrchestrator, DebateSession, SessionState
from models import DebatePhase, AgentRole, MessageType
from config import get_config
from db import DatabaseManager


class TestDebateOrchestrator:
    """Test suite for the DebateOrchestrator class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock()
        config.debate.max_debate_rounds = 5
        config.debate.max_clarification_rounds = 3
        config.debate.enable_ghost_loop_detection = True
        config.debate.similarity_threshold = 0.85
        config.agents.enable_specialists = True
        config.agents.max_specialists_per_debate = 2
        return config

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        db_manager = AsyncMock(spec=DatabaseManager)
        db_manager.create_session = AsyncMock(return_value="test_session_123")
        db_manager.get_session = AsyncMock(return_value={
            'id': 'test_session_123',
            'question': 'Test question',
            'status': 'active',
            'created_at': datetime.now().isoformat()
        })
        db_manager.update_session = AsyncMock()
        db_manager.log_message = AsyncMock()
        db_manager.get_debate_history = AsyncMock(return_value=[])
        return db_manager

    @pytest.fixture
    def mock_agents(self):
        """Mock agent instances for testing."""
        agents = {}
        
        # Mock clarifier agent
        clarifier = AsyncMock()
        clarifier.initiate_clarification = AsyncMock(return_value=Mock(
            success=True,
            content="What aspects of this topic would you like to explore?",
            metadata={'clarification_round': 1}
        ))
        clarifier.process_user_response = AsyncMock(return_value=Mock(
            success=True,
            content="That's an interesting perspective. Can you elaborate?",
            metadata={'needs_more_clarification': False}
        ))
        agents['clarifier'] = clarifier
        
        # Mock proponent agent
        proponent = AsyncMock()
        proponent.generate_opening_statement = AsyncMock(return_value=Mock(
            success=True,
            content="I argue that this position is correct because...",
            metadata={'argument_strength': 8.5}
        ))
        proponent.respond_to_opponent = AsyncMock(return_value=Mock(
            success=True,
            content="While the opposition raises valid points, my position remains...",
            metadata={'response_round': 1}
        ))
        agents['proponent'] = proponent
        
        # Mock opponent agent
        opponent = AsyncMock()
        opponent.generate_opening_opposition = AsyncMock(return_value=Mock(
            success=True,
            content="I disagree with this position because...",
            metadata={'counter_arguments': 3}
        ))
        opponent.respond_to_proponent = AsyncMock(return_value=Mock(
            success=True,
            content="The proponent's argument fails to consider...",
            metadata={'response_round': 1}
        ))
        agents['opponent'] = opponent
        
        # Mock synthesizer agent
        synthesizer = AsyncMock()
        synthesizer.generate_comprehensive_synthesis = AsyncMock(return_value=Mock(
            success=True,
            content="After considering both perspectives, we can find common ground in...",
            metadata={'synthesis_quality': 9.0}
        ))
        agents['synthesizer'] = synthesizer
        
        # Mock auditor agent
        auditor = AsyncMock()
        auditor.audit_argument = AsyncMock(return_value=Mock(
            success=True,
            content="Argument quality assessment: Strong logical structure with minor issues...",
            metadata={'quality_score': 7.5, 'fallacies_detected': 0}
        ))
        agents['auditor'] = auditor
        
        return agents

    @pytest.fixture
    async def orchestrator(self, mock_config, mock_db_manager, mock_agents):
        """Create orchestrator instance with mocked dependencies."""
        with patch('orchestrator.get_config', return_value=mock_config), \
             patch('orchestrator.DatabaseManager', return_value=mock_db_manager):
            
            orchestrator = DebateOrchestrator()
            
            # Replace agents with mocks
            for agent_name, mock_agent in mock_agents.items():
                setattr(orchestrator, f"{agent_name}_agent", mock_agent)
            
            await orchestrator.initialize()
            return orchestrator

    @pytest.mark.asyncio
    async def test_session_creation(self, orchestrator, mock_db_manager):
        """Test creating a new debate session."""
        question = "Should artificial intelligence be regulated?"
        config = {'max_rounds': 5}
        
        session_id = await orchestrator.create_session(question, config)
        
        assert session_id == "test_session_123"
        mock_db_manager.create_session.assert_called_once()
        assert session_id in orchestrator.active_sessions

    @pytest.mark.asyncio
    async def test_clarification_phase(self, orchestrator, mock_agents):
        """Test the clarification phase of debate."""
        session_id = "test_session_123"
        question = "Should AI be regulated?"
        
        # Start clarification
        response = await orchestrator.start_clarification(session_id, question)
        
        assert response['success'] is True
        assert response['phase'] == DebatePhase.CLARIFICATION.value
        mock_agents['clarifier'].initiate_clarification.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_response_processing(self, orchestrator, mock_agents):
        """Test processing user responses during clarification."""
        session_id = "test_session_123"
        user_response = "I'm interested in the ethical implications."
        
        # Setup session state
        session = DebateSession(
            session_id=session_id,
            question="Should AI be regulated?",
            current_phase=DebatePhase.CLARIFICATION,
            state=SessionState.WAITING_FOR_USER
        )
        orchestrator.active_sessions[session_id] = session
        
        response = await orchestrator.process_user_response(session_id, user_response)
        
        assert response['success'] is True
        mock_agents['clarifier'].process_user_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_initiation(self, orchestrator, mock_agents):
        """Test starting the main debate phase."""
        session_id = "test_session_123"
        refined_question = "Should AI development be regulated to prevent potential risks?"
        
        # Setup session
        session = DebateSession(
            session_id=session_id,
            question=refined_question,
            current_phase=DebatePhase.CLARIFICATION,
            refined_question=refined_question
        )
        orchestrator.active_sessions[session_id] = session
        
        response = await orchestrator.start_debate(session_id)
        
        assert response['success'] is True
        assert session.current_phase == DebatePhase.OPENING_STATEMENTS
        mock_agents['proponent'].generate_opening_statement.assert_called_once()
        mock_agents['opponent'].generate_opening_opposition.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_round_progression(self, orchestrator, mock_agents):
        """Test progression through debate rounds."""
        session_id = "test_session_123"
        
        # Setup session in main debate phase
        session = DebateSession(
            session_id=session_id,
            question="Should AI be regulated?",
            current_phase=DebatePhase.MAIN_DEBATE,
            debate_round=1
        )
        orchestrator.active_sessions[session_id] = session
        
        response = await orchestrator.continue_debate(session_id)
        
        assert response['success'] is True
        # Should have responses from both agents
        assert len(response.get('responses', [])) >= 2

    @pytest.mark.asyncio
    async def test_ghost_loop_detection(self, orchestrator):
        """Test detection of repetitive arguments (ghost loops)."""
        session_id = "test_session_123"
        
        # Setup session with repeated similar arguments
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.MAIN_DEBATE
        )
        
        # Add similar arguments to history
        similar_arg = "This is basically the same argument repeated."
        session.proponent_arguments = [similar_arg, similar_arg, similar_arg]
        orchestrator.active_sessions[session_id] = session
        
        # Mock similarity calculation to return high similarity
        with patch('orchestrator.calculate_similarity', return_value=0.9):
            ghost_detected = await orchestrator._detect_ghost_loops(session)
            
        assert ghost_detected is True

    @pytest.mark.asyncio
    async def test_specialist_consultation(self, orchestrator):
        """Test requesting specialist agent consultation."""
        session_id = "test_session_123"
        specialist_type = "science"
        
        # Setup session
        session = DebateSession(
            session_id=session_id,
            question="Should we invest in renewable energy?",
            current_phase=DebatePhase.MAIN_DEBATE
        )
        orchestrator.active_sessions[session_id] = session
        
        # Mock specialist manager
        mock_specialist_manager = AsyncMock()
        mock_specialist_manager.get_specialist_insights = AsyncMock(return_value=[
            Mock(
                specialist_type="science",
                content="From a scientific perspective, renewable energy offers...",
                confidence_level=0.85
            )
        ])
        
        with patch.object(orchestrator, 'specialist_manager', mock_specialist_manager):
            response = await orchestrator.request_specialist(session_id, specialist_type)
            
        assert response['success'] is True
        assert response['specialist_type'] == specialist_type
        mock_specialist_manager.get_specialist_insights.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesis_generation(self, orchestrator, mock_agents):
        """Test generating synthesis of debate arguments."""
        session_id = "test_session_123"
        
        # Setup session with completed debate
        session = DebateSession(
            session_id=session_id,
            question="Should AI be regulated?",
            current_phase=DebatePhase.MAIN_DEBATE,
            proponent_arguments=["AI regulation is necessary for safety..."],
            opponent_arguments=["AI regulation stifles innovation..."]
        )
        orchestrator.active_sessions[session_id] = session
        
        response = await orchestrator.generate_synthesis(session_id)
        
        assert response['success'] is True
        assert response['phase'] == DebatePhase.SYNTHESIS.value
        mock_agents['synthesizer'].generate_comprehensive_synthesis.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_process(self, orchestrator, mock_agents):
        """Test the argument auditing process."""
        session_id = "test_session_123"
        
        # Setup session with arguments to audit
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.AUDIT,
            proponent_arguments=["This is a strong argument with evidence..."],
            opponent_arguments=["This counter-argument challenges the premise..."]
        )
        orchestrator.active_sessions[session_id] = session
        
        response = await orchestrator.audit_debate(session_id)
        
        assert response['success'] is True
        # Should audit multiple arguments
        assert mock_agents['auditor'].audit_argument.call_count >= 2

    @pytest.mark.asyncio
    async def test_error_handling_agent_failure(self, orchestrator, mock_agents):
        """Test error handling when an agent fails."""
        session_id = "test_session_123"
        
        # Setup session
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.CLARIFICATION
        )
        orchestrator.active_sessions[session_id] = session
        
        # Make clarifier agent fail
        mock_agents['clarifier'].initiate_clarification.side_effect = Exception("Agent failed")
        
        response = await orchestrator.start_clarification(session_id, "Test question")
        
        assert response['success'] is False
        assert 'error' in response
        # Session should be in error state
        assert session.state == SessionState.ERROR

    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, orchestrator):
        """Test handling of session timeouts."""
        session_id = "test_session_123"
        
        # Create session with old timestamp
        old_time = datetime.now() - timedelta(hours=2)
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            created_at=old_time,
            last_activity=old_time,
            current_phase=DebatePhase.CLARIFICATION,
            state=SessionState.WAITING_FOR_USER
        )
        orchestrator.active_sessions[session_id] = session
        
        # Check for timeouts
        await orchestrator._check_session_timeouts()
        
        # Session should be marked as timed out
        assert session.state == SessionState.TIMEOUT

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, orchestrator):
        """Test handling multiple concurrent debate sessions."""
        questions = [
            "Should AI be regulated?",
            "Is renewable energy viable?", 
            "Should we colonize Mars?"
        ]
        
        # Create multiple sessions concurrently
        session_tasks = [
            orchestrator.create_session(question, {})
            for question in questions
        ]
        
        session_ids = await asyncio.gather(*session_tasks)
        
        assert len(session_ids) == 3
        assert len(orchestrator.active_sessions) == 3
        assert all(sid in orchestrator.active_sessions for sid in session_ids)

    @pytest.mark.asyncio
    async def test_debate_completion_criteria(self, orchestrator):
        """Test automatic debate completion based on various criteria."""
        session_id = "test_session_123"
        
        # Setup session at maximum rounds
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.MAIN_DEBATE,
            debate_round=5,  # Max rounds reached
            proponent_arguments=["Arg1", "Arg2", "Arg3"],
            opponent_arguments=["Counter1", "Counter2", "Counter3"]
        )
        orchestrator.active_sessions[session_id] = session
        
        # Mock config for max rounds
        with patch.object(orchestrator.config.debate, 'max_debate_rounds', 5):
            should_complete = await orchestrator._should_complete_debate(session)
            
        assert should_complete is True

    @pytest.mark.asyncio
    async def test_session_cleanup(self, orchestrator, mock_db_manager):
        """Test proper cleanup of completed sessions."""
        session_id = "test_session_123"
        
        # Setup completed session
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.COMPLETE,
            state=SessionState.COMPLETED
        )
        orchestrator.active_sessions[session_id] = session
        
        await orchestrator.cleanup_session(session_id)
        
        assert session_id not in orchestrator.active_sessions
        mock_db_manager.update_session.assert_called()

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, orchestrator):
        """Test collection of performance metrics during debate."""
        session_id = "test_session_123"
        
        # Setup session
        session = DebateSession(
            session_id=session_id,
            question="Test question",
            current_phase=DebatePhase.MAIN_DEBATE
        )
        orchestrator.active_sessions[session_id] = session
        
        # Record some metrics
        await orchestrator._record_performance_metric(session_id, "response_time", 1.5)
        await orchestrator._record_performance_metric(session_id, "agent_calls", 3)
        
        metrics = session.performance_metrics
        assert "response_time" in metrics
        assert "agent_calls" in metrics
        assert metrics["response_time"] == 1.5
        assert metrics["agent_calls"] == 3

    @pytest.mark.asyncio
    async def test_debate_state_persistence(self, orchestrator, mock_db_manager):
        """Test that debate state is properly persisted to database."""
        session_id = "test_session_123"
        question = "Test question"
        
        # Create session
        await orchestrator.create_session(question, {})
        
        # Start clarification
        await orchestrator.start_clarification(session_id, question)
        
        # Verify database calls for state persistence
        assert mock_db_manager.create_session.called
        assert mock_db_manager.update_session.called
        
        # Check that session updates include proper state information
        update_calls = mock_db_manager.update_session.call_args_list
        assert len(update_calls) > 0


class TestDebateSession:
    """Test suite for the DebateSession class."""

    def test_session_initialization(self):
        """Test proper initialization of debate session."""
        session_id = "test_123"
        question = "Test question"
        
        session = DebateSession(session_id=session_id, question=question)
        
        assert session.session_id == session_id
        assert session.question == question
        assert session.current_phase == DebatePhase.INITIALIZATION
        assert session.state == SessionState.ACTIVE
        assert session.debate_round == 0
        assert len(session.proponent_arguments) == 0
        assert len(session.opponent_arguments) == 0

    def test_session_state_transitions(self):
        """Test valid state transitions."""
        session = DebateSession("test", "question")
        
        # Test valid transitions
        session.state = SessionState.WAITING_FOR_USER
        assert session.state == SessionState.WAITING_FOR_USER
        
        session.state = SessionState.PROCESSING
        assert session.state == SessionState.PROCESSING
        
        session.state = SessionState.COMPLETED
        assert session.state == SessionState.COMPLETED

    def test_session_argument_tracking(self):
        """Test tracking of arguments during debate."""
        session = DebateSession("test", "question")
        
        # Add proponent arguments
        session.proponent_arguments.append("Proponent argument 1")
        session.proponent_arguments.append("Proponent argument 2")
        
        # Add opponent arguments
        session.opponent_arguments.append("Opponent argument 1")
        session.opponent_arguments.append("Opponent argument 2")
        
        assert len(session.proponent_arguments) == 2
        assert len(session.opponent_arguments) == 2

    def test_session_timing(self):
        """Test session timing calculations."""
        session = DebateSession("test", "question")
        
        # Simulate some elapsed time
        session.last_activity = datetime.now() - timedelta(minutes=30)
        
        # Calculate duration
        duration = datetime.now() - session.created_at
        assert duration.total_seconds() > 0

    def test_session_serialization(self):
        """Test serialization of session to dictionary."""
        session = DebateSession("test", "Test question?")
        session.current_phase = DebatePhase.MAIN_DEBATE
        session.debate_round = 2
        session.proponent_arguments = ["Arg1", "Arg2"]
        session.opponent_arguments = ["Counter1", "Counter2"]
        
        session_dict = session.to_dict()
        
        assert session_dict['session_id'] == "test"
        assert session_dict['question'] == "Test question?"
        assert session_dict['current_phase'] == DebatePhase.MAIN_DEBATE.value
        assert session_dict['debate_round'] == 2
        assert len(session_dict['proponent_arguments']) == 2
        assert len(session_dict['opponent_arguments']) == 2


class TestIntegration:
    """Integration tests for orchestrator with real components."""

    @pytest.mark.asyncio
    async def test_full_debate_flow_integration(self):
        """Test complete debate flow with minimal mocking."""
        # This test would use real database and agent instances
        # but with controlled/mock LLM responses to ensure deterministic behavior
        pass

    @pytest.mark.asyncio 
    async def test_database_integration(self):
        """Test integration with actual database operations."""
        # This test would use a test database to verify
        # all database operations work correctly
        pass

    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test integration with WebSocket communication."""
        # This test would verify WebSocket message handling
        # integrates properly with orchestrator
        pass


# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration for orchestrator."""
    return {
        'debate': {
            'max_debate_rounds': 3,
            'max_clarification_rounds': 2,
            'enable_ghost_loop_detection': True,
            'similarity_threshold': 0.8
        },
        'agents': {
            'enable_specialists': True,
            'max_specialists_per_debate': 1,
            'default_timeout': 30
        }
    }


# Utility functions for tests
def create_mock_agent_response(success=True, content="Mock response", metadata=None):
    """Create a mock agent response for testing."""
    response = Mock()
    response.success = success
    response.content = content
    response.metadata = metadata or {}
    return response


def create_test_session(session_id="test", question="Test question?", phase=DebatePhase.INITIALIZATION):
    """Create a test session with specified parameters."""
    return DebateSession(
        session_id=session_id,
        question=question,
        current_phase=phase,
        created_at=datetime.now(),
        last_activity=datetime.now()
    )


# Performance and load testing
@pytest.mark.performance
class TestPerformance:
    """Performance tests for orchestrator."""

    @pytest.mark.asyncio
    async def test_session_creation_performance(self, orchestrator):
        """Test performance of session creation under load."""
        import time
        
        start_time = time.time()
        
        # Create multiple sessions quickly
        tasks = [
            orchestrator.create_session(f"Question {i}", {})
            for i in range(10)
        ]
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds for 10 sessions

    @pytest.mark.asyncio
    async def test_memory_usage(self, orchestrator):
        """Test memory usage during debate operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple sessions
        for i in range(5):
            session_id = await orchestrator.create_session(f"Question {i}", {})
            await orchestrator.start_clarification(session_id, f"Question {i}")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
