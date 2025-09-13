#!/usr/bin/env python3
"""
Base agent class for EchoForge.
Provides common functionality for all AI agents including LLM integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
import logging
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageRole(Enum):
    """Message roles for conversation tracking."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    session_id: Optional[str] = None  # Fixed: Added session_id parameter
    tools_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    system_prompt_override: Optional[str] = None
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

@dataclass
class AgentResponse:
    """Response from an AI agent."""
    content: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    agent_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class BaseAgent:
    """Base class for all AI agents in EchoForge."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the base agent.
        
        Args:
            config: Agent configuration. If None, uses default config.
        """
        self.config = config or AgentConfig()
        self.agent_id = str(uuid.uuid4())
        self.conversation_history: List[ConversationMessage] = []
        self.agent_type = self.__class__.__name__
        
        # Set up logging for this agent
        self.logger = logging.getLogger(f"agents.{self.agent_type.lower()}")
        self.logger.info(f"Initialized {self.agent_type} agent: {self.agent_id}")
        
    @property
    def system_prompt(self) -> str:
        """Default system prompt. Override in subclasses."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
        return "You are a helpful AI assistant participating in structured debates and discussions."
    
    async def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> AgentResponse:
        """Generate a response using the configured LLM.
        
        Args:
            prompt: The input prompt/question
            context: Additional context for the response
            **kwargs: Additional parameters for the LLM
            
        Returns:
            AgentResponse with the generated content and metadata
        """
        try:
            # Try to import and use Ollama
            import ollama
            
            # Prepare the full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nUser: {prompt}"
            
            # Prepare messages for the conversation
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            for msg in self.conversation_history[-5:]:  # Keep last 5 messages for context
                messages.append({"role": msg.role.value, "content": msg.content})
            
            # Add current prompt
            messages.append({"role": "user", "content": full_prompt})
            
            # Generate response using Ollama
            self.logger.info(f"Generating response with model: {self.config.model}")
            
            response = ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **kwargs
                }
            )
            
            content = response['message']['content']
            
            # Add to conversation history
            self.conversation_history.append(
                ConversationMessage(MessageRole.USER, prompt)
            )
            self.conversation_history.append(
                ConversationMessage(MessageRole.ASSISTANT, content)
            )
            
            # Create response with metadata
            agent_response = AgentResponse(
                content=content,
                confidence=0.8,  # Default confidence
                agent_id=self.agent_id,
                metadata={
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "agent_type": self.agent_type,
                    "session_id": self.config.session_id,
                    "response_length": len(content),
                    "context_provided": context is not None
                }
            )
            
            self.logger.info(f"Generated response of {len(content)} characters")
            return agent_response
            
        except ImportError:
            self.logger.error("Ollama package not available")
            return self._create_error_response("Ollama package not installed")
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_msg: str) -> AgentResponse:
        """Create an error response when LLM generation fails."""
        fallback_content = f"I apologize, but I'm having trouble connecting to the language model. Error: {error_msg}"
        
        return AgentResponse(
            content=fallback_content,
            confidence=0.0,
            agent_id=self.agent_id,
            metadata={
                "error": True,
                "error_message": error_msg,
                "agent_type": self.agent_type,
                "session_id": self.config.session_id
            }
        )
    
    async def process_message(self, message: str, **kwargs) -> AgentResponse:
        """Process a message. Override in subclasses for specific behavior.
        
        Args:
            message: The input message to process
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with the processed response
        """
        return await self.generate_response(message, **kwargs)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history."""
        if not self.conversation_history:
            return "No conversation history"
        
        summary_lines = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role_indicator = "User" if msg.role == MessageRole.USER else "Agent"
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_lines.append(f"{role_indicator}: {preview}")
        
        return "\n".join(summary_lines)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model": self.config.model,
            "session_id": self.config.session_id,
            "conversation_length": len(self.conversation_history),
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt
        }
    
    async def test_connection(self) -> bool:
        """Test if the agent can connect to the LLM successfully."""
        try:
            test_response = await self.generate_response("Hello, can you respond briefly?")
            return test_response.confidence > 0 and not test_response.metadata.get("error", False)
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


class SpecializedAgent(BaseAgent):
    """Base class for specialized agents with specific roles."""
    
    def __init__(self, config: Optional[AgentConfig] = None, specialization: Optional[str] = None):
        """Initialize specialized agent.
        
        Args:
            config: Agent configuration
            specialization: The agent's area of specialization
        """
        super().__init__(config)
        self.specialization = specialization or "General"
        
    @property
    def system_prompt(self) -> str:
        """Specialized system prompt."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
            
        base_prompt = super().system_prompt
        specialized_prompt = f"{base_prompt} You are specialized in {self.specialization}."
        return specialized_prompt


def main():
    """Main function for testing agent functionality."""
    async def test_agent():
        print("EchoForge Base Agent Test")
        print("=" * 50)
        
        # Create test agent
        config = AgentConfig(
            model="llama3.1:8b",
            temperature=0.7,
            session_id="test_session"
        )
        
        agent = BaseAgent(config)
        print(f"Created agent: {agent.agent_type}")
        print(f"Agent ID: {agent.agent_id}")
        print(f"Session ID: {agent.config.session_id}")
        
        # Test connection
        print("\nTesting connection...")
        connection_ok = await agent.test_connection()
        print(f"Connection status: {'✓ OK' if connection_ok else '✗ Failed'}")
        
        if connection_ok:
            # Test basic response
            print("\nTesting response generation...")
            response = await agent.generate_response("What is 2+2?")
            print(f"Response: {response.content[:100]}...")
            print(f"Confidence: {response.confidence}")
        
        # Show agent info
        print(f"\nAgent Info: {agent.get_agent_info()}")
    
    # Run the test
    asyncio.run(test_agent())


if __name__ == "__main__":
    main()
