"""
agents/base_agent.py - Base class for all EchoForge agents
"""
import logging
import httpx
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all EchoForge agents with complete LLM integration
    """
    
    def __init__(self, agent_id: str, model: str = "llama3.1:8b", system_prompt: str = None):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0,
            "total_tokens": 0
        }
        
        logger.info(f"Initialized {self.__class__.__name__} agent: {agent_id}")
    
    async def generate_response(self, prompt: str, system_prompt: str = None, 
                              temperature: float = 0.7, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Generate response using Ollama with comprehensive error handling and metrics
        """
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        
        try:
            messages = []
            
            # Add system prompt if provided
            final_system_prompt = system_prompt or self.system_prompt
            if final_system_prompt:
                messages.append({"role": "system", "content": final_system_prompt})
            
            # Add conversation history (keep last 6 exchanges to avoid context overflow)
            recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
            messages.extend(recent_history)
            
            # Add current user prompt
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"Generating response with model: {self.model}")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("message", {}).get("content", "No response generated")
                    
                    # Calculate metrics
                    response_time = (time.time() - start_time) * 1000  # ms
                    token_count = len(content.split())  # Rough token estimate
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                    
                    # Update performance metrics
                    self.performance_metrics["successful_requests"] += 1
                    self.performance_metrics["total_tokens"] += token_count
                    
                    # Update average response time
                    total_time = (self.performance_metrics["average_response_time"] * 
                                (self.performance_metrics["successful_requests"] - 1) + response_time)
                    self.performance_metrics["average_response_time"] = total_time / self.performance_metrics["successful_requests"]
                    
                    logger.info(f"Generated response: {len(content)} chars, {token_count} tokens, {response_time:.1f}ms")
                    
                    return {
                        "content": content,
                        "success": True,
                        "response_time_ms": response_time,
                        "token_count": token_count,
                        "model": self.model
                    }
                else:
                    error_msg = f"Ollama API error: HTTP {response.status_code}"
                    logger.error(error_msg)
                    return {
                        "content": error_msg,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except httpx.TimeoutException:
            error_msg = "Request timeout - Ollama took too long to respond"
            logger.error(error_msg)
            return {
                "content": error_msg,
                "success": False,
                "error": "timeout"
            }
            
        except httpx.ConnectError:
            error_msg = "Connection error - Is Ollama running?"
            logger.error(error_msg)
            return {
                "content": error_msg,
                "success": False,
                "error": "connection_error"
            }
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                "content": error_msg,
                "success": False,
                "error": str(e)
            }
    
    async def generate_simple_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Simple response generation that returns just the content string
        """
        result = await self.generate_response(prompt, system_prompt)
        return result.get("content", "Error generating response")
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info(f"Conversation history reset for {self.agent_id}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history"""
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "total_messages": len(self.conversation_history),
            "latest_messages": self.conversation_history[-4:] if self.conversation_history else []
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        success_rate = (self.performance_metrics["successful_requests"] / 
                       max(1, self.performance_metrics["total_requests"])) * 100
        
        return {
            **self.performance_metrics,
            "success_rate": round(success_rate, 2),
            "agent_id": self.agent_id,
            "model": self.model
        }
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this agent"""
        self.system_prompt = system_prompt
        logger.info(f"System prompt updated for {self.agent_id}")
    
    def add_context(self, context: str, role: str = "system"):
        """Add context to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": context
        })
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the agent can communicate with Ollama"""
        try:
            result = await self.generate_response("Hello", "Respond with just 'OK' if you can see this.")
            return {
                "healthy": result.get("success", False),
                "model": self.model,
                "response_time_ms": result.get("response_time_ms", 0),
                "agent_id": self.agent_id
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "agent_id": self.agent_id
            }


class SpecializedAgent(BaseAgent):
    """
    Base class for specialized debate agents (Proponent, Opponent, etc.)
    """
    
    def __init__(self, agent_id: str, role: str, expertise_area: str = None, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.role = role
        self.expertise_area = expertise_area
        self.role_specific_prompts = self._get_role_prompts()
    
    def _get_role_prompts(self) -> Dict[str, str]:
        """Get role-specific system prompts"""
        prompts = {
            "proponent": """You are a skilled proponent in structured debates. Your role is to:
1. Present strong, evidence-based arguments supporting your assigned position
2. Use logical reasoning, data, and real-world examples
3. Address potential counterarguments proactively
4. Remain respectful while being persuasive
5. Structure your arguments clearly with main points and supporting evidence
6. Keep responses focused and substantive (3-4 paragraphs typically)""",
            
            "opponent": """You are a skilled opponent in structured debates. Your role is to:
1. Present compelling counter-arguments and critiques
2. Identify potential flaws, risks, and negative consequences
3. Use evidence and logical reasoning to challenge positions
4. Offer alternative perspectives and solutions
5. Be thorough but respectful in your analysis
6. Structure critiques clearly with specific points and evidence
7. Keep responses focused and substantive (3-4 paragraphs typically)""",
            
            "synthesizer": """You are a wise synthesizer who integrates different perspectives. Your role is to:
1. Identify valid points from all sides of a debate
2. Find areas of common ground and shared values
3. Suggest balanced approaches that address multiple concerns
4. Provide actionable insights and recommendations
5. Help users see nuance and complexity in issues
6. Bridge differences constructively
7. Offer thoughtful, balanced conclusions""",
            
            "auditor": """You are an impartial auditor who evaluates argument quality. Your role is to:
1. Assess the logical soundness of arguments
2. Identify logical fallacies and weak reasoning
3. Evaluate the quality and relevance of evidence
4. Point out biases and unsupported claims
5. Suggest improvements for stronger argumentation
6. Maintain objectivity and fairness in assessments""",
            
            "specialist": f"""You are a domain specialist with expertise in {self.expertise_area or 'your field'}. Your role is to:
1. Provide expert knowledge and insights
2. Offer technical accuracy and depth
3. Explain complex concepts clearly
4. Share relevant research and best practices
5. Identify domain-specific considerations others might miss
6. Balance expertise with accessibility"""
        }
        
        return prompts
    
    async def generate_role_response(self, prompt: str, context: str = None) -> str:
        """Generate response using role-specific system prompt"""
        system_prompt = self.role_specific_prompts.get(self.role, self.system_prompt)
        
        if context:
            full_prompt = f"Context: {context}\n\nRequest: {prompt}"
        else:
            full_prompt = prompt
            
        result = await self.generate_response(full_prompt, system_prompt)
        return result.get("content", f"Error generating {self.role} response")


# Specialized agent implementations
class ProponentAgent(SpecializedAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, "proponent", **kwargs)


class OpponentAgent(SpecializedAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, "opponent", **kwargs)


class SynthesizerAgent(SpecializedAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, "synthesizer", **kwargs)


class AuditorAgent(SpecializedAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, "auditor", **kwargs)


class SpecialistAgent(SpecializedAgent):
    def __init__(self, agent_id: str, expertise_area: str, **kwargs):
        super().__init__(agent_id, "specialist", expertise_area, **kwargs)
