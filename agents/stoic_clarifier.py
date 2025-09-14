"""
agents/stoic_clarifier.py - Socratic questioning agent for EchoForge
"""
import logging
import httpx
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all EchoForge agents"""
    
    def __init__(self, agent_id: str, model: str = "llama3.1:8b"):
        self.agent_id = agent_id
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"Initialized {self.__class__.__name__} agent: {agent_id}")
    
    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Ollama with proper error handling"""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 6 exchanges to avoid context overflow)
            recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
            messages.extend(recent_history)
            
            # Add current user prompt
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"Generating response with model: {self.model}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("message", {}).get("content", "No response generated")
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                    
                    logger.info(f"Generated response of {len(content)} characters")
                    return content
                else:
                    error_msg = f"Ollama API error: {response.status_code}"
                    logger.error(error_msg)
                    return error_msg
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info(f"Conversation history reset for {self.agent_id}")

class StoicClarifier(BaseAgent):
    """
    Socratic questioning agent that helps users refine and clarify their thoughts
    """
    
    def __init__(self, agent_id: str, model: str = "llama3.1:8b"):
        super().__init__(agent_id, model)
        self.system_prompt = """You are a Stoic clarifier using the Socratic method. Your role is to help users refine and clarify their thoughts through thoughtful questioning.

Key principles:
1. ENGAGE DIRECTLY with the user's specific question or topic
2. Ask relevant, probing questions that help them think deeper about THEIR specific situation
3. Focus on clarifying their goals, assumptions, and context for THEIR question
4. Be conversational and helpful, not abstract or philosophical
5. Keep responses focused and practical (2-3 specific questions)
6. Help them articulate what they really want to explore

You're helping them refine their question so it can be explored through structured debate. Make it personal and relevant to what they actually asked."""

    async def clarify_question(self, user_input: str) -> str:
        """
        Take user input and return clarifying questions to help refine their thinking
        """
        prompt = f"""A user has shared this thought or question: "{user_input}"

Using the Socratic method, ask 2-3 thoughtful questions that will help them:
1. Clarify what they really want to explore
2. Examine their underlying assumptions
3. Define key terms or concepts
4. Think more precisely about the issue

End with: "Once you've considered these questions, please share a more refined version of what you'd like to explore."
"""
        
        return await self.generate_response(prompt, self.system_prompt)
    
    async def refine_question(self, original_input: str, user_responses: str) -> str:
        """
        Take the original input and user's responses to clarifying questions,
        then produce a refined question suitable for debate
        """
        prompt = f"""Original user input: "{original_input}"

User's responses to clarifying questions: "{user_responses}"

Based on this clarification process, create a refined, debate-worthy question that:
1. Is specific and well-defined
2. Can be argued from multiple perspectives
3. Captures what the user really wants to explore
4. Is suitable for a structured debate between proponent and opponent agents

Format your response as:
"Refined question for debate: [THE REFINED QUESTION]"

Keep it concise but precise."""

        return await self.generate_response(prompt, self.system_prompt)

# Example usage and testing
async def test_stoic_clarifier():
    """Test function for the StoicClarifier"""
    clarifier = StoicClarifier("test_clarifier_001")
    
    # Test clarification
    user_input = "I'm thinking about getting a new job"
    response = await clarifier.clarify_question(user_input)
    print(f"Clarification response: {response}")
    
    # Test refinement
    user_responses = "I want more money and better work-life balance. I'm not sure if I'm ready for more responsibility though."
    refined = await clarifier.refine_question(user_input, user_responses)
    print(f"Refined question: {refined}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_stoic_clarifier())
