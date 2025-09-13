#!/usr/bin/env python3
"""
StoicClarifier Agent for EchoForge.
Uses Socratic questioning to help users refine and deepen their initial questions.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
import logging

logger = logging.getLogger(__name__)

class StoicClarifier(BaseAgent):
    """
    Agent that uses Socratic questioning to clarify and refine user questions.
    Helps users explore the deeper dimensions of their inquiries.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the StoicClarifier agent."""
        super().__init__(config)
        self.clarification_rounds = 0
        self.max_clarification_rounds = 3
        
    @property
    def system_prompt(self) -> str:
        """System prompt for the StoicClarifier agent."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
            
        return """You are a Socratic Clarifier, an AI agent specialized in helping people refine and deepen their questions through thoughtful inquiry.

Your role is to:
1. Listen carefully to the user's initial question or concern
2. Ask probing questions that help them explore assumptions, clarify terms, and consider different angles
3. Guide them toward a more precise, thoughtful formulation of their question
4. Help them understand the deeper dimensions of what they're asking

Use the Socratic method:
- Ask one focused question at a time
- Help them examine their assumptions
- Encourage them to define key terms
- Explore the broader context and implications
- Guide them to discover insights themselves

Be supportive, curious, and intellectually rigorous. Your goal is to help them arrive at a clearer, more meaningful question that will lead to better debates and insights.

Keep your responses concise but thoughtful. Ask genuine questions that promote deeper thinking."""

    async def clarify_question(self, original_question: str, context: Optional[str] = None) -> AgentResponse:
        """
        Primary method to clarify a user's question using Socratic dialogue.
        
        Args:
            original_question: The user's initial question
            context: Additional context about the question
            
        Returns:
            AgentResponse with clarifying questions and insights
        """
        self.clarification_rounds += 1
        
        # Construct the clarification prompt
        prompt = self._build_clarification_prompt(original_question, context)
        
        # Generate response
        response = await self.generate_response(prompt)
        
        # Add clarifier-specific metadata
        response.metadata.update({
            "clarification_round": self.clarification_rounds,
            "original_question": original_question,
            "clarification_type": "socratic_questioning"
        })
        
        return response
    
    def _build_clarification_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build the clarification prompt based on the question and context."""
        
        base_prompt = f"""A user has asked this question: "{question}"

Please help clarify this question by asking ONE thoughtful follow-up question that will help them:
- Examine their assumptions
- Define key terms more precisely  
- Consider the broader context
- Explore what they really want to understand

"""
        
        if context:
            base_prompt += f"Additional context: {context}\n\n"
        
        # Adjust approach based on clarification round
        if self.clarification_rounds == 1:
            base_prompt += "This is the first clarification. Focus on understanding what they really want to explore."
        elif self.clarification_rounds == 2:
            base_prompt += "This is a follow-up clarification. Help them be more specific about their question."
        else:
            base_prompt += "This is a final clarification. Help them formulate a clear, focused question ready for debate."
        
        base_prompt += "\n\nProvide your clarifying question along with a brief explanation of why this question matters."
        
        return base_prompt
    
    async def process_clarification_response(self, user_response: str, original_question: str) -> AgentResponse:
        """
        Process the user's response to a clarifying question.
        
        Args:
            user_response: User's response to the clarifying question
            original_question: The original question being clarified
            
        Returns:
            AgentResponse with next clarifying question or refined question
        """
        
        # Auto-complete after 2 rounds or if user gives substantial response
        if self.clarification_rounds >= 2 or len(user_response.strip()) > 100:
            # Final synthesis - create refined question for debate
            prompt = f"""Based on our clarification dialogue, the user started with: "{original_question}"

Their latest response was: "{user_response}"

Please provide a clear, refined version of their question that would be excellent for a structured debate. The refined question should be:
- Specific and focused
- Free of ambiguity  
- Suitable for exploring multiple perspectives
- Intellectually substantive

Format your response as:
REFINED QUESTION: [The refined question]
USER POSITION SUMMARY: [A brief summary of the user's perspective and what they want to explore]

This refined question will now be used for a multi-agent debate between Proponent and Opponent agents."""
            
            response = await self.generate_response(prompt)
            response.metadata.update({
                "clarification_stage": "completed",
                "refined_question": True,
                "original_question": original_question,
                "ready_for_debate": True
            })
            
            return response
        
        else:
            # Continue clarification
            prompt = f"""The user originally asked: "{original_question}"

In response to our clarification, they said: "{user_response}"

Based on their response, ask ONE more clarifying question that builds on what they've shared and helps them get even more specific about what they want to explore.

Focus on helping them think deeper about their question. After this, we'll proceed to the debate phase."""
            
            return await self.generate_response(prompt)
    
    async def suggest_debate_angles(self, refined_question: str) -> AgentResponse:
        """
        Suggest different angles for debating the refined question.
        
        Args:
            refined_question: The clarified question
            
        Returns:
            AgentResponse with suggested debate angles
        """
        prompt = f"""For this refined question: "{refined_question}"

Suggest 3-4 different angles or perspectives that could make for a rich, multi-faceted debate. For each angle, briefly explain:
- What perspective it represents
- What key arguments might emerge
- Why this angle is important to consider

Format your response to help the user understand the complexity and richness of their question."""
        
        response = await self.generate_response(prompt)
        response.metadata.update({
            "stage": "debate_preparation",
            "refined_question": refined_question,
            "suggestion_type": "debate_angles"
        })
        
        return response
    
    def reset_clarification(self):
        """Reset the clarification process for a new question."""
        self.clarification_rounds = 0
        self.clear_history()
        logger.info("Clarification process reset")
    
    def get_clarification_status(self) -> dict:
        """Get the current status of the clarification process."""
        return {
            "rounds_completed": self.clarification_rounds,
            "max_rounds": self.max_clarification_rounds,
            "is_complete": self.clarification_rounds >= self.max_clarification_rounds,
            "agent_id": self.agent_id,
            "conversation_length": len(self.conversation_history)
        }


async def main():
    """Main function for testing the StoicClarifier."""
    import asyncio
    
    print("StoicClarifier Agent Test")
    print("=" * 40)
    
    # Create clarifier
    config = AgentConfig(model="llama3.1:8b", session_id="test_clarifier")
    clarifier = StoicClarifier(config)
    
    # Test questions
    test_questions = [
        "What should I do with my life?",
        "Is AI dangerous?",
        "How can I be happy?",
        "Should I change jobs?"
    ]
    
    for question in test_questions:
        print(f"\n🤔 Original question: {question}")
        
        # Test connection first
        if not await clarifier.test_connection():
            print("❌ Cannot connect to Ollama. Skipping test.")
            continue
        
        try:
            # First clarification
            response1 = await clarifier.clarify_question(question)
            print(f"🎯 Clarification 1: {response1.content[:200]}...")
            
            # Simulate user response
            user_response = "I want to understand this better and make a good decision."
            response2 = await clarifier.process_clarification_response(user_response, question)
            print(f"🎯 Clarification 2: {response2.content[:200]}...")
            
            # Show status
            status = clarifier.get_clarification_status()
            print(f"📊 Status: {status['rounds_completed']}/{status['max_rounds']} rounds")
            
            # Reset for next question
            clarifier.reset_clarification()
            
        except Exception as e:
            print(f"❌ Error testing question '{question}': {e}")


if __name__ == "__main__":
    asyncio.run(main())
