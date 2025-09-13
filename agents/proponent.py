#!/usr/bin/env python3
"""
Proponent Agent for EchoForge.
Builds affirmative arguments and defends the positive side of debates.
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Optional, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
import logging

logger = logging.getLogger(__name__)

class ProponentAgent(BaseAgent):
    """
    Agent that builds and defends affirmative arguments in debates.
    Takes the 'pro' or 'yes' side of questions.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Proponent agent."""
        super().__init__(config)
        self.debate_rounds = 0
        self.max_debate_rounds = 3
        
    @property
    def system_prompt(self) -> str:
        """System prompt for the Proponent agent."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
            
        return """You are a Proponent Agent in a structured debate system. Your role is to:

1. Build strong affirmative arguments supporting the 'yes' or 'pro' side of questions
2. Present evidence, reasoning, and examples that support your position
3. Anticipate counter-arguments and address potential weaknesses
4. Remain respectful but persuasive in your argumentation
5. Use logical reasoning, credible sources, and compelling examples

Guidelines for your responses:
- Be confident but not arrogant
- Use evidence-based reasoning
- Address potential objections proactively
- Stay focused on the specific question
- Aim for clarity and persuasiveness
- Keep responses substantial but concise (2-3 paragraphs)

You are engaging in good-faith intellectual discourse to explore ideas thoroughly."""

    async def make_opening_argument(self, question: str, context: Optional[str] = None) -> AgentResponse:
        """
        Make the opening affirmative argument for a debate question.
        
        Args:
            question: The debate question
            context: Additional context about the question
            
        Returns:
            AgentResponse with the opening argument
        """
        self.debate_rounds += 1
        
        prompt = f"""You are starting a structured debate on this question: "{question}"

As the Proponent, you must argue for the affirmative/positive side of this question.

Please provide your opening argument that:
1. Clearly states your position (supporting the 'yes' side)
2. Presents 2-3 strong reasons supporting your position
3. Includes evidence, examples, or logical reasoning
4. Acknowledges this is a complex topic worthy of debate

{f"Additional context: {context}" if context else ""}

Make a compelling case for why someone should agree with the affirmative position."""
        
        response = await self.generate_response(prompt)
        
        # Add proponent-specific metadata
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "proponent",
            "argument_type": "opening",
            "position": "affirmative"
        })
        
        return response
    
    async def make_rebuttal(self, opponent_argument: str, original_question: str) -> AgentResponse:
        """
        Make a rebuttal to an opponent's argument.
        
        Args:
            opponent_argument: The opponent's argument to respond to
            original_question: The original debate question
            
        Returns:
            AgentResponse with the rebuttal
        """
        self.debate_rounds += 1
        
        prompt = f"""You are continuing a debate on: "{original_question}"

The opponent just made this argument:
"{opponent_argument}"

As the Proponent (arguing for the affirmative position), please provide a rebuttal that:
1. Acknowledges any valid points the opponent made
2. Points out flaws, gaps, or weaknesses in their reasoning
3. Reinforces your own position with additional evidence or examples
4. Maintains respect while being intellectually rigorous

Your rebuttal should strengthen the case for the affirmative position while addressing the opponent's concerns."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "proponent",
            "argument_type": "rebuttal",
            "position": "affirmative"
        })
        
        return response
    
    async def make_closing_argument(self, debate_history: List[str], original_question: str) -> AgentResponse:
        """
        Make a closing argument summarizing the proponent's case.
        
        Args:
            debate_history: List of previous arguments in the debate
            original_question: The original debate question
            
        Returns:
            AgentResponse with the closing argument
        """
        
        history_summary = "\n".join([f"Round {i+1}: {arg[:200]}..." for i, arg in enumerate(debate_history)])
        
        prompt = f"""You are making the closing argument in a debate on: "{original_question}"

Here's a summary of the debate so far:
{history_summary}

As the Proponent, provide a strong closing argument that:
1. Summarizes your strongest points from the debate
2. Explains why the affirmative position is more compelling
3. Addresses the most serious challenges raised by the opponent
4. Makes a final persuasive case for your position

This is your final opportunity to convince the audience of the affirmative position."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "proponent", 
            "argument_type": "closing",
            "position": "affirmative"
        })
        
        return response
    
    def reset_debate(self):
        """Reset the debate state for a new debate."""
        self.debate_rounds = 0
        self.clear_history()
        logger.info("Proponent debate state reset")
    
    def get_debate_status(self) -> dict:
        """Get the current debate status."""
        return {
            "rounds_completed": self.debate_rounds,
            "max_rounds": self.max_debate_rounds,
            "position": "affirmative",
            "role": "proponent",
            "agent_id": self.agent_id
        }


async def main():
    """Main function for testing the Proponent agent."""
    import asyncio
    
    print("Proponent Agent Test")
    print("=" * 40)
    
    # Create proponent
    config = AgentConfig(model="llama3.1:8b", session_id="test_proponent")
    proponent = ProponentAgent(config)
    
    # Test questions
    test_questions = [
        "Should artificial intelligence be regulated by governments?",
        "Is remote work better than traditional office work?",
        "Should everyone learn programming in school?"
    ]
    
    for question in test_questions:
        print(f"\nTesting with question: {question}")
        
        # Test connection first
        if not await proponent.test_connection():
            print("Cannot connect to Ollama. Skipping test.")
            continue
        
        try:
            # Test opening argument
            response = await proponent.make_opening_argument(question)
            print(f"Opening argument: {response.content[:200]}...")
            
            # Simulate opponent response for rebuttal test
            opponent_arg = "I disagree because there are significant risks and downsides to consider."
            rebuttal = await proponent.make_rebuttal(opponent_arg, question)
            print(f"Rebuttal: {rebuttal.content[:200]}...")
            
            # Show status
            status = proponent.get_debate_status()
            print(f"Status: {status['rounds_completed']} rounds completed")
            
            # Reset for next question
            proponent.reset_debate()
            
        except Exception as e:
            print(f"Error testing question '{question}': {e}")


if __name__ == "__main__":
    asyncio.run(main())
