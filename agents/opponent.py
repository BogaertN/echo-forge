#!/usr/bin/env python3
"""
Opponent Agent for EchoForge.
Builds counter-arguments and defends the negative side of debates.
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

class OpponentAgent(BaseAgent):
    """
    Agent that builds and defends counter-arguments in debates.
    Takes the 'con' or 'no' side of questions.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Opponent agent."""
        super().__init__(config)
        self.debate_rounds = 0
        self.max_debate_rounds = 3
        
    @property
    def system_prompt(self) -> str:
        """System prompt for the Opponent agent."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
            
        return """You are an Opponent Agent in a structured debate system. Your role is to:

1. Build strong counter-arguments opposing the 'yes' or 'pro' side of questions
2. Present evidence, reasoning, and examples that challenge the affirmative position
3. Identify weaknesses, risks, and unintended consequences
4. Remain respectful but intellectually rigorous in your opposition
5. Use logical reasoning, credible sources, and compelling examples

Guidelines for your responses:
- Be skeptical but fair-minded
- Focus on logical flaws, missing evidence, or overlooked consequences
- Present alternative perspectives and interpretations
- Challenge assumptions underlying the affirmative position
- Stay focused on the specific question
- Aim for clarity and intellectual rigor
- Keep responses substantial but concise (2-3 paragraphs)

You are engaging in good-faith intellectual discourse to explore ideas thoroughly by providing necessary opposition."""

    async def make_opening_argument(self, question: str, proponent_argument: str, context: Optional[str] = None) -> AgentResponse:
        """
        Make the opening counter-argument responding to the proponent.
        
        Args:
            question: The debate question
            proponent_argument: The proponent's opening argument
            context: Additional context about the question
            
        Returns:
            AgentResponse with the counter-argument
        """
        self.debate_rounds += 1
        
        prompt = f"""You are responding in a structured debate on this question: "{question}"

The Proponent just argued:
"{proponent_argument}"

As the Opponent, you must argue against the affirmative position and challenge the Proponent's arguments.

Please provide your counter-argument that:
1. Clearly states your opposition to the affirmative position
2. Points out specific flaws or gaps in the Proponent's reasoning
3. Presents 2-3 strong reasons supporting the negative position
4. Includes evidence, examples, or logical reasoning for your position
5. Acknowledges complexity while maintaining your opposing stance

{f"Additional context: {context}" if context else ""}

Make a compelling case for why the affirmative position is flawed or insufficient."""
        
        response = await self.generate_response(prompt)
        
        # Add opponent-specific metadata
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "opponent",
            "argument_type": "counter_opening",
            "position": "negative"
        })
        
        return response
    
    async def make_rebuttal(self, proponent_argument: str, original_question: str) -> AgentResponse:
        """
        Make a rebuttal to a proponent's argument.
        
        Args:
            proponent_argument: The proponent's argument to respond to
            original_question: The original debate question
            
        Returns:
            AgentResponse with the rebuttal
        """
        self.debate_rounds += 1
        
        prompt = f"""You are continuing a debate on: "{original_question}"

The Proponent just argued:
"{proponent_argument}"

As the Opponent (arguing against the affirmative position), please provide a rebuttal that:
1. Acknowledges any valid points while maintaining your opposition
2. Identifies logical flaws, weak evidence, or problematic assumptions
3. Presents counter-evidence or alternative interpretations
4. Reinforces why the negative position is stronger
5. Maintains intellectual rigor while being respectful

Your rebuttal should strengthen the case against the affirmative position while addressing the Proponent's latest points."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "opponent",
            "argument_type": "rebuttal",
            "position": "negative"
        })
        
        return response
    
    async def make_closing_argument(self, debate_history: List[str], original_question: str) -> AgentResponse:
        """
        Make a closing argument summarizing the opponent's case.
        
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

As the Opponent, provide a strong closing argument that:
1. Summarizes your strongest challenges to the affirmative position
2. Explains why the negative position is more compelling
3. Highlights unresolved weaknesses in the Proponent's case
4. Makes a final persuasive case against the affirmative position

This is your final opportunity to convince the audience that the affirmative position should be rejected or at least viewed with serious skepticism."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "debate_round": self.debate_rounds,
            "role": "opponent",
            "argument_type": "closing",
            "position": "negative"
        })
        
        return response
    
    def reset_debate(self):
        """Reset the debate state for a new debate."""
        self.debate_rounds = 0
        self.clear_history()
        logger.info("Opponent debate state reset")
    
    def get_debate_status(self) -> dict:
        """Get the current debate status."""
        return {
            "rounds_completed": self.debate_rounds,
            "max_rounds": self.max_debate_rounds,
            "position": "negative",
            "role": "opponent",
            "agent_id": self.agent_id
        }


async def main():
    """Main function for testing the Opponent agent."""
    import asyncio
    
    print("Opponent Agent Test")
    print("=" * 40)
    
    # Create opponent
    config = AgentConfig(model="llama3.1:8b", session_id="test_opponent")
    opponent = OpponentAgent(config)
    
    # Test with a sample proponent argument
    test_question = "Should artificial intelligence be regulated by governments?"
    proponent_arg = "AI regulation is essential because AI systems can cause significant harm if left unchecked, and only government oversight can provide the necessary coordination and enforcement."
    
    print(f"Testing with question: {test_question}")
    print(f"Proponent argument: {proponent_arg[:100]}...")
    
    # Test connection first
    if not await opponent.test_connection():
        print("Cannot connect to Ollama. Skipping test.")
        return
    
    try:
        # Test counter-argument
        response = await opponent.make_opening_argument(test_question, proponent_arg)
        print(f"Counter-argument: {response.content[:200]}...")
        
        # Test rebuttal
        proponent_rebuttal = "Government regulation provides necessary oversight and prevents AI misuse."
        rebuttal = await opponent.make_rebuttal(proponent_rebuttal, test_question)
        print(f"Rebuttal: {rebuttal.content[:200]}...")
        
        # Show status
        status = opponent.get_debate_status()
        print(f"Status: {status['rounds_completed']} rounds completed")
        
    except Exception as e:
        print(f"Error testing opponent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
