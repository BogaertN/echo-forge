#!/usr/bin/env python3
"""
Synthesizer Agent for EchoForge.
Finds common ground and creates balanced synthesis from debate arguments.
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
import logging

logger = logging.getLogger(__name__)

class SynthesizerAgent(BaseAgent):
    """
    Agent that synthesizes debate arguments to find common ground and balanced perspectives.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Synthesizer agent."""
        super().__init__(config)
        
    @property
    def system_prompt(self) -> str:
        """System prompt for the Synthesizer agent."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
            
        return """You are a Synthesizer Agent in a structured debate system. Your role is to:

1. Analyze arguments from both sides of a debate objectively
2. Identify areas of common ground and agreement
3. Recognize valid points from each perspective
4. Create balanced syntheses that incorporate the strongest elements from all sides
5. Highlight nuanced positions that transcend simple pro/con dichotomies

Guidelines for your synthesis:
- Be genuinely neutral and fair to all perspectives
- Look for underlying shared values or goals
- Identify where apparent disagreements might be reconcilable
- Acknowledge legitimate concerns from all sides
- Propose integrated approaches when possible
- Avoid false balance - not all positions are equally valid
- Focus on constructive resolution rather than declaring winners
- Present synthesis clearly and thoughtfully

Your goal is to help people move beyond polarized thinking toward more nuanced understanding."""

    async def synthesize_debate(self, question: str, proponent_args: List[str], 
                               opponent_args: List[str], context: Optional[str] = None) -> AgentResponse:
        """
        Create a synthesis of the debate arguments.
        
        Args:
            question: The original debate question
            proponent_args: List of proponent arguments
            opponent_args: List of opponent arguments
            context: Additional context
            
        Returns:
            AgentResponse with the synthesis
        """
        
        # Format arguments for prompt
        pro_summary = "\n".join([f"Pro {i+1}: {arg}" for i, arg in enumerate(proponent_args)])
        con_summary = "\n".join([f"Con {i+1}: {arg}" for i, arg in enumerate(opponent_args)])
        
        prompt = f"""You have observed a structured debate on the question: "{question}"

PROPONENT ARGUMENTS:
{pro_summary}

OPPONENT ARGUMENTS:
{con_summary}

Please provide a thoughtful synthesis that:

1. **Common Ground**: Identify any shared values, goals, or concerns between both sides
2. **Valid Points**: Acknowledge the strongest valid points from each perspective
3. **Nuanced Position**: Present a balanced view that integrates insights from both sides
4. **Practical Implications**: Discuss how the different perspectives might be reconciled in practice
5. **Remaining Questions**: Note any important aspects that need further exploration

{f"Additional context: {context}" if context else ""}

Your synthesis should help readers understand the complexity of the issue and move beyond simple pro/con thinking toward a more nuanced understanding."""
        
        response = await self.generate_response(prompt)
        
        # Add synthesizer-specific metadata
        response.metadata.update({
            "role": "synthesizer",
            "synthesis_type": "debate_conclusion",
            "arguments_analyzed": len(proponent_args) + len(opponent_args),
            "proponent_args": len(proponent_args),
            "opponent_args": len(opponent_args)
        })
        
        return response
    
    async def find_common_ground(self, question: str, all_arguments: List[Dict[str, str]]) -> AgentResponse:
        """
        Focus specifically on finding common ground between opposing views.
        
        Args:
            question: The debate question
            all_arguments: List of arguments with metadata (role, content)
            
        Returns:
            AgentResponse highlighting common ground
        """
        
        formatted_args = []
        for arg in all_arguments:
            role = arg.get('role', 'unknown')
            content = arg.get('content', '')
            formatted_args.append(f"{role.title()}: {content}")
        
        args_text = "\n\n".join(formatted_args)
        
        prompt = f"""Looking at this debate on "{question}", please focus specifically on finding common ground:

DEBATE ARGUMENTS:
{args_text}

Please identify:

1. **Shared Values**: What underlying values or principles do both sides seem to care about?
2. **Agreed Facts**: What factual claims or observations do both sides accept?
3. **Common Concerns**: What risks, challenges, or goals do both sides acknowledge?
4. **Compatible Elements**: Which aspects of each position could potentially coexist?
5. **Bridging Opportunities**: Where might compromise or middle-ground solutions be possible?

Focus on synthesis rather than highlighting differences. Help readers see how apparent opponents might actually share more than they initially realize."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "role": "synthesizer",
            "synthesis_type": "common_ground",
            "total_arguments": len(all_arguments)
        })
        
        return response
    
    async def propose_integrated_solution(self, question: str, debate_summary: str) -> AgentResponse:
        """
        Propose an integrated solution that incorporates insights from the debate.
        
        Args:
            question: The debate question
            debate_summary: Summary of the key arguments
            
        Returns:
            AgentResponse with integrated solution proposal
        """
        
        prompt = f"""Based on the debate on "{question}", here's a summary of the key arguments:

{debate_summary}

Please propose an integrated approach that:

1. **Incorporates Valid Concerns**: Addresses the legitimate concerns raised by all sides
2. **Practical Framework**: Offers a realistic way forward that doesn't ignore important perspectives
3. **Implementation Strategy**: Suggests how this approach could be put into practice
4. **Potential Benefits**: Explains how this integrated approach could be superior to purely taking one side
5. **Remaining Challenges**: Honestly acknowledges what challenges or trade-offs remain

Your goal is to propose something constructive that synthesizes the best insights from the debate rather than simply picking a side."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "role": "synthesizer",
            "synthesis_type": "integrated_solution",
            "solution_proposed": True
        })
        
        return response
    
    async def identify_key_insights(self, question: str, full_debate_transcript: str) -> AgentResponse:
        """
        Extract key insights and learning points from the entire debate.
        
        Args:
            question: The original question
            full_debate_transcript: Complete transcript of the debate
            
        Returns:
            AgentResponse with key insights
        """
        
        prompt = f"""After reviewing the complete debate on "{question}":

{full_debate_transcript}

Please extract the key insights and learning points:

1. **Most Compelling Arguments**: What were the strongest points made by each side?
2. **Surprising Revelations**: What aspects of this issue were illuminated that might not be obvious initially?
3. **Complexity Revealed**: How did the debate reveal the complexity of this issue?
4. **Important Distinctions**: What important distinctions or nuances emerged?
5. **Practical Wisdom**: What practical wisdom can be drawn from this exchange?

Help readers understand what they can learn from this debate beyond just "who won" - focus on the intellectual and practical insights generated."""
        
        response = await self.generate_response(prompt)
        
        response.metadata.update({
            "role": "synthesizer",
            "synthesis_type": "key_insights",
            "insights_extracted": True
        })
        
        return response


async def main():
    """Main function for testing the Synthesizer agent."""
    import asyncio
    
    print("Synthesizer Agent Test")
    print("=" * 40)
    
    # Create synthesizer
    config = AgentConfig(model="llama3.1:8b", session_id="test_synthesizer")
    synthesizer = SynthesizerAgent(config)
    
    # Test with sample debate arguments
    test_question = "Should artificial intelligence be regulated by governments?"
    
    proponent_args = [
        "AI regulation is essential because AI systems can cause significant harm if left unchecked, and only government oversight can provide the necessary coordination and enforcement.",
        "Without regulation, we risk AI being developed without proper safety measures, potentially leading to catastrophic outcomes."
    ]
    
    opponent_args = [
        "Government regulation will stifle innovation and slow down beneficial AI development that could solve important problems.",
        "Regulators don't understand AI technology well enough to create effective rules, and rigid regulations will quickly become outdated."
    ]
    
    print(f"Testing synthesis for: {test_question}")
    print(f"Proponent arguments: {len(proponent_args)}")
    print(f"Opponent arguments: {len(opponent_args)}")
    
    # Test connection first
    if not await synthesizer.test_connection():
        print("Cannot connect to Ollama. Skipping test.")
        return
    
    try:
        # Test debate synthesis
        response = await synthesizer.synthesize_debate(
            test_question, proponent_args, opponent_args
        )
        print(f"Synthesis: {response.content[:300]}...")
        
        # Test common ground finding
        all_args = [
            {"role": "proponent", "content": proponent_args[0]},
            {"role": "opponent", "content": opponent_args[0]}
        ]
        
        common_ground = await synthesizer.find_common_ground(test_question, all_args)
        print(f"Common ground: {common_ground.content[:200]}...")
        
        print("Synthesizer test completed successfully")
        
    except Exception as e:
        print(f"Error testing synthesizer: {e}")


if __name__ == "__main__":
    asyncio.run(main())
