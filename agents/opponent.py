import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from agents.base_agent import BaseAgent, ConversationMessage, AgentResponse, AgentConfig
from models import DebatePhase, ArgumentType, MessageType
from utils import extract_key_concepts, calculate_similarity, clean_text

logger = logging.getLogger(__name__)


@dataclass
class CounterArgument:
    """Represents a structured counter-argument."""
    target_claim: str
    refutation_type: str  # contradiction, undermining, alternative_explanation, etc.
    evidence: List[str]
    reasoning: str
    strength_score: float
    logical_structure: str
    fallacy_identified: Optional[str] = None


@dataclass
class CriticalAnalysis:
    """Analysis of proponent's argument for weaknesses."""
    claim: str
    evidence_quality: float
    logical_consistency: float
    identified_fallacies: List[str]
    assumptions: List[str]
    weaknesses: List[str]
    attack_vectors: List[str]


@dataclass
class OppositionContext:
    """Context for building opposition arguments."""
    original_question: str
    proponent_position: str
    proponent_arguments: List[str]
    debate_history: List[Dict[str, Any]]
    critical_analyses: List[CriticalAnalysis] = field(default_factory=list)
    counter_arguments: List[CounterArgument] = field(default_factory=list)
    opposition_strategy: str = "comprehensive"
    current_round: int = 1


class OpponentAgent(BaseAgent):
    """
    Agent specialized in building counter-arguments and challenging positions.
    
    Uses critical analysis, logical reasoning, and systematic opposition
    techniques to challenge proponent arguments and present alternative
    perspectives on debate topics.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.agent_role = "opponent"
        self.opposition_strategies = [
            "contradiction",
            "undermining",
            "alternative_explanation", 
            "burden_of_proof",
            "scope_limitation",
            "precedent_challenge",
            "consequence_analysis",
            "definitional_challenge"
        ]
        
        # Critical analysis tools
        self.fallacy_patterns = [
            "ad_hominem", "straw_man", "false_dichotomy", "slippery_slope",
            "appeal_to_authority", "appeal_to_emotion", "hasty_generalization",
            "circular_reasoning", "post_hoc", "bandwagon", "red_herring"
        ]

    async def analyze_proponent_argument(
        self, 
        argument: str, 
        context: OppositionContext
    ) -> CriticalAnalysis:
        """
        Perform critical analysis of proponent's argument to identify weaknesses.
        """
        try:
            # Build analysis prompt
            analysis_prompt = f"""
            You are an expert critical analyst examining arguments for logical flaws, 
            weak evidence, and questionable assumptions.

            ARGUMENT TO ANALYZE:
            {argument}

            CONTEXT:
            Question: {context.original_question}
            Proponent Position: {context.proponent_position}
            
            Please provide a detailed critical analysis including:
            
            1. LOGICAL STRUCTURE ANALYSIS:
               - Identify the main claim and supporting premises
               - Assess logical consistency and validity
               - Look for logical fallacies or reasoning errors
            
            2. EVIDENCE QUALITY ASSESSMENT:
               - Evaluate strength and reliability of evidence presented
               - Identify gaps in evidence or unsupported claims
               - Note any cherry-picking or selective evidence use
            
            3. ASSUMPTION IDENTIFICATION:
               - List underlying assumptions (stated and unstated)
               - Evaluate reasonableness of assumptions
               - Identify controversial or questionable assumptions
            
            4. WEAKNESS IDENTIFICATION:
               - Point out specific weaknesses in the argument
               - Identify attack vectors for counter-arguments
               - Note areas where the argument is vulnerable
            
            5. FALLACY DETECTION:
               - Identify any logical fallacies present
               - Explain how each fallacy undermines the argument
            
            Be thorough but fair in your analysis. Focus on the argument structure 
            and evidence quality, not personal attacks.
            
            Format your response as structured analysis with clear sections.
            """

            # Get analysis from LLM
            response = await self._get_completion(analysis_prompt, context_key="critical_analysis")
            
            if not response.success:
                logger.error(f"Failed to analyze argument: {response.error}")
                return self._create_default_analysis(argument)

            # Parse the analysis (simplified - in production would use more sophisticated parsing)
            analysis_text = response.content
            
            # Extract components (this is simplified - would use more robust parsing)
            fallacies = self._extract_fallacies(analysis_text)
            assumptions = self._extract_assumptions(analysis_text)
            weaknesses = self._extract_weaknesses(analysis_text)
            attack_vectors = self._extract_attack_vectors(analysis_text)
            
            # Score evidence quality and logical consistency
            evidence_score = self._assess_evidence_quality(analysis_text)
            logic_score = self._assess_logical_consistency(analysis_text)

            return CriticalAnalysis(
                claim=argument,
                evidence_quality=evidence_score,
                logical_consistency=logic_score,
                identified_fallacies=fallacies,
                assumptions=assumptions,
                weaknesses=weaknesses,
                attack_vectors=attack_vectors
            )

        except Exception as e:
            logger.error(f"Error in critical analysis: {e}")
            return self._create_default_analysis(argument)

    async def build_counter_argument(
        self, 
        target_analysis: CriticalAnalysis,
        context: OppositionContext,
        strategy: str = "comprehensive"
    ) -> CounterArgument:
        """
        Build a structured counter-argument based on critical analysis.
        """
        try:
            # Select specific refutation approach
            refutation_type = self._select_refutation_type(target_analysis, strategy)
            
            # Build counter-argument prompt
            counter_prompt = f"""
            You are a skilled debater building a systematic counter-argument.

            TARGET CLAIM TO REFUTE:
            {target_analysis.claim}

            CRITICAL ANALYSIS INSIGHTS:
            - Evidence Quality Score: {target_analysis.evidence_quality}/10
            - Logical Consistency Score: {target_analysis.logical_consistency}/10
            - Identified Fallacies: {', '.join(target_analysis.identified_fallacies)}
            - Key Assumptions: {', '.join(target_analysis.assumptions)}
            - Main Weaknesses: {', '.join(target_analysis.weaknesses)}

            REFUTATION STRATEGY: {refutation_type}

            CONTEXT:
            Original Question: {context.original_question}
            Current Round: {context.current_round}
            Previous Counter-Arguments: {len(context.counter_arguments)}

            Build a strong counter-argument using the {refutation_type} approach:

            1. CLEAR COUNTER-CLAIM:
               - State your opposing position clearly
               - Explain why the target claim is problematic

            2. EVIDENCE AND REASONING:
               - Provide evidence that contradicts or undermines the claim
               - Use logical reasoning to show flaws in the original argument
               - Address specific weaknesses identified in the analysis

            3. ALTERNATIVE PERSPECTIVE:
               - Present a different way of viewing the issue
               - Show how alternative interpretations are more reasonable

            4. LOGICAL STRUCTURE:
               - Use valid logical reasoning
               - Avoid fallacies while pointing out fallacies in the target claim
               - Build a coherent case step by step

            {"5. TOOL INTEGRATION: Consider requesting web search or fact-checking if needed to strengthen your argument." if self.config.tools_enabled else ""}

            Be persuasive but intellectually honest. Focus on the strongest possible
            counter-argument while maintaining logical rigor.
            
            Format as a structured argument with clear reasoning.
            """

            # Get counter-argument from LLM
            response = await self._get_completion(counter_prompt, context_key="counter_argument")
            
            if not response.success:
                logger.error(f"Failed to build counter-argument: {response.error}")
                return self._create_default_counter_argument(target_analysis.claim)

            # Extract key concepts and assess strength
            key_concepts = extract_key_concepts(response.content)
            strength_score = self._assess_argument_strength(response.content, target_analysis)
            
            # Parse logical structure
            logical_structure = self._parse_logical_structure(response.content)

            return CounterArgument(
                target_claim=target_analysis.claim,
                refutation_type=refutation_type,
                evidence=self._extract_evidence_points(response.content),
                reasoning=response.content,
                strength_score=strength_score,
                logical_structure=logical_structure,
                fallacy_identified=target_analysis.identified_fallacies[0] if target_analysis.identified_fallacies else None
            )

        except Exception as e:
            logger.error(f"Error building counter-argument: {e}")
            return self._create_default_counter_argument(target_analysis.claim)

    async def generate_opening_opposition(
        self, 
        question: str, 
        proponent_position: str,
        context: OppositionContext
    ) -> AgentResponse:
        """
        Generate an opening opposition statement challenging the proponent's position.
        """
        try:
            # Analyze the proponent's position first
            position_analysis = await self.analyze_proponent_argument(proponent_position, context)
            context.critical_analyses.append(position_analysis)

            opening_prompt = f"""
            You are a skilled debater presenting the opening opposition statement 
            in a formal debate.

            DEBATE QUESTION:
            {question}

            PROPONENT'S POSITION:
            {proponent_position}

            YOUR TASK:
            Present a compelling opening statement that challenges the proponent's 
            position and establishes your opposition case.

            OPENING STATEMENT STRUCTURE:

            1. POSITION STATEMENT:
               - Clearly state your opposing position
               - Explain why the proponent's view is problematic or incomplete

            2. MAIN ARGUMENTS:
               - Present 2-3 core arguments against the proponent's position
               - Use evidence, logic, and reasoning to support your points
               - Address the strongest aspects of their position to show you understand it

            3. FRAMEWORK ESTABLISHMENT:
               - Establish criteria for evaluating the debate question
               - Explain your analytical framework or perspective
               - Set up the key issues that will determine the debate outcome

            4. PREVIEW OF CASE:
               - Outline what you will prove in this debate
               - Preview your main lines of argument
               - Explain why your position is stronger

            {"5. EVIDENCE FOUNDATION: Use web search or fact-checking tools if needed to strengthen your opening." if self.config.tools_enabled else ""}

            DEBATE PRINCIPLES:
            - Be respectful but assertive
            - Use logical reasoning and evidence
            - Address the proponent's strongest points
            - Establish clear grounds for opposition
            - Maintain intellectual rigor

            Format as a structured opening statement suitable for formal debate.
            """

            response = await self._get_completion(opening_prompt, context_key="opening_opposition")
            
            if not response.success:
                return AgentResponse(
                    success=False,
                    content="Failed to generate opening opposition statement.",
                    error=response.error,
                    agent_type=self.agent_role
                )

            # Build counter-argument for the position
            counter_arg = await self.build_counter_argument(position_analysis, context)
            context.counter_arguments.append(counter_arg)

            # Track performance metrics
            await self._track_performance("opening_opposition", {
                "response_length": len(response.content),
                "key_concepts": len(extract_key_concepts(response.content)),
                "opposition_strength": counter_arg.strength_score
            })

            return AgentResponse(
                success=True,
                content=response.content,
                agent_type=self.agent_role,
                metadata={
                    "debate_phase": DebatePhase.OPENING_STATEMENTS.value,
                    "message_type": MessageType.OPENING_OPPOSITION.value,
                    "counter_arguments_generated": 1,
                    "fallacies_identified": len(position_analysis.identified_fallacies),
                    "opposition_strategy": context.opposition_strategy,
                    "argument_strength": counter_arg.strength_score
                }
            )

        except Exception as e:
            logger.error(f"Error generating opening opposition: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate opening opposition statement.",
                error=str(e),
                agent_type=self.agent_role
            )

    async def respond_to_proponent(
        self,
        proponent_response: str,
        context: OppositionContext
    ) -> AgentResponse:
        """
        Generate a response to the proponent's argument in ongoing debate.
        """
        try:
            # Analyze the new proponent argument
            new_analysis = await self.analyze_proponent_argument(proponent_response, context)
            context.critical_analyses.append(new_analysis)

            # Build counter-argument
            counter_arg = await self.build_counter_argument(new_analysis, context)
            context.counter_arguments.append(counter_arg)

            response_prompt = f"""
            You are responding in an ongoing debate as the opposition.

            PROPONENT'S LATEST ARGUMENT:
            {proponent_response}

            YOUR CRITICAL ANALYSIS FOUND:
            - Evidence Quality: {new_analysis.evidence_quality}/10
            - Logic Score: {new_analysis.logical_consistency}/10
            - Fallacies: {', '.join(new_analysis.identified_fallacies) if new_analysis.identified_fallacies else 'None identified'}
            - Key Weaknesses: {', '.join(new_analysis.weaknesses)}

            DEBATE CONTEXT:
            - Original Question: {context.original_question}
            - Your Position: Opposition to "{context.proponent_position}"
            - Round: {context.current_round}
            - Previous Exchanges: {len(context.debate_history)}

            YOUR RESPONSE SHOULD:

            1. DIRECT REFUTATION:
               - Address the specific points made by the proponent
               - Show where their argument fails or is insufficient
               - Use evidence and logic to counter their claims

            2. EXPOSE WEAKNESSES:
               - Highlight logical flaws or weak evidence
               - Point out unstated assumptions
               - Identify any fallacies in their reasoning

            3. STRENGTHEN YOUR CASE:
               - Reinforce your opposition position
               - Present additional evidence or arguments
               - Build on your previous points

            4. ADVANCE THE DEBATE:
               - Raise new challenges to their position
               - Introduce aspects they haven't addressed
               - Push the debate to deeper levels

            {"5. RESEARCH INTEGRATION: Use tools for fact-checking or additional evidence if beneficial." if self.config.tools_enabled else ""}

            DEBATE STANDARDS:
            - Be respectful but firm in opposition
            - Use evidence-based reasoning
            - Maintain logical consistency
            - Address their strongest points directly
            - Avoid personal attacks or irrelevant tangents

            Format as a structured debate response.
            """

            response = await self._get_completion(response_prompt, context_key="debate_response")
            
            if not response.success:
                return AgentResponse(
                    success=False,
                    content="Failed to generate debate response.",
                    error=response.error,
                    agent_type=self.agent_role
                )

            # Update context
            context.current_round += 1
            context.debate_history.append({
                "round": context.current_round - 1,
                "proponent_argument": proponent_response,
                "opponent_response": response.content,
                "analysis": new_analysis,
                "counter_argument": counter_arg,
                "timestamp": datetime.now().isoformat()
            })

            # Track performance
            await self._track_performance("debate_response", {
                "response_length": len(response.content),
                "arguments_refuted": len(new_analysis.weaknesses),
                "fallacies_identified": len(new_analysis.identified_fallacies),
                "counter_strength": counter_arg.strength_score,
                "round_number": context.current_round
            })

            return AgentResponse(
                success=True,
                content=response.content,
                agent_type=self.agent_role,
                metadata={
                    "debate_phase": DebatePhase.MAIN_DEBATE.value,
                    "message_type": MessageType.DEBATE_RESPONSE.value,
                    "round_number": context.current_round,
                    "fallacies_identified": len(new_analysis.identified_fallacies),
                    "weaknesses_exposed": len(new_analysis.weaknesses),
                    "counter_argument_strength": counter_arg.strength_score,
                    "analysis_confidence": new_analysis.logical_consistency
                }
            )

        except Exception as e:
            logger.error(f"Error generating debate response: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate debate response.",
                error=str(e),
                agent_type=self.agent_role
            )

    async def generate_final_rebuttal(
        self,
        context: OppositionContext
    ) -> AgentResponse:
        """
        Generate a final rebuttal summarizing the opposition case.
        """
        try:
            # Summarize all analyses and counter-arguments
            total_fallacies = sum(len(analysis.identified_fallacies) for analysis in context.critical_analyses)
            avg_evidence_quality = sum(analysis.evidence_quality for analysis in context.critical_analyses) / len(context.critical_analyses) if context.critical_analyses else 0
            avg_counter_strength = sum(counter.strength_score for counter in context.counter_arguments) / len(context.counter_arguments) if context.counter_arguments else 0

            rebuttal_prompt = f"""
            You are delivering the final rebuttal in a formal debate as the opposition.

            DEBATE SUMMARY:
            - Original Question: {context.original_question}
            - Proponent Position: {context.proponent_position}
            - Total Rounds: {context.current_round}
            - Arguments Analyzed: {len(context.critical_analyses)}
            - Counter-Arguments Presented: {len(context.counter_arguments)}

            ANALYSIS SUMMARY:
            - Total Fallacies Identified: {total_fallacies}
            - Average Evidence Quality Score: {avg_evidence_quality:.1f}/10
            - Average Counter-Argument Strength: {avg_counter_strength:.1f}/10

            KEY POINTS FROM DEBATE:
            {self._summarize_debate_points(context)}

            YOUR FINAL REBUTTAL SHOULD:

            1. SUMMARIZE YOUR CASE:
               - Recap your main opposition arguments
               - Highlight the strongest counter-points you've made
               - Show the cumulative weight of your case

            2. EXPOSE FUNDAMENTAL FLAWS:
               - Identify the core weaknesses in the proponent's position
               - Show how their arguments fail to address key challenges
               - Demonstrate insufficient evidence or reasoning

            3. FINAL CHALLENGE:
               - Present your strongest closing argument
               - Challenge the proponent's position at its foundation
               - Provide a compelling alternative perspective

            4. CONCLUSION:
               - Summarize why your opposition is justified
               - Explain what the proponent has failed to prove
               - Leave the audience with clear reasons to reject their position

            REBUTTAL STANDARDS:
            - Be conclusive and definitive
            - Use the strongest evidence and reasoning
            - Maintain respect while being forceful
            - Tie together all threads of your opposition
            - End with impact and conviction

            Format as a powerful closing rebuttal.
            """

            response = await self._get_completion(rebuttal_prompt, context_key="final_rebuttal")
            
            if not response.success:
                return AgentResponse(
                    success=False,
                    content="Failed to generate final rebuttal.",
                    error=response.error,
                    agent_type=self.agent_role
                )

            # Track final performance metrics
            await self._track_performance("final_rebuttal", {
                "rebuttal_length": len(response.content),
                "total_rounds": context.current_round,
                "total_fallacies_identified": total_fallacies,
                "avg_counter_strength": avg_counter_strength,
                "debate_comprehensiveness": len(context.critical_analyses)
            })

            return AgentResponse(
                success=True,
                content=response.content,
                agent_type=self.agent_role,
                metadata={
                    "debate_phase": DebatePhase.CLOSING_STATEMENTS.value,
                    "message_type": MessageType.FINAL_REBUTTAL.value,
                    "total_rounds": context.current_round,
                    "total_fallacies_identified": total_fallacies,
                    "average_evidence_quality": avg_evidence_quality,
                    "average_counter_strength": avg_counter_strength,
                    "opposition_effectiveness": self._calculate_opposition_effectiveness(context)
                }
            )

        except Exception as e:
            logger.error(f"Error generating final rebuttal: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate final rebuttal.",
                error=str(e),
                agent_type=self.agent_role
            )

    def _select_refutation_type(self, analysis: CriticalAnalysis, strategy: str) -> str:
        """Select the most appropriate refutation approach based on analysis."""
        if analysis.identified_fallacies:
            return "logical_fallacy_exposure"
        elif analysis.evidence_quality < 5.0:
            return "evidence_undermining"
        elif analysis.assumptions:
            return "assumption_challenge"
        elif strategy == "comprehensive":
            return "systematic_contradiction"
        else:
            return "alternative_explanation"

    def _assess_argument_strength(self, argument: str, target_analysis: CriticalAnalysis) -> float:
        """Assess the strength of a counter-argument."""
        base_score = 5.0
        
        # Bonus for addressing specific weaknesses
        if target_analysis.weaknesses:
            base_score += 1.0
        
        # Bonus for identifying fallacies
        if target_analysis.identified_fallacies:
            base_score += 1.5
        
        # Assessment based on argument length and complexity
        word_count = len(argument.split())
        if word_count > 200:
            base_score += 1.0
        
        # Check for evidence markers
        evidence_markers = ["research shows", "studies indicate", "data reveals", "evidence suggests"]
        if any(marker in argument.lower() for marker in evidence_markers):
            base_score += 1.0
        
        return min(base_score, 10.0)

    def _extract_fallacies(self, analysis_text: str) -> List[str]:
        """Extract identified fallacies from analysis text."""
        fallacies = []
        analysis_lower = analysis_text.lower()
        
        for fallacy in self.fallacy_patterns:
            if fallacy.replace("_", " ") in analysis_lower:
                fallacies.append(fallacy)
        
        return fallacies

    def _extract_assumptions(self, analysis_text: str) -> List[str]:
        """Extract assumptions from analysis text."""
        # Simplified extraction - would use more sophisticated NLP in production
        assumptions = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if 'assumption' in line.lower() and ':' in line:
                assumption = line.split(':', 1)[1].strip()
                if assumption:
                    assumptions.append(assumption)
        
        return assumptions[:5]  # Limit to top 5

    def _extract_weaknesses(self, analysis_text: str) -> List[str]:
        """Extract weaknesses from analysis text."""
        weaknesses = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['weakness', 'weak', 'flaw', 'problem']):
                if ':' in line:
                    weakness = line.split(':', 1)[1].strip()
                    if weakness:
                        weaknesses.append(weakness)
        
        return weaknesses[:5]  # Limit to top 5

    def _extract_attack_vectors(self, analysis_text: str) -> List[str]:
        """Extract attack vectors from analysis text."""
        vectors = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['attack', 'challenge', 'counter', 'refute']):
                if ':' in line:
                    vector = line.split(':', 1)[1].strip()
                    if vector:
                        vectors.append(vector)
        
        return vectors[:3]  # Limit to top 3

    def _assess_evidence_quality(self, analysis_text: str) -> float:
        """Assess evidence quality from analysis."""
        quality_indicators = {
            'strong evidence': 8.0,
            'solid evidence': 7.0,
            'good evidence': 6.0,
            'weak evidence': 3.0,
            'poor evidence': 2.0,
            'no evidence': 1.0,
            'lacking evidence': 2.0,
            'insufficient evidence': 3.0
        }
        
        analysis_lower = analysis_text.lower()
        for indicator, score in quality_indicators.items():
            if indicator in analysis_lower:
                return score
        
        return 5.0  # Default middle score

    def _assess_logical_consistency(self, analysis_text: str) -> float:
        """Assess logical consistency from analysis."""
        consistency_indicators = {
            'logically sound': 9.0,
            'consistent logic': 8.0,
            'valid reasoning': 7.0,
            'logical flaws': 3.0,
            'inconsistent': 2.0,
            'illogical': 1.0,
            'fallacious': 2.0,
            'contradictory': 2.0
        }
        
        analysis_lower = analysis_text.lower()
        for indicator, score in consistency_indicators.items():
            if indicator in analysis_lower:
                return score
        
        return 5.0  # Default middle score

    def _parse_logical_structure(self, argument: str) -> str:
        """Parse the logical structure of an argument."""
        # Simplified structure parsing
        if "therefore" in argument.lower() or "thus" in argument.lower():
            return "deductive"
        elif "because" in argument.lower() or "since" in argument.lower():
            return "causal"
        elif "however" in argument.lower() or "but" in argument.lower():
            return "contrastive"
        else:
            return "descriptive"

    def _extract_evidence_points(self, argument: str) -> List[str]:
        """Extract evidence points from argument text."""
        evidence_points = []
        sentences = argument.split('.')
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['research', 'study', 'data', 'evidence', 'shows', 'indicates']):
                evidence_points.append(sentence.strip())
        
        return evidence_points[:5]  # Limit to top 5

    def _summarize_debate_points(self, context: OppositionContext) -> str:
        """Summarize key points from the debate."""
        if not context.debate_history:
            return "No debate exchanges recorded."
        
        summary_points = []
        for i, exchange in enumerate(context.debate_history[:3], 1):  # Limit to last 3 exchanges
            if exchange.get('analysis') and exchange['analysis'].weaknesses:
                weakness = exchange['analysis'].weaknesses[0]
                summary_points.append(f"Round {i}: Exposed weakness - {weakness}")
        
        return '\n'.join(summary_points) if summary_points else "Key debate points not available."

    def _calculate_opposition_effectiveness(self, context: OppositionContext) -> float:
        """Calculate overall effectiveness of the opposition case."""
        if not context.critical_analyses:
            return 0.0
        
        # Factor in various metrics
        avg_evidence_quality = sum(a.evidence_quality for a in context.critical_analyses) / len(context.critical_analyses)
        total_fallacies = sum(len(a.identified_fallacies) for a in context.critical_analyses)
        avg_counter_strength = sum(c.strength_score for c in context.counter_arguments) / len(context.counter_arguments) if context.counter_arguments else 0
        
        # Calculate effectiveness score
        effectiveness = (
            (10 - avg_evidence_quality) * 0.3 +  # Lower opponent evidence quality = higher opposition effectiveness
            min(total_fallacies * 2, 10) * 0.3 +  # More fallacies identified = higher effectiveness
            avg_counter_strength * 0.4  # Higher counter-argument strength = higher effectiveness
        )
        
        return min(effectiveness, 10.0)

    def _create_default_analysis(self, argument: str) -> CriticalAnalysis:
        """Create a default analysis when LLM analysis fails."""
        return CriticalAnalysis(
            claim=argument,
            evidence_quality=5.0,
            logical_consistency=5.0,
            identified_fallacies=[],
            assumptions=["Analysis unavailable"],
            weaknesses=["Unable to analyze"],
            attack_vectors=["General disagreement"]
        )

    def _create_default_counter_argument(self, claim: str) -> CounterArgument:
        """Create a default counter-argument when generation fails."""
        return CounterArgument(
            target_claim=claim,
            refutation_type="general_opposition",
            evidence=["Counter-argument generation failed"],
            reasoning="Unable to generate detailed counter-argument at this time.",
            strength_score=3.0,
            logical_structure="descriptive"
        )

    async def cleanup(self):
        """Clean up agent resources."""
        await super().cleanup()
        logger.info(f"OpponentAgent {self.session_id} cleaned up successfully")
