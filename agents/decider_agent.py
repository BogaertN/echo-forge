import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from agents.base_agent import BaseAgent, ConversationMessage, AgentResponse, AgentConfig
from models import DebatePhase, MessageType, SynthesisType
from utils import extract_key_concepts, calculate_similarity, clean_text

logger = logging.getLogger(__name__)


@dataclass
class CommonGround:
    """Represents areas of agreement between opposing positions."""
    shared_values: List[str]
    agreed_facts: List[str]
    compatible_goals: List[str]
    convergent_reasoning: List[str]
    similarity_score: float
    consensus_potential: float


@dataclass
class PerspectiveAnalysis:
    """Analysis of a specific perspective or argument."""
    position: str
    key_arguments: List[str]
    evidence_strength: float
    logical_coherence: float
    scope_limitations: List[str]
    unstated_assumptions: List[str]
    strongest_points: List[str]
    weakest_points: List[str]


@dataclass
class SynthesisInsight:
    """Synthesized insight combining multiple perspectives."""
    insight_type: str  # integration, transcendence, refinement, qualification
    content: str
    supporting_perspectives: List[str]
    confidence_level: float
    implications: List[str]
    action_items: List[str]


@dataclass
class SynthesisContext:
    """Context for synthesis operations."""
    original_question: str
    proponent_arguments: List[str]
    opponent_arguments: List[str]
    debate_history: List[Dict[str, Any]]
    common_ground: Optional[CommonGround] = None
    perspective_analyses: List[PerspectiveAnalysis] = field(default_factory=list)
    synthesis_insights: List[SynthesisInsight] = field(default_factory=list)
    synthesis_type: str = "comprehensive"
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class SynthesizerAgent(BaseAgent):
    """
    Agent specialized in synthesizing multiple perspectives and finding common ground.
    
    Analyzes opposing viewpoints to identify areas of agreement, integrate
    compatible insights, and generate nuanced conclusions that incorporate
    the strongest elements from different positions.
    """

    def __init__(self, **kwargs):
        # Default configuration optimized for synthesis
        default_config = AgentConfig(
            model="qwen2:1.5b",   # Balanced model for synthesis
            temperature=0.7,      # Balanced creativity and consistency
            max_tokens=1024,      # Room for detailed synthesis
            timeout=90,           # More time for complex reasoning
            enable_tools=False,   # Pure reasoning for synthesis
            enable_memory=True,
            memory_limit=30       # Remember extensive debate context
        )
        
        # Merge with provided config
        config = default_config.__dict__.copy()
        config.update(kwargs)
        final_config = AgentConfig(**config)
        
        super().__init__(final_config)
        self.agent_role = "synthesizer"
        self.synthesis_approaches = [
            "dialectical",  # Thesis + antithesis → synthesis
            "integrative",  # Combine compatible elements
            "transcendent", # Rise above the conflict to higher level
            "pluralistic",  # Accept multiple valid perspectives
            "pragmatic",    # Focus on practical outcomes
            "evidence_based" # Synthesize based on evidence quality
        ]

    async def analyze_perspective(
        self, 
        arguments: List[str], 
        position_label: str,
        context: SynthesisContext
    ) -> PerspectiveAnalysis:
        """
        Analyze a specific perspective (proponent or opponent) for synthesis.
        """
        try:
            # Combine all arguments for analysis
            combined_arguments = "\n".join(arguments)
            
            analysis_prompt = f"""
            You are an expert analyst examining a debate perspective for synthesis purposes.

            PERSPECTIVE TO ANALYZE: {position_label}
            ARGUMENTS:
            {combined_arguments}

            ORIGINAL QUESTION: {context.original_question}

            Provide a comprehensive analysis of this perspective including:

            1. POSITION SUMMARY:
               - What is the core position or thesis?
               - What are the main claims being made?

            2. KEY ARGUMENTS:
               - List the 3-5 strongest arguments presented
               - Identify the logical structure of each argument
               - Note the types of reasoning used (empirical, logical, ethical, etc.)

            3. EVIDENCE ASSESSMENT:
               - Evaluate the overall strength of evidence presented
               - Rate evidence quality on a scale of 1-10
               - Identify gaps in evidence or support

            4. LOGICAL COHERENCE:
               - Assess internal consistency of the arguments
               - Rate logical coherence on a scale of 1-10
               - Note any logical tensions or contradictions

            5. SCOPE AND LIMITATIONS:
               - What aspects of the question does this perspective address well?
               - What important aspects does it miss or underemphasize?
               - What are the boundaries of this perspective's applicability?

            6. ASSUMPTIONS:
               - What unstated assumptions underlie this perspective?
               - Which assumptions are reasonable vs questionable?
               - How do these assumptions affect the conclusions?

            7. STRENGTHS AND WEAKNESSES:
               - Identify the 2-3 strongest points in this perspective
               - Identify the 2-3 weakest or most vulnerable points
               - Consider both logical and empirical strengths/weaknesses

            Be objective and fair in your analysis. Look for genuine insights
            and valid points even if you might disagree with the overall position.
            
            Format your analysis with clear sections for each component.
            """

            response = await self._get_completion(analysis_prompt, context_key="perspective_analysis")
            
            if not response.success:
                logger.error(f"Failed to analyze perspective: {response.error}")
                return self._create_default_analysis(position_label)

            # Parse the analysis response
            analysis_text = response.content
            
            # Extract components (simplified parsing - would use more sophisticated NLP in production)
            key_arguments = self._extract_key_arguments(analysis_text)
            evidence_strength = self._extract_rating(analysis_text, "evidence")
            logical_coherence = self._extract_rating(analysis_text, "coherence")
            scope_limitations = self._extract_limitations(analysis_text)
            assumptions = self._extract_assumptions(analysis_text)
            strongest_points = self._extract_strongest_points(analysis_text)
            weakest_points = self._extract_weakest_points(analysis_text)

            return PerspectiveAnalysis(
                position=position_label,
                key_arguments=key_arguments,
                evidence_strength=evidence_strength,
                logical_coherence=logical_coherence,
                scope_limitations=scope_limitations,
                unstated_assumptions=assumptions,
                strongest_points=strongest_points,
                weakest_points=weakest_points
            )

        except Exception as e:
            logger.error(f"Error analyzing perspective: {e}")
            return self._create_default_analysis(position_label)

    async def identify_common_ground(
        self, 
        proponent_analysis: PerspectiveAnalysis,
        opponent_analysis: PerspectiveAnalysis,
        context: SynthesisContext
    ) -> CommonGround:
        """
        Identify areas of agreement and compatibility between opposing perspectives.
        """
        try:
            common_ground_prompt = f"""
            You are identifying common ground between opposing debate positions.

            PROPONENT POSITION: {proponent_analysis.position}
            Key Arguments: {', '.join(proponent_analysis.key_arguments)}
            Strongest Points: {', '.join(proponent_analysis.strongest_points)}

            OPPONENT POSITION: {opponent_analysis.position}  
            Key Arguments: {', '.join(opponent_analysis.key_arguments)}
            Strongest Points: {', '.join(opponent_analysis.strongest_points)}

            ORIGINAL QUESTION: {context.original_question}

            Identify areas where these opposing perspectives might find common ground:

            1. SHARED VALUES:
               - What underlying values do both sides seem to share?
               - What principles or goals do they both care about?
               - What concerns motivate both perspectives?

            2. AGREED FACTS:
               - What factual claims would both sides likely accept?
               - What empirical evidence is acknowledged by both?
               - What basic definitions or premises are shared?

            3. COMPATIBLE GOALS:
               - What outcomes would both sides potentially support?
               - What problems do both sides want to solve?
               - What improvements would both sides welcome?

            4. CONVERGENT REASONING:
               - Where do both sides use similar types of reasoning?
               - What logical principles do both sides rely on?
               - What methods of analysis do both sides accept?

            5. SYNTHESIS OPPORTUNITIES:
               - Where might these perspectives be integrated or combined?
               - What aspects are complementary rather than contradictory?
               - How might both sides contribute to a fuller understanding?

            6. CONSENSUS POTENTIAL:
               - On what specific points might both sides reach agreement?
               - What compromises or middle positions are possible?
               - What would it take for convergence on key issues?

            Be honest about areas where common ground may be limited,
            but look actively for genuine possibilities for synthesis.
            
            Format with clear sections for each type of common ground.
            """

            response = await self._get_completion(common_ground_prompt, context_key="common_ground")
            
            if not response.success:
                logger.error(f"Failed to identify common ground: {response.error}")
                return self._create_default_common_ground()

            # Parse common ground elements
            analysis_text = response.content
            shared_values = self._extract_shared_values(analysis_text)
            agreed_facts = self._extract_agreed_facts(analysis_text)
            compatible_goals = self._extract_compatible_goals(analysis_text)
            convergent_reasoning = self._extract_convergent_reasoning(analysis_text)
            
            # Calculate similarity and consensus scores
            similarity_score = self._calculate_perspective_similarity(proponent_analysis, opponent_analysis)
            consensus_potential = self._assess_consensus_potential(analysis_text, similarity_score)

            return CommonGround(
                shared_values=shared_values,
                agreed_facts=agreed_facts,
                compatible_goals=compatible_goals,
                convergent_reasoning=convergent_reasoning,
                similarity_score=similarity_score,
                consensus_potential=consensus_potential
            )

        except Exception as e:
            logger.error(f"Error identifying common ground: {e}")
            return self._create_default_common_ground()

    async def generate_synthesis_insights(
        self,
        context: SynthesisContext
    ) -> List[SynthesisInsight]:
        """
        Generate synthesis insights that integrate multiple perspectives.
        """
        try:
            insights = []
            
            # Generate different types of synthesis insights
            insight_types = ["integration", "transcendence", "refinement", "qualification"]
            
            for insight_type in insight_types:
                insight = await self._generate_insight_by_type(insight_type, context)
                if insight:
                    insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Error generating synthesis insights: {e}")
            return []

    async def _generate_insight_by_type(
        self, 
        insight_type: str, 
        context: SynthesisContext
    ) -> Optional[SynthesisInsight]:
        """Generate a specific type of synthesis insight."""
        try:
            # Build type-specific prompt
            if insight_type == "integration":
                approach = "Combine compatible elements from both perspectives into a coherent whole"
                focus = "How can the best parts of each perspective work together?"
            elif insight_type == "transcendence":
                approach = "Rise above the immediate conflict to find a higher-level perspective"
                focus = "What broader framework encompasses both perspectives?"
            elif insight_type == "refinement":
                approach = "Use insights from each side to refine and improve the other"
                focus = "How can each perspective help improve the other?"
            else:  # qualification
                approach = "Identify when and where each perspective applies best"
                focus = "Under what conditions is each perspective most valid?"

            insight_prompt = f"""
            Generate a {insight_type} synthesis insight for this debate.

            APPROACH: {approach}
            FOCUS: {focus}

            DEBATE CONTEXT:
            Question: {context.original_question}
            Proponent Arguments: {context.proponent_arguments}
            Opponent Arguments: {context.opponent_arguments}

            COMMON GROUND IDENTIFIED:
            {self._format_common_ground(context.common_ground) if context.common_ground else "Not yet identified"}

            PERSPECTIVE ANALYSES:
            {self._format_perspective_analyses(context.perspective_analyses)}

            Generate a synthesis insight that:

            1. INTEGRATES PERSPECTIVES:
               - Shows how different viewpoints can work together
               - Identifies complementary rather than contradictory elements
               - Creates a more complete understanding

            2. PROVIDES NEW UNDERSTANDING:
               - Offers insights that neither side alone could provide
               - Reveals aspects of the question not previously considered
               - Advances thinking beyond the original positions

            3. SUGGESTS PRACTICAL IMPLICATIONS:
               - What does this synthesis mean for action or decision-making?
               - How does this change how we should approach the question?
               - What new possibilities does this open up?

            4. IDENTIFIES SUPPORTING EVIDENCE:
               - What from each perspective supports this synthesis?
               - How do the different arguments contribute to this insight?
               - What confidence level is appropriate for this synthesis?

            Format as a clear synthesis insight with practical implications.
            """

            response = await self._get_completion(insight_prompt, context_key=f"synthesis_{insight_type}")
            
            if not response.success:
                logger.warning(f"Failed to generate {insight_type} insight: {response.error}")
                return None

            # Parse insight components
            content = response.content
            supporting_perspectives = self._extract_supporting_perspectives(content)
            confidence_level = self._assess_insight_confidence(content, context)
            implications = self._extract_implications(content)
            action_items = self._extract_action_items(content)

            return SynthesisInsight(
                insight_type=insight_type,
                content=content,
                supporting_perspectives=supporting_perspectives,
                confidence_level=confidence_level,
                implications=implications,
                action_items=action_items
            )

        except Exception as e:
            logger.error(f"Error generating {insight_type} insight: {e}")
            return None

    async def generate_comprehensive_synthesis(
        self,
        context: SynthesisContext
    ) -> AgentResponse:
        """
        Generate a comprehensive synthesis of the entire debate.
        """
        try:
            # Analyze both perspectives if not already done
            if not context.perspective_analyses:
                proponent_analysis = await self.analyze_perspective(
                    context.proponent_arguments, "Proponent", context
                )
                opponent_analysis = await self.analyze_perspective(
                    context.opponent_arguments, "Opponent", context
                )
                context.perspective_analyses = [proponent_analysis, opponent_analysis]

            # Identify common ground if not already done
            if not context.common_ground:
                context.common_ground = await self.identify_common_ground(
                    context.perspective_analyses[0], context.perspective_analyses[1], context
                )

            # Generate synthesis insights
            context.synthesis_insights = await self.generate_synthesis_insights(context)

            # Calculate quality metrics
            context.quality_metrics = self._calculate_synthesis_quality(context)

            synthesis_prompt = f"""
            You are generating a comprehensive synthesis of a multi-perspective debate.

            ORIGINAL QUESTION:
            {context.original_question}

            PERSPECTIVE SUMMARIES:
            {self._format_perspective_summaries(context.perspective_analyses)}

            COMMON GROUND IDENTIFIED:
            {self._format_common_ground(context.common_ground)}

            SYNTHESIS INSIGHTS:
            {self._format_synthesis_insights(context.synthesis_insights)}

            QUALITY METRICS:
            - Debate Quality Score: {context.quality_metrics.get('debate_quality', 0):.1f}/10
            - Synthesis Potential: {context.quality_metrics.get('synthesis_potential', 0):.1f}/10
            - Evidence Integration: {context.quality_metrics.get('evidence_integration', 0):.1f}/10

            Generate a comprehensive synthesis that:

            1. EXECUTIVE SUMMARY:
               - Provide a clear, concise summary of the synthesis
               - State the key insight or conclusion
               - Explain how this advances understanding of the question

            2. INTEGRATED ANALYSIS:
               - Show how different perspectives contribute to fuller understanding
               - Demonstrate where perspectives complement rather than contradict
               - Identify what each side contributes that the other lacks

            3. COMMON GROUND EXPLORATION:
               - Highlight areas of genuine agreement
               - Show shared values and compatible goals
               - Identify foundation for potential consensus

            4. SYNTHESIS INSIGHTS:
               - Present key insights that emerge from combining perspectives
               - Show how synthesis transcends or integrates the original positions
               - Offer new ways of understanding the question

            5. NUANCED CONCLUSIONS:
               - Avoid false balance - acknowledge where evidence clearly favors one side
               - Recognize complexity and avoid oversimplification
               - Present conclusions that honor the insights from all perspectives

            6. PRACTICAL IMPLICATIONS:
               - What does this synthesis mean for decision-making?
               - What actions or approaches does this suggest?
               - How should this inform future thinking about the question?

            7. AREAS FOR FURTHER EXPLORATION:
               - What questions remain unresolved?
               - Where might additional perspectives be valuable?
               - What further research or analysis would be helpful?

            SYNTHESIS PRINCIPLES:
            - Be intellectually honest about areas of genuine disagreement
            - Don't force artificial consensus where fundamental differences exist
            - Acknowledge uncertainty and complexity where appropriate
            - Focus on advancing understanding rather than declaring winners
            - Respect the insights and validity of different perspectives

            Format as a comprehensive synthesis document with clear sections.
            """

            response = await self._get_completion(synthesis_prompt, context_key="comprehensive_synthesis")
            
            if not response.success:
                return AgentResponse(
                    success=False,
                    content="Failed to generate comprehensive synthesis.",
                    error=response.error,
                    agent_type=self.agent_role
                )

            # Track performance metrics
            await self._track_performance("comprehensive_synthesis", {
                "synthesis_length": len(response.content),
                "perspectives_analyzed": len(context.perspective_analyses),
                "insights_generated": len(context.synthesis_insights),
                "common_ground_strength": context.common_ground.consensus_potential if context.common_ground else 0,
                "synthesis_quality": context.quality_metrics.get('overall_quality', 0)
            })

            return AgentResponse(
                success=True,
                content=response.content,
                agent_type=self.agent_role,
                metadata={
                    "synthesis_type": SynthesisType.COMPREHENSIVE.value,
                    "perspectives_analyzed": len(context.perspective_analyses),
                    "common_ground_strength": context.common_ground.consensus_potential if context.common_ground else 0,
                    "synthesis_insights": len(context.synthesis_insights),
                    "quality_metrics": context.quality_metrics,
                    "debate_phase": DebatePhase.SYNTHESIS.value,
                    "message_type": MessageType.SYNTHESIS.value
                }
            )

        except Exception as e:
            logger.error(f"Error generating comprehensive synthesis: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate comprehensive synthesis.",
                error=str(e),
                agent_type=self.agent_role
            )

    async def generate_quick_synthesis(
        self,
        proponent_arg: str,
        opponent_arg: str,
        question: str
    ) -> AgentResponse:
        """
        Generate a quick synthesis of two specific arguments.
        """
        try:
            quick_synthesis_prompt = f"""
            Provide a brief synthesis of these two opposing arguments.

            QUESTION: {question}

            PROPONENT ARGUMENT:
            {proponent_arg}

            OPPONENT ARGUMENT:
            {opponent_arg}

            Generate a concise synthesis that:

            1. IDENTIFIES KEY DIFFERENCES:
               - What are the main points of disagreement?
               - Where do the arguments directly conflict?

            2. FINDS COMMON ELEMENTS:
               - What do both arguments acknowledge or agree on?
               - What shared concerns or values are evident?

            3. SYNTHESIS INSIGHT:
               - How might both perspectives contribute to understanding?
               - What does combining these viewpoints reveal?

            4. BALANCED ASSESSMENT:
               - What are the strongest elements from each side?
               - Where might there be room for integration or compromise?

            Keep this synthesis concise but substantive - aim for clarity over comprehensiveness.
            """

            response = await self._get_completion(quick_synthesis_prompt, context_key="quick_synthesis")
            
            if not response.success:
                return AgentResponse(
                    success=False,
                    content="Failed to generate quick synthesis.",
                    error=response.error,
                    agent_type=self.agent_role
                )

            return AgentResponse(
                success=True,
                content=response.content,
                agent_type=self.agent_role,
                metadata={
                    "synthesis_type": SynthesisType.QUICK.value,
                    "arguments_synthesized": 2,
                    "message_type": MessageType.QUICK_SYNTHESIS.value
                }
            )

        except Exception as e:
            logger.error(f"Error generating quick synthesis: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate quick synthesis.",
                error=str(e),
                agent_type=self.agent_role
            )

    def _extract_key_arguments(self, analysis_text: str) -> List[str]:
        """Extract key arguments from analysis text."""
        arguments = []
        lines = analysis_text.split('\n')
        
        in_arguments_section = False
        for line in lines:
            if 'key arguments' in line.lower() or 'arguments' in line.lower():
                in_arguments_section = True
                continue
            elif in_arguments_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    argument = line.strip()[1:].strip()
                    if argument:
                        arguments.append(argument)
                elif any(section in line.lower() for section in ['evidence', 'coherence', 'scope']):
                    break
        
        return arguments[:5]  # Limit to top 5

    def _extract_rating(self, analysis_text: str, metric: str) -> float:
        """Extract rating scores from analysis text."""
        import re
        
        # Look for patterns like "evidence quality: 7/10" or "7 out of 10"
        patterns = [
            rf'{metric}[^0-9]*(\d+)[/\s]*10',
            rf'{metric}[^0-9]*(\d+)\s*out\s*of\s*10',
            rf'(\d+)[/\s]*10[^0-9]*{metric}'
        ]
        
        analysis_lower = analysis_text.lower()
        for pattern in patterns:
            match = re.search(pattern, analysis_lower)
            if match:
                return float(match.group(1))
        
        return 5.0  # Default middle score

    def _extract_limitations(self, analysis_text: str) -> List[str]:
        """Extract scope limitations from analysis text."""
        limitations = []
        lines = analysis_text.split('\n')
        
        in_limitations_section = False
        for line in lines:
            if 'limitation' in line.lower() or 'scope' in line.lower():
                in_limitations_section = True
                continue
            elif in_limitations_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    limitation = line.strip()[1:].strip()
                    if limitation:
                        limitations.append(limitation)
                elif any(section in line.lower() for section in ['assumption', 'strength', 'weakness']):
                    break
        
        return limitations[:3]  # Limit to top 3

    def _extract_assumptions(self, analysis_text: str) -> List[str]:
        """Extract assumptions from analysis text."""
        assumptions = []
        lines = analysis_text.split('\n')
        
        in_assumptions_section = False
        for line in lines:
            if 'assumption' in line.lower():
                in_assumptions_section = True
                continue
            elif in_assumptions_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    assumption = line.strip()[1:].strip()
                    if assumption:
                        assumptions.append(assumption)
                elif any(section in line.lower() for section in ['strength', 'weakness']):
                    break
        
        return assumptions[:3]  # Limit to top 3

    def _extract_strongest_points(self, analysis_text: str) -> List[str]:
        """Extract strongest points from analysis text."""
        points = []
        lines = analysis_text.split('\n')
        
        in_strengths_section = False
        for line in lines:
            if 'strongest' in line.lower() or 'strength' in line.lower():
                in_strengths_section = True
                continue
            elif in_strengths_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    point = line.strip()[1:].strip()
                    if point:
                        points.append(point)
                elif 'weak' in line.lower():
                    break
        
        return points[:3]  # Limit to top 3

    def _extract_weakest_points(self, analysis_text: str) -> List[str]:
        """Extract weakest points from analysis text."""
        points = []
        lines = analysis_text.split('\n')
        
        in_weaknesses_section = False
        for line in lines:
            if 'weakest' in line.lower() or 'weakness' in line.lower():
                in_weaknesses_section = True
                continue
            elif in_weaknesses_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    point = line.strip()[1:].strip()
                    if point:
                        points.append(point)
        
        return points[:3]  # Limit to top 3

    def _extract_shared_values(self, analysis_text: str) -> List[str]:
        """Extract shared values from common ground analysis."""
        return self._extract_section_items(analysis_text, 'shared values')

    def _extract_agreed_facts(self, analysis_text: str) -> List[str]:
        """Extract agreed facts from common ground analysis."""
        return self._extract_section_items(analysis_text, 'agreed facts')

    def _extract_compatible_goals(self, analysis_text: str) -> List[str]:
        """Extract compatible goals from common ground analysis."""
        return self._extract_section_items(analysis_text, 'compatible goals')

    def _extract_convergent_reasoning(self, analysis_text: str) -> List[str]:
        """Extract convergent reasoning from common ground analysis."""
        return self._extract_section_items(analysis_text, 'convergent reasoning')

    def _extract_section_items(self, text: str, section_name: str) -> List[str]:
        """Generic method to extract items from a named section."""
        items = []
        lines = text.split('\n')
        
        in_section = False
        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            elif in_section and line.strip():
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    item = line.strip()[1:].strip()
                    if item:
                        items.append(item)
                elif line.startswith(str(len(items) + 1)) and ':' in line:
                    # Handle numbered lists
                    item = line.split(':', 1)[1].strip()
                    if item:
                        items.append(item)
                elif any(keyword in line.lower() for keyword in ['synthesis', 'consensus', 'compatible', 'convergent']) and section_name.lower() not in line.lower():
                    break
        
        return items[:5]  # Limit to top 5

    def _calculate_perspective_similarity(
        self, 
        proponent: PerspectiveAnalysis, 
        opponent: PerspectiveAnalysis
    ) -> float:
        """Calculate similarity score between perspectives."""
        # Compare key concepts from strongest points
        proponent_concepts = set()
        opponent_concepts = set()
        
        for point in proponent.strongest_points:
            proponent_concepts.update(extract_key_concepts(point))
        
        for point in opponent.strongest_points:
            opponent_concepts.update(extract_key_concepts(point))
        
        if not proponent_concepts or not opponent_concepts:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(proponent_concepts.intersection(opponent_concepts))
        union = len(proponent_concepts.union(opponent_concepts))
        
        return intersection / union if union > 0 else 0.0

    def _assess_consensus_potential(self, analysis_text: str, similarity_score: float) -> float:
        """Assess potential for consensus based on analysis."""
        base_score = similarity_score * 5  # Convert to 0-5 scale
        
        # Bonus for explicit consensus indicators
        consensus_indicators = [
            'consensus possible', 'agreement potential', 'common ground',
            'shared concerns', 'compatible', 'convergence'
        ]
        
        analysis_lower = analysis_text.lower()
        for indicator in consensus_indicators:
            if indicator in analysis_lower:
                base_score += 1.0
        
        # Penalty for strong disagreement indicators
        disagreement_indicators = [
            'fundamental disagreement', 'irreconcilable', 'incompatible',
            'cannot agree', 'no common ground'
        ]
        
        for indicator in disagreement_indicators:
            if indicator in analysis_lower:
                base_score -= 2.0
        
        return max(0.0, min(10.0, base_score))

    def _extract_supporting_perspectives(self, content: str) -> List[str]:
        """Extract which perspectives support a synthesis insight."""
        perspectives = []
        content_lower = content.lower()
        
        if 'proponent' in content_lower or 'affirmative' in content_lower:
            perspectives.append("proponent")
        if 'opponent' in content_lower or 'opposition' in content_lower:
            perspectives.append("opponent")
        if 'both' in content_lower or 'all' in content_lower:
            perspectives = ["proponent", "opponent"]
        
        return perspectives

    def _assess_insight_confidence(self, content: str, context: SynthesisContext) -> float:
        """Assess confidence level of a synthesis insight."""
        base_confidence = 5.0
        
        # Adjust based on evidence quality
        if context.perspective_analyses:
            avg_evidence = sum(p.evidence_strength for p in context.perspective_analyses) / len(context.perspective_analyses)
            base_confidence += (avg_evidence - 5.0) * 0.5
        
        # Adjust based on common ground strength
        if context.common_ground:
            base_confidence += context.common_ground.consensus_potential * 0.3
        
        # Check for confidence indicators in the content
        confidence_indicators = {
            'strongly suggest': 8.0,
            'clearly indicate': 8.0,
            'likely': 6.0,
            'probably': 6.0,
            'possibly': 4.0,
            'might': 3.0,
            'uncertain': 2.0
        }
        
        content_lower = content.lower()
        for indicator, score in confidence_indicators.items():
            if indicator in content_lower:
                base_confidence = score
                break
        
        return min(10.0, max(1.0, base_confidence))

    def _extract_implications(self, content: str) -> List[str]:
        """Extract implications from synthesis content."""
        return self._extract_section_items(content, 'implications')

    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from synthesis content."""
        return self._extract_section_items(content, 'action')

    def _calculate_synthesis_quality(self, context: SynthesisContext) -> Dict[str, float]:
        """Calculate quality metrics for the synthesis."""
        metrics = {}
        
        # Debate quality based on perspective analyses
        if context.perspective_analyses:
            avg_evidence = sum(p.evidence_strength for p in context.perspective_analyses) / len(context.perspective_analyses)
            avg_coherence = sum(p.logical_coherence for p in context.perspective_analyses) / len(context.perspective_analyses)
            metrics['debate_quality'] = (avg_evidence + avg_coherence) / 2
        else:
            metrics['debate_quality'] = 0.0
        
        # Synthesis potential based on common ground
        if context.common_ground:
            metrics['synthesis_potential'] = context.common_ground.consensus_potential
        else:
            metrics['synthesis_potential'] = 0.0
        
        # Evidence integration
        if context.perspective_analyses:
            metrics['evidence_integration'] = min(
                sum(p.evidence_strength for p in context.perspective_analyses) / len(context.perspective_analyses),
                10.0
            )
        else:
            metrics['evidence_integration'] = 0.0
        
        # Overall quality
        metrics['overall_quality'] = (
            metrics['debate_quality'] * 0.4 +
            metrics['synthesis_potential'] * 0.3 +
            metrics['evidence_integration'] * 0.3
        )
        
        return metrics

    def _format_common_ground(self, common_ground: Optional[CommonGround]) -> str:
        """Format common ground for display."""
        if not common_ground:
            return "No common ground analysis available."
        
        return f"""
        Shared Values: {', '.join(common_ground.shared_values)}
        Agreed Facts: {', '.join(common_ground.agreed_facts)}
        Compatible Goals: {', '.join(common_ground.compatible_goals)}
        Convergent Reasoning: {', '.join(common_ground.convergent_reasoning)}
        Similarity Score: {common_ground.similarity_score:.2f}
        Consensus Potential: {common_ground.consensus_potential:.1f}/10
        """

    def _format_perspective_analyses(self, analyses: List[PerspectiveAnalysis]) -> str:
        """Format perspective analyses for display."""
        if not analyses:
            return "No perspective analyses available."
        
        formatted = []
        for analysis in analyses:
            formatted.append(f"""
            {analysis.position}:
            - Evidence Strength: {analysis.evidence_strength:.1f}/10
            - Logical Coherence: {analysis.logical_coherence:.1f}/10
            - Strongest Points: {', '.join(analysis.strongest_points)}
            - Key Limitations: {', '.join(analysis.scope_limitations)}
            """)
        
        return '\n'.join(formatted)

    def _format_perspective_summaries(self, analyses: List[PerspectiveAnalysis]) -> str:
        """Format brief perspective summaries."""
        if not analyses:
            return "No perspective analyses available."
        
        summaries = []
        for analysis in analyses:
            summaries.append(f"""
            {analysis.position} (Evidence: {analysis.evidence_strength:.1f}/10, Logic: {analysis.logical_coherence:.1f}/10):
            Key Arguments: {', '.join(analysis.key_arguments[:2])}
            Strongest Points: {', '.join(analysis.strongest_points[:2])}
            """)
        
        return '\n'.join(summaries)

    def _format_synthesis_insights(self, insights: List[SynthesisInsight]) -> str:
        """Format synthesis insights for display."""
        if not insights:
            return "No synthesis insights generated."
        
        formatted = []
        for insight in insights:
            formatted.append(f"""
            {insight.insight_type.title()} Insight (Confidence: {insight.confidence_level:.1f}/10):
            {insight.content[:200]}{'...' if len(insight.content) > 200 else ''}
            Supporting Perspectives: {', '.join(insight.supporting_perspectives)}
            """)
        
        return '\n'.join(formatted)

    def _create_default_analysis(self, position_label: str) -> PerspectiveAnalysis:
        """Create default analysis when processing fails."""
        return PerspectiveAnalysis(
            position=position_label,
            key_arguments=["Analysis unavailable"],
            evidence_strength=5.0,
            logical_coherence=5.0,
            scope_limitations=["Cannot assess"],
            unstated_assumptions=["Cannot identify"],
            strongest_points=["Unable to analyze"],
            weakest_points=["Unable to analyze"]
        )

    def _create_default_common_ground(self) -> CommonGround:
        """Create default common ground when identification fails."""
        return CommonGround(
            shared_values=["Unable to identify"],
            agreed_facts=["Unable to identify"],
            compatible_goals=["Unable to identify"],
            convergent_reasoning=["Unable to identify"],
            similarity_score=0.0,
            consensus_potential=0.0
        )

    async def cleanup(self):
        """Clean up agent resources."""
        await super().cleanup()
        logger.info(f"SynthesizerAgent {self.session_id} cleaned up successfully")

    def get_system_prompt(self) -> str:
        """Get the system prompt for the synthesizer agent."""
        return """You are a synthesizer agent in a structured debate. Your role is to:
        1. Find common ground between opposing arguments
        2. Identify shared values and compatible goals
        3. Synthesize different perspectives into a balanced view
        4. Help bridge differences and create consensus
        5. Provide nuanced analysis that incorporates multiple viewpoints
        
        Be objective, balanced, and focus on integration rather than taking sides."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request and generate a synthesis response."""
        try:
            if context is None:
                context = {}
            
            # Build the prompt for synthesis
            prompt = f"""
            Context: You need to synthesize the following debate positions.
            
            Debate Content: {request}
            
            Your task: Find common ground and create a balanced synthesis that incorporates
            the strongest elements from different perspectives. Look for shared values,
            compatible goals, and areas where different viewpoints can coexist.
            """
            
            # Get response from LLM
            response = await self._get_llm_response(prompt, context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=response,
                metadata={
                    "agent_type": "synthesizer",
                    "timestamp": datetime.now().isoformat(),
                    "context": context
                }
            )
            
        except Exception as e:
            logger.error(f"Error in synthesizer agent request processing: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="I encountered an error while generating the synthesis.",
                metadata={"error": str(e)}
            )
