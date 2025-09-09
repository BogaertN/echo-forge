import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from abc import ABC, abstractmethod

from .base_agent import BaseAgent, ConversationMessage, AgentResponse, AgentConfig
from ..models import DebatePhase, MessageType, SpecialistType
from ..utils import extract_key_concepts, calculate_similarity, clean_text

logger = logging.getLogger(__name__)


@dataclass
class SpecialistContext:
    """Context for specialist agent operations."""
    debate_question: str
    current_arguments: List[str]
    debate_phase: str
    specialist_request: str
    urgency: str = "normal"  # low, normal, high
    depth_required: str = "intermediate"  # surface, intermediate, deep
    fact_check_needed: bool = False


@dataclass
class SpecialistContribution:
    """Represents a contribution from a specialist agent."""
    specialist_type: str
    content: str
    confidence_level: float
    key_points: List[str]
    sources_referenced: List[str]
    limitations_noted: List[str]
    follow_up_questions: List[str]
    fact_check_results: Optional[Dict[str, Any]] = None


class BaseSpecialistAgent(BaseAgent, ABC):
    """
    Abstract base class for all specialist agents.
    
    Provides common functionality for domain-specific experts while
    requiring each specialist to implement their own domain knowledge
    and analysis methods.
    """

    def __init__(self, config: Optional[AgentConfig] = None, specialist_type: str = "general"):
        super().__init__(config)
        self.specialist_type = specialist_type
        self.agent_role = f"specialist_{specialist_type}"
        self.domain_keywords = []  # To be defined by subclasses
        self.expertise_areas = []  # To be defined by subclasses

    @abstractmethod
    async def provide_specialist_insight(
        self, 
        context: SpecialistContext
    ) -> SpecialistContribution:
        """Provide domain-specific insight on the debate topic."""
        pass

    @abstractmethod
    def assess_relevance(self, debate_content: str) -> float:
        """Assess how relevant this specialist is to the debate content."""
        pass

    async def fact_check_claims(self, claims: List[str], context: SpecialistContext) -> Dict[str, Any]:
        """Perform domain-specific fact-checking of claims."""
        try:
            fact_check_prompt = f"""
            As a {self.specialist_type} expert, fact-check these claims for accuracy:

            CLAIMS TO VERIFY:
            {chr(10).join(f"- {claim}" for claim in claims)}

            CONTEXT: {context.debate_question}

            For each claim, assess:
            1. ACCURACY: Is this claim factually correct? (True/False/Partially True/Unverifiable)
            2. EVIDENCE: What evidence supports or contradicts this claim?
            3. CONTEXT: What important context or nuance is missing?
            4. SOURCES: What reliable sources address this claim?

            Focus on your area of expertise ({self.specialist_type}). If a claim is 
            outside your domain, clearly state that.

            Provide clear, objective assessments without taking sides in the debate.
            """

            response = await self._get_completion(fact_check_prompt, context_key="fact_check")
            
            if response.success:
                return self._parse_fact_check_results(response.content, claims)
            else:
                return {"status": "failed", "error": response.error}

        except Exception as e:
            logger.error(f"Error in fact-checking: {e}")
            return {"status": "error", "error": str(e)}

    def _parse_fact_check_results(self, results_text: str, original_claims: List[str]) -> Dict[str, Any]:
        """Parse fact-check results from LLM response."""
        results = {
            "status": "completed",
            "claim_assessments": [],
            "overall_confidence": 0.7,
            "limitations": []
        }

        # Simple parsing - would use more sophisticated NLP in production
        lines = results_text.split('\n')
        current_claim = None
        current_assessment = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(claim[:20].lower() in line.lower() for claim in original_claims):
                if current_claim and current_assessment:
                    results["claim_assessments"].append(current_assessment)
                current_claim = line
                current_assessment = {"claim": current_claim, "accuracy": "unverified"}

            elif "accuracy:" in line.lower():
                accuracy = line.split(":", 1)[1].strip().lower()
                if "true" in accuracy and "false" not in accuracy:
                    current_assessment["accuracy"] = "true"
                elif "false" in accuracy:
                    current_assessment["accuracy"] = "false"
                elif "partial" in accuracy:
                    current_assessment["accuracy"] = "partially_true"
                else:
                    current_assessment["accuracy"] = "unverifiable"

            elif "evidence:" in line.lower():
                current_assessment["evidence"] = line.split(":", 1)[1].strip()

        # Don't forget the last assessment
        if current_claim and current_assessment:
            results["claim_assessments"].append(current_assessment)

        return results


class ScienceSpecialistAgent(BaseSpecialistAgent):
    """Specialist agent for scientific topics and empirical claims."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config, "science")
        self.domain_keywords = [
            "study", "research", "data", "experiment", "hypothesis", "theory",
            "evidence", "peer-reviewed", "statistics", "correlation", "causation",
            "methodology", "sample size", "control group", "bias", "replication"
        ]
        self.expertise_areas = [
            "research methodology", "statistical analysis", "peer review process",
            "scientific evidence evaluation", "experimental design", "data interpretation"
        ]

    async def provide_specialist_insight(self, context: SpecialistContext) -> SpecialistContribution:
        """Provide scientific perspective on the debate topic."""
        try:
            science_prompt = f"""
            As a scientific methodology expert, provide insight on this debate topic.

            DEBATE QUESTION: {context.debate_question}
            CURRENT ARGUMENTS: {chr(10).join(context.current_arguments)}

            Provide scientific perspective focusing on:

            1. EMPIRICAL EVIDENCE:
               - What scientific evidence exists on this topic?
               - What are the strongest studies or data sources?
               - Where is the evidence unclear or conflicting?

            2. METHODOLOGICAL CONSIDERATIONS:
               - What research methods are most appropriate for this question?
               - What are common methodological limitations in this area?
               - How should we interpret conflicting studies?

            3. SCIENTIFIC CONSENSUS:
               - Is there scientific consensus on any aspects of this topic?
               - Where do legitimate scientific disagreements exist?
               - What does the weight of evidence suggest?

            4. CRITICAL EVALUATION:
               - What claims being made are well-supported by evidence?
               - What claims lack sufficient scientific backing?
               - What are the limitations of current knowledge?

            Be objective and acknowledge uncertainty where it exists.
            Focus on what the science can and cannot tell us about this topic.
            """

            response = await self._get_completion(science_prompt, context_key="science_insight")
            
            if not response.success:
                return self._create_fallback_contribution("Failed to generate scientific insight")

            key_points = self._extract_key_points(response.content)
            limitations = self._extract_limitations(response.content)
            follow_ups = self._extract_follow_up_questions(response.content)

            return SpecialistContribution(
                specialist_type=self.specialist_type,
                content=response.content,
                confidence_level=0.8,
                key_points=key_points,
                sources_referenced=["peer-reviewed research", "scientific methodology"],
                limitations_noted=limitations,
                follow_up_questions=follow_ups
            )

        except Exception as e:
            logger.error(f"Error in science specialist insight: {e}")
            return self._create_fallback_contribution(f"Error: {str(e)}")

    def assess_relevance(self, debate_content: str) -> float:
        """Assess relevance based on scientific keywords and concepts."""
        content_lower = debate_content.lower()
        
        # Count domain keywords
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in content_lower)
        keyword_score = min(keyword_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Look for empirical claims
        empirical_indicators = ["research shows", "studies indicate", "data reveals", "statistics", "evidence"]
        empirical_count = sum(1 for indicator in empirical_indicators if indicator in content_lower)
        empirical_score = min(empirical_count / 3.0, 1.0)
        
        return (keyword_score + empirical_score) / 2


class EthicsSpecialistAgent(BaseSpecialistAgent):
    """Specialist agent for ethical considerations and moral reasoning."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config, "ethics")
        self.domain_keywords = [
            "ethical", "moral", "right", "wrong", "ought", "should", "values",
            "principles", "rights", "duties", "consequences", "harm", "benefit",
            "justice", "fairness", "equality", "autonomy", "dignity", "virtue"
        ]
        self.expertise_areas = [
            "consequentialist ethics", "deontological ethics", "virtue ethics",
            "applied ethics", "moral reasoning", "ethical frameworks"
        ]

    async def provide_specialist_insight(self, context: SpecialistContext) -> SpecialistContribution:
        """Provide ethical analysis of the debate topic."""
        try:
            ethics_prompt = f"""
            As an ethics expert, analyze the moral dimensions of this debate.

            DEBATE QUESTION: {context.debate_question}
            CURRENT ARGUMENTS: {chr(10).join(context.current_arguments)}

            Provide ethical analysis focusing on:

            1. MORAL FRAMEWORKS:
               - How would different ethical theories approach this issue?
               - What do consequentialist, deontological, and virtue ethics suggest?
               - Where do these frameworks agree or conflict?

            2. STAKEHOLDER ANALYSIS:
               - Who are the key stakeholders affected by this issue?
               - What are their rights, interests, and moral standing?
               - How should competing interests be balanced?

            3. MORAL PRINCIPLES:
               - What fundamental moral principles are at stake?
               - How do principles like autonomy, beneficence, justice apply?
               - Where do moral principles conflict in this case?

            4. ETHICAL CONSIDERATIONS:
               - What are the potential harms and benefits of different positions?
               - What are the moral implications of action vs. inaction?
               - What precedent would different choices set?

            Present multiple ethical perspectives fairly without taking a definitive stance.
            Help clarify the moral landscape rather than providing simple answers.
            """

            response = await self._get_completion(ethics_prompt, context_key="ethics_insight")
            
            if not response.success:
                return self._create_fallback_contribution("Failed to generate ethical insight")

            key_points = self._extract_key_points(response.content)
            limitations = ["Ethical analysis may vary based on cultural and philosophical perspectives"]
            follow_ups = self._extract_follow_up_questions(response.content)

            return SpecialistContribution(
                specialist_type=self.specialist_type,
                content=response.content,
                confidence_level=0.75,
                key_points=key_points,
                sources_referenced=["ethical frameworks", "moral philosophy"],
                limitations_noted=limitations,
                follow_up_questions=follow_ups
            )

        except Exception as e:
            logger.error(f"Error in ethics specialist insight: {e}")
            return self._create_fallback_contribution(f"Error: {str(e)}")

    def assess_relevance(self, debate_content: str) -> float:
        """Assess relevance based on ethical keywords and moral language."""
        content_lower = debate_content.lower()
        
        # Count domain keywords
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in content_lower)
        keyword_score = min(keyword_count / 8.0, 1.0)
        
        # Look for moral reasoning patterns
        moral_patterns = ["should", "ought", "right to", "wrong to", "moral", "ethical"]
        moral_count = sum(1 for pattern in moral_patterns if pattern in content_lower)
        moral_score = min(moral_count / 4.0, 1.0)
        
        return (keyword_score + moral_score) / 2


class EconomicsSpecialistAgent(BaseSpecialistAgent):
    """Specialist agent for economic analysis and market considerations."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config, "economics")
        self.domain_keywords = [
            "cost", "benefit", "price", "market", "economy", "economic", "financial",
            "supply", "demand", "efficiency", "externalities", "incentives", "policy",
            "growth", "inflation", "employment", "investment", "budget", "trade"
        ]
        self.expertise_areas = [
            "microeconomics", "macroeconomics", "cost-benefit analysis",
            "market mechanisms", "economic policy", "behavioral economics"
        ]

    async def provide_specialist_insight(self, context: SpecialistContext) -> SpecialistContribution:
        """Provide economic analysis of the debate topic."""
        try:
            economics_prompt = f"""
            As an economics expert, analyze the economic dimensions of this debate.

            DEBATE QUESTION: {context.debate_question}
            CURRENT ARGUMENTS: {chr(10).join(context.current_arguments)}

            Provide economic analysis focusing on:

            1. ECONOMIC MECHANISMS:
               - What market forces or economic principles apply?
               - How do supply and demand dynamics affect this issue?
               - What role do incentives play?

            2. COST-BENEFIT ANALYSIS:
               - What are the economic costs of different approaches?
               - What are the economic benefits?
               - How should we value intangible costs and benefits?

            3. EXTERNALITIES AND EFFECTS:
               - What are the broader economic impacts?
               - Who bears the costs and who receives the benefits?
               - What unintended economic consequences might occur?

            4. POLICY IMPLICATIONS:
               - What do economic theories suggest about policy options?
               - What are the trade-offs between efficiency and equity?
               - What does economic evidence say about similar policies?

            Focus on economic analysis while acknowledging that economics is one of many 
            relevant perspectives on complex issues.
            """

            response = await self._get_completion(economics_prompt, context_key="economics_insight")
            
            if not response.success:
                return self._create_fallback_contribution("Failed to generate economic insight")

            key_points = self._extract_key_points(response.content)
            limitations = ["Economic analysis may not capture all social and ethical considerations"]
            follow_ups = self._extract_follow_up_questions(response.content)

            return SpecialistContribution(
                specialist_type=self.specialist_type,
                content=response.content,
                confidence_level=0.8,
                key_points=key_points,
                sources_referenced=["economic theory", "empirical economic research"],
                limitations_noted=limitations,
                follow_up_questions=follow_ups
            )

        except Exception as e:
            logger.error(f"Error in economics specialist insight: {e}")
            return self._create_fallback_contribution(f"Error: {str(e)}")

    def assess_relevance(self, debate_content: str) -> float:
        """Assess relevance based on economic keywords and concepts."""
        content_lower = debate_content.lower()
        
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in content_lower)
        keyword_score = min(keyword_count / 6.0, 1.0)
        
        # Look for economic reasoning
        economic_indicators = ["cost", "benefit", "market", "economic", "financial", "money"]
        economic_count = sum(1 for indicator in economic_indicators if indicator in content_lower)
        economic_score = min(economic_count / 3.0, 1.0)
        
        return (keyword_score + economic_score) / 2


class HistorySpecialistAgent(BaseSpecialistAgent):
    """Specialist agent for historical context and precedents."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config, "history")
        self.domain_keywords = [
            "history", "historical", "past", "precedent", "tradition", "evolution",
            "originally", "previously", "centuries", "decades", "era", "period",
            "lessons", "patterns", "change", "continuity", "context"
        ]
        self.expertise_areas = [
            "historical analysis", "historical precedents", "social change patterns",
            "institutional history", "comparative history", "historical methodology"
        ]

    async def provide_specialist_insight(self, context: SpecialistContext) -> SpecialistContribution:
        """Provide historical perspective on the debate topic."""
        try:
            history_prompt = f"""
            As a historian, provide historical context and perspective on this debate.

            DEBATE QUESTION: {context.debate_question}
            CURRENT ARGUMENTS: {chr(10).join(context.current_arguments)}

            Provide historical analysis focusing on:

            1. HISTORICAL PRECEDENTS:
               - How have similar issues been addressed in the past?
               - What can we learn from historical examples?
               - What patterns emerge across different times and places?

            2. HISTORICAL DEVELOPMENT:
               - How has thinking on this issue evolved over time?
               - What historical forces shaped current perspectives?
               - How have institutions and practices changed?

            3. LESSONS FROM HISTORY:
               - What do historical cases teach us about potential outcomes?
               - What unintended consequences have occurred in similar situations?
               - What factors led to success or failure in the past?

            4. CONTEXTUAL UNDERSTANDING:
               - What historical context is important for understanding this issue?
               - How do current circumstances differ from historical ones?
               - What continuities and changes are relevant?

            Focus on providing historical perspective while acknowledging that
            historical lessons must be applied carefully to contemporary contexts.
            """

            response = await self._get_completion(history_prompt, context_key="history_insight")
            
            if not response.success:
                return self._create_fallback_contribution("Failed to generate historical insight")

            key_points = self._extract_key_points(response.content)
            limitations = ["Historical analogies have limitations when applied to contemporary contexts"]
            follow_ups = self._extract_follow_up_questions(response.content)

            return SpecialistContribution(
                specialist_type=self.specialist_type,
                content=response.content,
                confidence_level=0.75,
                key_points=key_points,
                sources_referenced=["historical records", "historical analysis"],
                limitations_noted=limitations,
                follow_up_questions=follow_ups
            )

        except Exception as e:
            logger.error(f"Error in history specialist insight: {e}")
            return self._create_fallback_contribution(f"Error: {str(e)}")

    def assess_relevance(self, debate_content: str) -> float:
        """Assess relevance based on historical keywords and temporal references."""
        content_lower = debate_content.lower()
        
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in content_lower)
        keyword_score = min(keyword_count / 5.0, 1.0)
        
        # Look for temporal and historical references
        temporal_indicators = ["history", "past", "before", "previously", "tradition", "precedent"]
        temporal_count = sum(1 for indicator in temporal_indicators if indicator in content_lower)
        temporal_score = min(temporal_count / 3.0, 1.0)
        
        return (keyword_score + temporal_score) / 2


class LegalSpecialistAgent(BaseSpecialistAgent):
    """Specialist agent for legal analysis and jurisprudence."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config, "legal")
        self.domain_keywords = [
            "legal", "law", "court", "constitution", "rights", "statute", "regulation",
            "precedent", "jurisprudence", "jurisdiction", "liability", "contract",
            "due process", "constitutional", "legislation", "judicial", "enforcement"
        ]
        self.expertise_areas = [
            "constitutional law", "statutory interpretation", "legal precedents",
            "jurisprudence", "legal reasoning", "rights analysis"
        ]

    async def provide_specialist_insight(self, context: SpecialistContext) -> SpecialistContribution:
        """Provide legal analysis of the debate topic."""
        try:
            legal_prompt = f"""
            As a legal expert, analyze the legal dimensions of this debate.

            DEBATE QUESTION: {context.debate_question}
            CURRENT ARGUMENTS: {chr(10).join(context.current_arguments)}

            Provide legal analysis focusing on:

            1. LEGAL FRAMEWORKS:
               - What legal principles and frameworks apply?
               - What constitutional considerations are relevant?
               - How do different areas of law interact on this issue?

            2. RIGHTS AND OBLIGATIONS:
               - What legal rights are at stake?
               - What legal obligations exist?
               - How should competing rights be balanced?

            3. PRECEDENTS AND JURISPRUDENCE:
               - What legal precedents are relevant?
               - How have courts approached similar issues?
               - What trends exist in legal interpretation?

            4. LEGAL IMPLICATIONS:
               - What would be the legal consequences of different approaches?
               - What enforcement challenges might arise?
               - How might this affect future legal development?

            Focus on legal analysis while noting that legal considerations are one
            part of broader policy and ethical discussions.
            """

            response = await self._get_completion(legal_prompt, context_key="legal_insight")
            
            if not response.success:
                return self._create_fallback_contribution("Failed to generate legal insight")

            key_points = self._extract_key_points(response.content)
            limitations = ["Legal analysis may vary by jurisdiction and evolving jurisprudence"]
            follow_ups = self._extract_follow_up_questions(response.content)

            return SpecialistContribution(
                specialist_type=self.specialist_type,
                content=response.content,
                confidence_level=0.8,
                key_points=key_points,
                sources_referenced=["legal precedents", "constitutional law", "statutory analysis"],
                limitations_noted=limitations,
                follow_up_questions=follow_ups
            )

        except Exception as e:
            logger.error(f"Error in legal specialist insight: {e}")
            return self._create_fallback_contribution(f"Error: {str(e)}")

    def assess_relevance(self, debate_content: str) -> float:
        """Assess relevance based on legal keywords and concepts."""
        content_lower = debate_content.lower()
        
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in content_lower)
        keyword_score = min(keyword_count / 6.0, 1.0)
        
        # Look for legal reasoning and concepts
        legal_indicators = ["legal", "law", "rights", "constitutional", "court", "precedent"]
        legal_count = sum(1 for indicator in legal_indicators if indicator in content_lower)
        legal_score = min(legal_count / 3.0, 1.0)
        
        return (keyword_score + legal_score) / 2


class SpecialistManager:
    """
    Manages the collection of specialist agents and routes requests appropriately.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config
        self.specialists = {
            "science": ScienceSpecialistAgent(config),
            "ethics": EthicsSpecialistAgent(config),
            "economics": EconomicsSpecialistAgent(config),
            "history": HistorySpecialistAgent(config),
            "legal": LegalSpecialistAgent(config)
        }

    async def select_relevant_specialists(
        self, 
        debate_content: str, 
        max_specialists: int = 2
    ) -> List[BaseSpecialistAgent]:
        """
        Select the most relevant specialists based on debate content.
        """
        relevance_scores = {}
        
        for specialist_type, specialist in self.specialists.items():
            relevance = specialist.assess_relevance(debate_content)
            relevance_scores[specialist_type] = relevance
        
        # Sort by relevance and return top specialists
        sorted_specialists = sorted(
            relevance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected = []
        for specialist_type, score in sorted_specialists[:max_specialists]:
            if score > 0.3:  # Minimum relevance threshold
                selected.append(self.specialists[specialist_type])
        
        return selected

    async def get_specialist_insights(
        self, 
        context: SpecialistContext,
        requested_specialists: Optional[List[str]] = None
    ) -> List[SpecialistContribution]:
        """
        Get insights from requested specialists or auto-select based on content.
        """
        insights = []
        
        if requested_specialists:
            # Use specifically requested specialists
            specialists_to_consult = [
                self.specialists[spec_type] 
                for spec_type in requested_specialists 
                if spec_type in self.specialists
            ]
        else:
            # Auto-select based on content
            debate_content = f"{context.debate_question} {' '.join(context.current_arguments)}"
            specialists_to_consult = await self.select_relevant_specialists(debate_content)
        
        # Get insights from selected specialists
        for specialist in specialists_to_consult:
            try:
                insight = await specialist.provide_specialist_insight(context)
                insights.append(insight)
            except Exception as e:
                logger.error(f"Error getting insight from {specialist.specialist_type}: {e}")
        
        return insights

    async def fact_check_with_specialists(
        self, 
        claims: List[str], 
        context: SpecialistContext
    ) -> Dict[str, Any]:
        """
        Perform fact-checking using relevant specialists.
        """
        all_results = {}
        
        # Select specialists for fact-checking
        debate_content = f"{context.debate_question} {' '.join(claims)}"
        relevant_specialists = await self.select_relevant_specialists(debate_content, max_specialists=3)
        
        for specialist in relevant_specialists:
            try:
                specialist_results = await specialist.fact_check_claims(claims, context)
                all_results[specialist.specialist_type] = specialist_results
            except Exception as e:
                logger.error(f"Error in {specialist.specialist_type} fact-checking: {e}")
                all_results[specialist.specialist_type] = {"status": "error", "error": str(e)}
        
        return all_results

    async def generate_specialist_summary(
        self, 
        contributions: List[SpecialistContribution],
        context: SpecialistContext
    ) -> str:
        """
        Generate a summary synthesizing multiple specialist contributions.
        """
        try:
            if not contributions:
                return "No specialist insights were available for this topic."
            
            summary_content = []
            
            for contribution in contributions:
                specialist_section = f"""
## {contribution.specialist_type.title()} Perspective

{contribution.content[:500]}{'...' if len(contribution.content) > 500 else ''}

**Key Points:**
{chr(10).join(f"• {point}" for point in contribution.key_points[:3])}
"""
                summary_content.append(specialist_section)
            
            # Add synthesis if multiple specialists
            if len(contributions) > 1:
                summary_content.append("""
## Synthesis

The specialist perspectives above offer different lenses for understanding this issue. Consider how insights from different domains complement or tension with each other when forming your own views.
""")
            
            return "\n".join(summary_content)

        except Exception as e:
            logger.error(f"Error generating specialist summary: {e}")
            return "Error generating specialist summary."

    async def cleanup(self):
        """Clean up all specialist agents."""
        for specialist in self.specialists.values():
            await specialist.cleanup()


# Shared methods for BaseSpecialistAgent subclasses
def _extract_key_points(content: str) -> List[str]:
    """Extract key points from specialist content."""
    points = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
            point = line[1:].strip()
            if point:
                points.append(point)
    
    return points[:5]  # Limit to top 5


def _extract_limitations(content: str) -> List[str]:
    """Extract noted limitations from specialist content."""
    limitations = []
    lines = content.split('\n')
    
    in_limitations_section = False
    for line in lines:
        line = line.strip()
        if 'limitation' in line.lower() or 'caveat' in line.lower():
            in_limitations_section = True
        elif in_limitations_section and line:
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                limitation = line[1:].strip()
                if limitation:
                    limitations.append(limitation)
    
    return limitations[:3]  # Limit to top 3


def _extract_follow_up_questions(content: str) -> List[str]:
    """Extract follow-up questions from specialist content."""
    questions = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.endswith('?'):
            questions.append(line)
    
    return questions[:3]  # Limit to top 3


# Add shared methods to BaseSpecialistAgent
BaseSpecialistAgent._extract_key_points = lambda self, content: _extract_key_points(content)
BaseSpecialistAgent._extract_limitations = lambda self, content: _extract_limitations(content)
BaseSpecialistAgent._extract_follow_up_questions = lambda self, content: _extract_follow_up_questions(content)


def _create_fallback_contribution(self, error_message: str) -> SpecialistContribution:
    """Create a fallback contribution when specialist insight fails."""
    return SpecialistContribution(
        specialist_type=self.specialist_type,
        content=f"I apologize, but I'm unable to provide detailed {self.specialist_type} insight at this time. {error_message}",
        confidence_level=0.1,
        key_points=["Analysis unavailable"],
        sources_referenced=[],
        limitations_noted=["Technical error in analysis"],
        follow_up_questions=[]
    )


BaseSpecialistAgent._create_fallback_contribution = _create_fallback_contribution
