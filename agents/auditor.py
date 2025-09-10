#!/usr/bin/env python3
"""
EchoForge Auditor Agent
=======================
Specialized agent that audits debate quality, detects logical fallacies, and provides meta-analysis of reasoning.

Features:
- Comprehensive logical fallacy detection
- Debate quality assessment and scoring
- Evidence evaluation and fact-checking
- Reasoning pattern analysis
- Meta-cognitive insights
- Quality improvement recommendations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

from agents.base_agent import BaseAgent, ConversationMessage, AgentResponse, AgentConfig
from models import DebatePhase, MessageType, AuditSeverity
from utils import extract_key_concepts, calculate_similarity, clean_text

logger = logging.getLogger(__name__)


@dataclass
class LogicalFallacy:
    """Represents a detected logical fallacy."""
    fallacy_type: str
    description: str
    location: str  # Where in the text it was found
    severity: str  # low, medium, high
    explanation: str
    confidence_score: float
    suggested_correction: str


@dataclass
class EvidenceAssessment:
    """Assessment of evidence quality and reliability."""
    claim: str
    evidence_type: str  # empirical, anecdotal, statistical, expert_opinion, etc.
    reliability_score: float
    relevance_score: float
    sufficiency_score: float
    source_quality: str
    potential_biases: List[str]
    verification_status: str


@dataclass
class ReasoningPattern:
    """Analysis of reasoning patterns used in arguments."""
    pattern_type: str  # deductive, inductive, abductive, analogical, etc.
    structure_quality: float
    logical_validity: float
    premise_strength: float
    conclusion_support: float
    identified_gaps: List[str]


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for debate content."""
    overall_score: float
    logical_coherence: float
    evidence_quality: float
    reasoning_clarity: float
    fallacy_count: int
    argument_strength: float
    persuasiveness: float
    intellectual_honesty: float


@dataclass
class AuditReport:
    """Comprehensive audit report for debate content."""
    content_analyzed: str
    timestamp: datetime
    fallacies_detected: List[LogicalFallacy]
    evidence_assessments: List[EvidenceAssessment]
    reasoning_patterns: List[ReasoningPattern]
    quality_metrics: QualityMetrics
    recommendations: List[str]
    strengths: List[str]
    improvement_areas: List[str]
    confidence_score: float


@dataclass
class AuditContext:
    """Context for audit operations."""
    session_id: str
    debate_question: str
    debate_phase: str
    participant_role: str
    previous_audits: List[AuditReport] = field(default_factory=list)
    audit_focus: str = "comprehensive"  # comprehensive, fallacy_focused, evidence_focused
    severity_threshold: str = "medium"


class AuditorAgent(BaseAgent):
    """
    Agent specialized in auditing debate quality and detecting reasoning flaws.
    
    Performs comprehensive analysis of arguments to identify logical fallacies,
    assess evidence quality, evaluate reasoning patterns, and provide detailed
    feedback for improving debate quality and critical thinking.
    """

    def __init__(self, **kwargs):
        # Default configuration optimized for auditing
        default_config = AgentConfig(
            model="tinyllama:1.1b",  # Lightweight model for quick checks
            temperature=0.5,         # Lower temperature for consistency
            max_tokens=512,          # Shorter responses for auditing
            timeout=60,              # Quick turnaround for audits
            enable_tools=False,      # Pure reasoning for auditing
            enable_memory=True,
            memory_limit=20          # Focused audit context
        )
        
        # Merge with provided config
        config = default_config.__dict__.copy()
        config.update(kwargs)
        final_config = AgentConfig(**config)
        
        super().__init__(final_config)
        self.agent_role = "auditor"
        
        # Comprehensive fallacy taxonomy
        self.fallacy_patterns = {
            "ad_hominem": {
                "keywords": ["stupid", "idiot", "moron", "foolish", "ignorant"],
                "patterns": [r"you are (stupid|wrong|ignorant)", r"only an? (idiot|fool) would"],
                "description": "Attacking the person rather than their argument"
            },
            "straw_man": {
                "keywords": ["claims that", "believes that", "argues that"],
                "patterns": [r"so you're saying", r"what you really mean is"],
                "description": "Misrepresenting someone's argument to make it easier to attack"
            },
            "false_dichotomy": {
                "keywords": ["either", "only two", "must choose"],
                "patterns": [r"either .+ or .+", r"only two (options|choices|ways)"],
                "description": "Presenting only two options when more exist"
            },
            "slippery_slope": {
                "keywords": ["will lead to", "inevitably", "next thing"],
                "patterns": [r"if .+ then .+ will", r"leads inevitably to"],
                "description": "Assuming one event will lead to extreme consequences"
            },
            "appeal_to_authority": {
                "keywords": ["expert says", "authority", "scientist claims"],
                "patterns": [r"dr\.? .+ says", r"according to experts"],
                "description": "Using authority as evidence without proper justification"
            },
            "appeal_to_emotion": {
                "keywords": ["think of the children", "heartbreaking", "tragic"],
                "patterns": [r"think of (the children|your family)", r"how can you"],
                "description": "Using emotions rather than logic to persuade"
            },
            "hasty_generalization": {
                "keywords": ["all", "always", "never", "every"],
                "patterns": [r"all .+ are", r"every .+ does"],
                "description": "Drawing broad conclusions from limited examples"
            },
            "circular_reasoning": {
                "keywords": ["because", "proves", "shows"],
                "patterns": [r"(.+) because \1", r"proves .+ by assuming"],
                "description": "Using the conclusion as evidence for itself"
            },
            "post_hoc": {
                "keywords": ["after", "therefore", "caused by"],
                "patterns": [r"after .+ therefore", r"since .+ happened, .+ caused"],
                "description": "Assuming correlation implies causation"
            },
            "bandwagon": {
                "keywords": ["everyone", "popular", "most people"],
                "patterns": [r"everyone (thinks|believes|knows)", r"most people agree"],
                "description": "Arguing something is true because it's popular"
            },
            "red_herring": {
                "keywords": ["but what about", "real issue", "more important"],
                "patterns": [r"but what about", r"the real issue is"],
                "description": "Diverting attention from the main argument"
            },
            "no_true_scotsman": {
                "keywords": ["real", "true", "genuine"],
                "patterns": [r"no (real|true) .+ would", r"that's not a (real|true)"],
                "description": "Redefining terms to exclude counterexamples"
            }
        }
        
        # Evidence quality indicators
        self.evidence_indicators = {
            "strong": ["peer-reviewed", "meta-analysis", "randomized controlled trial", "systematic review"],
            "moderate": ["published study", "research shows", "data indicates", "survey found"],
            "weak": ["some say", "people think", "it's believed", "common knowledge"],
            "anecdotal": ["I know someone", "personal experience", "my friend", "I heard"]
        }

    async def audit_argument(
        self, 
        argument: str, 
        context: AuditContext
    ) -> AuditReport:
        """
        Perform comprehensive audit of a single argument.
        """
        try:
            logger.info(f"Starting comprehensive audit for session {context.session_id}")
            
            # Initialize audit components
            fallacies = await self._detect_fallacies(argument, context)
            evidence_assessments = await self._assess_evidence(argument, context)
            reasoning_patterns = await self._analyze_reasoning_patterns(argument, context)
            quality_metrics = await self._calculate_quality_metrics(
                argument, fallacies, evidence_assessments, reasoning_patterns
            )
            recommendations = await self._generate_recommendations(
                argument, fallacies, evidence_assessments, reasoning_patterns, context
            )
            strengths, improvement_areas = await self._identify_strengths_and_improvements(
                argument, quality_metrics, fallacies
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_audit_confidence(
                fallacies, evidence_assessments, reasoning_patterns
            )

            report = AuditReport(
                content_analyzed=argument,
                timestamp=datetime.now(),
                fallacies_detected=fallacies,
                evidence_assessments=evidence_assessments,
                reasoning_patterns=reasoning_patterns,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                strengths=strengths,
                improvement_areas=improvement_areas,
                confidence_score=confidence_score
            )

            # Store audit for historical analysis
            context.previous_audits.append(report)
            
            # Track performance metrics
            await self._track_performance("argument_audit", {
                "argument_length": len(argument),
                "fallacies_detected": len(fallacies),
                "evidence_items_assessed": len(evidence_assessments),
                "overall_quality": quality_metrics.overall_score,
                "audit_confidence": confidence_score
            })

            return report

        except Exception as e:
            logger.error(f"Error during argument audit: {e}")
            return self._create_default_audit_report(argument)

    async def _detect_fallacies(
        self, 
        argument: str, 
        context: AuditContext
    ) -> List[LogicalFallacy]:
        """
        Detect logical fallacies in the argument.
        """
        try:
            fallacies = []
            
            # Pattern-based detection for quick identification
            pattern_fallacies = self._detect_pattern_fallacies(argument)
            fallacies.extend(pattern_fallacies)
            
            # LLM-based detection for sophisticated analysis
            llm_fallacies = await self._detect_llm_fallacies(argument, context)
            fallacies.extend(llm_fallacies)
            
            # Remove duplicates and rank by confidence
            fallacies = self._deduplicate_and_rank_fallacies(fallacies)
            
            return fallacies

        except Exception as e:
            logger.error(f"Error detecting fallacies: {e}")
            return []

    def _detect_pattern_fallacies(self, argument: str) -> List[LogicalFallacy]:
        """
        Use pattern matching to detect obvious fallacies.
        """
        fallacies = []
        argument_lower = argument.lower()
        
        for fallacy_type, patterns in self.fallacy_patterns.items():
            # Check keywords
            keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in argument_lower)
            
            # Check regex patterns
            pattern_matches = 0
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, argument_lower):
                    pattern_matches += 1
            
            # Calculate confidence based on matches
            total_indicators = len(patterns["keywords"]) + len(patterns.get("patterns", []))
            total_matches = keyword_matches + pattern_matches
            
            if total_matches > 0:
                confidence = min(total_matches / total_indicators * 2, 1.0)  # Scale to 0-1
                
                if confidence > 0.3:  # Threshold for reporting
                    fallacy = LogicalFallacy(
                        fallacy_type=fallacy_type,
                        description=patterns["description"],
                        location=f"Pattern detected in text",
                        severity=self._determine_severity(confidence),
                        explanation=f"Detected through pattern matching: {patterns['description']}",
                        confidence_score=confidence,
                        suggested_correction=f"Consider revising to avoid {fallacy_type.replace('_', ' ')}"
                    )
                    fallacies.append(fallacy)
        
        return fallacies

    async def _detect_llm_fallacies(
        self, 
        argument: str, 
        context: AuditContext
    ) -> List[LogicalFallacy]:
        """
        Use LLM to detect sophisticated logical fallacies.
        """
        try:
            fallacy_prompt = f"""
            You are an expert logician analyzing an argument for logical fallacies.

            ARGUMENT TO ANALYZE:
            {argument}

            CONTEXT:
            - Debate Question: {context.debate_question}
            - Debate Phase: {context.debate_phase}
            - Participant Role: {context.participant_role}

            Please identify any logical fallacies in this argument. For each fallacy found:

            1. FALLACY IDENTIFICATION:
               - Name the specific type of logical fallacy
               - Quote the exact text where it occurs
               - Rate your confidence (1-10) in this identification

            2. EXPLANATION:
               - Explain why this constitutes a logical fallacy
               - Describe how it undermines the argument
               - Show what valid reasoning would look like instead

            3. SEVERITY ASSESSMENT:
               - Rate severity as low/medium/high
               - Consider impact on argument validity
               - Consider whether it's central or peripheral to the main point

            4. CORRECTION SUGGESTIONS:
               - Provide specific suggestions for improvement
               - Show how the argument could be restructured
               - Maintain the intent while fixing the logical flaw

            FOCUS AREAS:
            - Formal logical fallacies (invalid inference patterns)
            - Informal fallacies (irrelevant or misleading arguments)
            - Evidential fallacies (misuse of evidence or sources)
            - Causal fallacies (incorrect cause-effect reasoning)

            Be thorough but fair. Don't force fallacy detection where none exist.
            Only identify fallacies you're confident about (6+ out of 10 confidence).
            
            Format your response with clear sections for each fallacy found.
            If no significant fallacies are detected, state this clearly.
            """

            response = await self._get_completion(fallacy_prompt, context_key="fallacy_detection")
            
            if not response.success:
                logger.warning(f"LLM fallacy detection failed: {response.error}")
                return []

            # Parse LLM response to extract fallacies
            return self._parse_llm_fallacies(response.content)

        except Exception as e:
            logger.error(f"Error in LLM fallacy detection: {e}")
            return []

    def _parse_llm_fallacies(self, llm_response: str) -> List[LogicalFallacy]:
        """
        Parse LLM response to extract structured fallacy information.
        """
        fallacies = []
        
        # This is a simplified parser - in production would use more sophisticated NLP
        lines = llm_response.split('\n')
        current_fallacy = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for fallacy type indicators
            if any(indicator in line.lower() for indicator in ['fallacy:', 'fallacy type:', 'type:']):
                if current_fallacy:
                    fallacy = self._create_fallacy_from_parsed_data(current_fallacy)
                    if fallacy:
                        fallacies.append(fallacy)
                    current_fallacy = {}
                
                fallacy_type = line.split(':', 1)[1].strip() if ':' in line else line
                current_fallacy['type'] = fallacy_type.lower().replace(' ', '_')
            
            elif 'confidence:' in line.lower():
                try:
                    confidence_text = line.split(':', 1)[1].strip()
                    confidence = float(re.search(r'(\d+)', confidence_text).group(1)) / 10.0
                    current_fallacy['confidence'] = confidence
                except:
                    current_fallacy['confidence'] = 0.5
            
            elif 'severity:' in line.lower():
                severity = line.split(':', 1)[1].strip().lower()
                current_fallacy['severity'] = severity if severity in ['low', 'medium', 'high'] else 'medium'
            
            elif 'explanation:' in line.lower():
                current_fallacy['explanation'] = line.split(':', 1)[1].strip()
            
            elif 'suggestion:' in line.lower() or 'correction:' in line.lower():
                current_fallacy['suggestion'] = line.split(':', 1)[1].strip()
        
        # Don't forget the last fallacy
        if current_fallacy:
            fallacy = self._create_fallacy_from_parsed_data(current_fallacy)
            if fallacy:
                fallacies.append(fallacy)
        
        return fallacies

    def _create_fallacy_from_parsed_data(self, data: Dict[str, Any]) -> Optional[LogicalFallacy]:
        """
        Create LogicalFallacy object from parsed data.
        """
        if 'type' not in data:
            return None
        
        return LogicalFallacy(
            fallacy_type=data.get('type', 'unknown'),
            description=data.get('explanation', 'No description provided'),
            location="Identified by LLM analysis",
            severity=data.get('severity', 'medium'),
            explanation=data.get('explanation', 'No explanation provided'),
            confidence_score=data.get('confidence', 0.5),
            suggested_correction=data.get('suggestion', 'Consider revising this section')
        )

    async def _assess_evidence(
        self, 
        argument: str, 
        context: AuditContext
    ) -> List[EvidenceAssessment]:
        """
        Assess quality and reliability of evidence presented.
        """
        try:
            evidence_prompt = f"""
            You are evaluating the evidence presented in this argument.

            ARGUMENT:
            {argument}

            CONTEXT:
            - Debate Question: {context.debate_question}
            - Phase: {context.debate_phase}

            For each piece of evidence or claim in the argument, assess:

            1. EVIDENCE IDENTIFICATION:
               - What specific claims require evidential support?
               - What evidence is actually provided for each claim?
               - What type of evidence is it? (statistical, empirical, expert opinion, anecdotal, etc.)

            2. QUALITY ASSESSMENT:
               - Reliability (1-10): How trustworthy is this evidence?
               - Relevance (1-10): How well does it support the specific claim?
               - Sufficiency (1-10): Is there enough evidence to support the claim?

            3. SOURCE EVALUATION:
               - What can we infer about the source quality?
               - Are there potential biases or conflicts of interest?
               - Is the source appropriately qualified for this type of claim?

            4. VERIFICATION STATUS:
               - Can this evidence be independently verified?
               - Are citations or sources provided?
               - How specific and concrete is the evidence?

            5. GAPS AND WEAKNESSES:
               - What important evidence is missing?
               - Where are the weakest evidential links?
               - What would strengthen the evidential case?

            Be objective in your assessment. Strong arguments acknowledge limitations
            and provide appropriate evidence for their claims.
            
            Format with clear sections for each piece of evidence evaluated.
            """

            response = await self._get_completion(evidence_prompt, context_key="evidence_assessment")
            
            if not response.success:
                logger.warning(f"Evidence assessment failed: {response.error}")
                return []

            return self._parse_evidence_assessments(response.content, argument)

        except Exception as e:
            logger.error(f"Error assessing evidence: {e}")
            return []

    def _parse_evidence_assessments(self, assessment_text: str, original_argument: str) -> List[EvidenceAssessment]:
        """
        Parse evidence assessment from LLM response.
        """
        assessments = []
        
        # Simple parsing - would use more sophisticated NLP in production
        # Look for evidence-related claims in the original argument
        sentences = original_argument.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains evidence indicators
            evidence_type = self._classify_evidence_type(sentence)
            if evidence_type != "no_evidence":
                # Extract scores from assessment (simplified)
                reliability = self._extract_score_from_assessment(assessment_text, "reliability")
                relevance = self._extract_score_from_assessment(assessment_text, "relevance")
                sufficiency = self._extract_score_from_assessment(assessment_text, "sufficiency")
                
                assessment = EvidenceAssessment(
                    claim=sentence,
                    evidence_type=evidence_type,
                    reliability_score=reliability,
                    relevance_score=relevance,
                    sufficiency_score=sufficiency,
                    source_quality=self._assess_source_quality(sentence),
                    potential_biases=self._identify_potential_biases(sentence),
                    verification_status=self._assess_verification_status(sentence)
                )
                assessments.append(assessment)
        
        return assessments[:5]  # Limit to top 5 for performance

    def _classify_evidence_type(self, sentence: str) -> str:
        """
        Classify the type of evidence in a sentence.
        """
        sentence_lower = sentence.lower()
        
        for evidence_type, indicators in self.evidence_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                return evidence_type
        
        # Check for specific patterns
        if any(word in sentence_lower for word in ["study", "research", "data", "statistics"]):
            return "empirical"
        elif any(word in sentence_lower for word in ["expert", "professor", "doctor"]):
            return "expert_opinion"
        elif any(word in sentence_lower for word in ["experience", "example", "case"]):
            return "anecdotal"
        
        return "no_evidence"

    def _extract_score_from_assessment(self, assessment_text: str, metric: str) -> float:
        """
        Extract numerical scores from assessment text.
        """
        pattern = rf'{metric}[^0-9]*(\d+)'
        match = re.search(pattern, assessment_text.lower())
        if match:
            return float(match.group(1)) if int(match.group(1)) <= 10 else float(match.group(1)) / 10
        return 5.0  # Default middle score

    def _assess_source_quality(self, sentence: str) -> str:
        """
        Assess source quality indicators in sentence.
        """
        sentence_lower = sentence.lower()
        
        if any(indicator in sentence_lower for indicator in ["peer-reviewed", "published", "journal"]):
            return "high"
        elif any(indicator in sentence_lower for indicator in ["study", "research", "data"]):
            return "medium"
        elif any(indicator in sentence_lower for indicator in ["says", "claims", "reports"]):
            return "low"
        else:
            return "unknown"

    def _identify_potential_biases(self, sentence: str) -> List[str]:
        """
        Identify potential biases in evidence presentation.
        """
        biases = []
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ["obviously", "clearly", "undoubtedly"]):
            biases.append("confirmation_bias")
        if any(word in sentence_lower for word in ["some", "many", "most"]) and "study" in sentence_lower:
            biases.append("selection_bias")
        if "personal" in sentence_lower or "experience" in sentence_lower:
            biases.append("anecdotal_bias")
        
        return biases

    def _assess_verification_status(self, sentence: str) -> str:
        """
        Assess how verifiable the evidence is.
        """
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ["doi", "journal", "published"]):
            return "verifiable"
        elif any(word in sentence_lower for word in ["study", "research", "data"]):
            return "partially_verifiable"
        else:
            return "unverifiable"

    async def _analyze_reasoning_patterns(
        self, 
        argument: str, 
        context: AuditContext
    ) -> List[ReasoningPattern]:
        """
        Analyze the reasoning patterns used in the argument.
        """
        try:
            reasoning_prompt = f"""
            Analyze the reasoning patterns and logical structure of this argument.

            ARGUMENT:
            {argument}

            Examine the argument for:

            1. REASONING TYPES:
               - Deductive reasoning (general to specific)
               - Inductive reasoning (specific to general) 
               - Abductive reasoning (best explanation)
               - Analogical reasoning (comparison-based)
               - Causal reasoning (cause-effect)

            2. LOGICAL STRUCTURE:
               - Are premises clearly stated?
               - Do conclusions follow from premises?
               - Is the logical chain coherent?
               - Are there hidden assumptions?

            3. ARGUMENT QUALITY:
               - Structure quality (1-10): How well-organized is the reasoning?
               - Logical validity (1-10): Do conclusions follow from premises?
               - Premise strength (1-10): How solid are the starting assumptions?
               - Conclusion support (1-10): How well do premises support conclusions?

            4. GAPS AND ISSUES:
               - What logical gaps exist in the reasoning?
               - Where are the weakest links in the argument?
               - What assumptions need better support?

            Provide specific analysis for each reasoning pattern you identify.
            Rate each aspect on a 1-10 scale with explanations.
            """

            response = await self._get_completion(reasoning_prompt, context_key="reasoning_analysis")
            
            if not response.success:
                logger.warning(f"Reasoning analysis failed: {response.error}")
                return []

            return self._parse_reasoning_patterns(response.content)

        except Exception as e:
            logger.error(f"Error analyzing reasoning patterns: {e}")
            return []

    def _parse_reasoning_patterns(self, analysis_text: str) -> List[ReasoningPattern]:
        """
        Parse reasoning pattern analysis from LLM response.
        """
        patterns = []
        
        # Identify reasoning types mentioned
        reasoning_types = ["deductive", "inductive", "abductive", "analogical", "causal"]
        
        for reasoning_type in reasoning_types:
            if reasoning_type in analysis_text.lower():
                # Extract quality scores
                structure_quality = self._extract_score_from_assessment(analysis_text, "structure")
                logical_validity = self._extract_score_from_assessment(analysis_text, "validity")
                premise_strength = self._extract_score_from_assessment(analysis_text, "premise")
                conclusion_support = self._extract_score_from_assessment(analysis_text, "conclusion")
                
                # Identify gaps mentioned in the text
                gaps = self._extract_reasoning_gaps(analysis_text)
                
                pattern = ReasoningPattern(
                    pattern_type=reasoning_type,
                    structure_quality=structure_quality,
                    logical_validity=logical_validity,
                    premise_strength=premise_strength,
                    conclusion_support=conclusion_support,
                    identified_gaps=gaps
                )
                patterns.append(pattern)
        
        # If no specific patterns identified, create a general analysis
        if not patterns:
            patterns.append(ReasoningPattern(
                pattern_type="general",
                structure_quality=5.0,
                logical_validity=5.0,
                premise_strength=5.0,
                conclusion_support=5.0,
                identified_gaps=["Analysis unavailable"]
            ))
        
        return patterns

    def _extract_reasoning_gaps(self, analysis_text: str) -> List[str]:
        """
        Extract identified reasoning gaps from analysis.
        """
        gaps = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['gap', 'missing', 'weakness', 'flaw']):
                if ':' in line:
                    gap = line.split(':', 1)[1].strip()
                    if gap:
                        gaps.append(gap)
        
        return gaps[:3]  # Limit to top 3

    async def _calculate_quality_metrics(
        self,
        argument: str,
        fallacies: List[LogicalFallacy],
        evidence_assessments: List[EvidenceAssessment],
        reasoning_patterns: List[ReasoningPattern]
    ) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for the argument.
        """
        try:
            # Base scores
            base_score = 7.0  # Start with decent score
            
            # Penalty for fallacies
            fallacy_penalty = len(fallacies) * 0.5
            serious_fallacy_penalty = sum(1 for f in fallacies if f.severity == "high") * 1.0
            
            # Evidence quality score
            if evidence_assessments:
                evidence_scores = [
                    (e.reliability_score + e.relevance_score + e.sufficiency_score) / 3
                    for e in evidence_assessments
                ]
                evidence_quality = sum(evidence_scores) / len(evidence_scores)
            else:
                evidence_quality = 5.0  # Neutral if no evidence assessed
            
            # Reasoning quality score
            if reasoning_patterns:
                reasoning_scores = [
                    (p.structure_quality + p.logical_validity + p.premise_strength + p.conclusion_support) / 4
                    for p in reasoning_patterns
                ]
                reasoning_quality = sum(reasoning_scores) / len(reasoning_scores)
            else:
                reasoning_quality = 5.0
            
            # Calculate component scores
            logical_coherence = max(0, base_score - fallacy_penalty - serious_fallacy_penalty)
            reasoning_clarity = reasoning_quality
            
            # Argument strength (combination of evidence and reasoning)
            argument_strength = (evidence_quality + reasoning_quality + logical_coherence) / 3
            
            # Persuasiveness (affected by fallacies and evidence)
            persuasiveness = max(0, argument_strength - (fallacy_penalty * 0.5))
            
            # Intellectual honesty (penalty for fallacies and poor evidence)
            intellectual_honesty = max(0, base_score - serious_fallacy_penalty - (evidence_quality < 4.0) * 2.0)
            
            # Overall score
            overall_score = (
                logical_coherence * 0.25 +
                evidence_quality * 0.25 +
                reasoning_clarity * 0.20 +
                argument_strength * 0.15 +
                persuasiveness * 0.10 +
                intellectual_honesty * 0.05
            )

            return QualityMetrics(
                overall_score=min(10.0, max(0.0, overall_score)),
                logical_coherence=min(10.0, max(0.0, logical_coherence)),
                evidence_quality=min(10.0, max(0.0, evidence_quality)),
                reasoning_clarity=min(10.0, max(0.0, reasoning_clarity)),
                fallacy_count=len(fallacies),
                argument_strength=min(10.0, max(0.0, argument_strength)),
                persuasiveness=min(10.0, max(0.0, persuasiveness)),
                intellectual_honesty=min(10.0, max(0.0, intellectual_honesty))
            )

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return QualityMetrics(
                overall_score=5.0, logical_coherence=5.0, evidence_quality=5.0,
                reasoning_clarity=5.0, fallacy_count=0, argument_strength=5.0,
                persuasiveness=5.0, intellectual_honesty=5.0
            )

    async def _generate_recommendations(
        self,
        argument: str,
        fallacies: List[LogicalFallacy],
        evidence_assessments: List[EvidenceAssessment],
        reasoning_patterns: List[ReasoningPattern],
        context: AuditContext
    ) -> List[str]:
        """
        Generate specific recommendations for improving the argument.
        """
        recommendations = []
        
        # Fallacy-based recommendations
        if fallacies:
            high_severity_fallacies = [f for f in fallacies if f.severity == "high"]
            if high_severity_fallacies:
                recommendations.append(f"Address critical logical fallacies: {', '.join(f.fallacy_type.replace('_', ' ') for f in high_severity_fallacies)}")
            
            for fallacy in fallacies[:3]:  # Top 3 fallacies
                if fallacy.suggested_correction:
                    recommendations.append(fallacy.suggested_correction)
        
        # Evidence-based recommendations
        if evidence_assessments:
            weak_evidence = [e for e in evidence_assessments if e.reliability_score < 5.0]
            if weak_evidence:
                recommendations.append("Strengthen evidence by using more reliable sources and providing specific citations")
            
            insufficient_evidence = [e for e in evidence_assessments if e.sufficiency_score < 5.0]
            if insufficient_evidence:
                recommendations.append("Provide additional evidence to better support your claims")
        
        # Reasoning-based recommendations
        if reasoning_patterns:
            weak_reasoning = [p for p in reasoning_patterns if p.logical_validity < 5.0]
            if weak_reasoning:
                recommendations.append("Improve logical structure by clearly stating premises and ensuring conclusions follow")
            
            weak_premises = [p for p in reasoning_patterns if p.premise_strength < 5.0]
            if weak_premises:
                recommendations.append("Strengthen the foundational assumptions of your argument")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Consider providing more specific evidence and clearer logical connections")
        
        return recommendations[:5]  # Limit to top 5

    async def _identify_strengths_and_improvements(
        self,
        argument: str,
        quality_metrics: QualityMetrics,
        fallacies: List[LogicalFallacy]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify strengths and areas for improvement.
        """
        strengths = []
        improvements = []
        
        # Identify strengths based on quality metrics
        if quality_metrics.evidence_quality > 7.0:
            strengths.append("Strong evidence quality with reliable sources")
        if quality_metrics.logical_coherence > 7.0:
            strengths.append("Logically coherent argument structure")
        if quality_metrics.reasoning_clarity > 7.0:
            strengths.append("Clear and well-structured reasoning")
        if len(fallacies) == 0:
            strengths.append("No significant logical fallacies detected")
        
        # Identify improvement areas
        if quality_metrics.evidence_quality < 5.0:
            improvements.append("Evidence quality needs significant improvement")
        if quality_metrics.logical_coherence < 5.0:
            improvements.append("Logical structure needs strengthening")
        if len(fallacies) > 2:
            improvements.append("Multiple logical fallacies should be addressed")
        
        # Default messages if nothing specific identified
        if not strengths:
            strengths.append("Argument demonstrates engagement with the topic")
        if not improvements:
            improvements.append("Continue refining evidence and logical structure")
        
        return strengths, improvements

    def _calculate_audit_confidence(
        self,
        fallacies: List[LogicalFallacy],
        evidence_assessments: List[EvidenceAssessment],
        reasoning_patterns: List[ReasoningPattern]
    ) -> float:
        """
        Calculate confidence score for the audit.
        """
        confidence_scores = []
        
        # Fallacy detection confidence
        if fallacies:
            fallacy_confidence = sum(f.confidence_score for f in fallacies) / len(fallacies)
            confidence_scores.append(fallacy_confidence)
        
        # Evidence assessment confidence (based on how much evidence was found)
        evidence_confidence = min(len(evidence_assessments) / 3.0, 1.0)  # Up to 3 pieces of evidence for full confidence
        confidence_scores.append(evidence_confidence)
        
        # Reasoning analysis confidence
        reasoning_confidence = min(len(reasoning_patterns) / 2.0, 1.0)  # Up to 2 patterns for full confidence
        confidence_scores.append(reasoning_confidence)
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Default moderate confidence

    def _determine_severity(self, confidence: float) -> str:
        """
        Determine severity level based on confidence score.
        """
        if confidence > 0.7:
            return "high"
        elif confidence > 0.4:
            return "medium"
        else:
            return "low"

    def _deduplicate_and_rank_fallacies(self, fallacies: List[LogicalFallacy]) -> List[LogicalFallacy]:
        """
        Remove duplicate fallacies and rank by importance.
        """
        # Remove duplicates by type
        seen_types = set()
        unique_fallacies = []
        
        for fallacy in fallacies:
            if fallacy.fallacy_type not in seen_types:
                unique_fallacies.append(fallacy)
                seen_types.add(fallacy.fallacy_type)
        
        # Sort by severity and confidence
        severity_order = {"high": 3, "medium": 2, "low": 1}
        unique_fallacies.sort(
            key=lambda f: (severity_order.get(f.severity, 0), f.confidence_score),
            reverse=True
        )
        
        return unique_fallacies[:5]  # Limit to top 5

    async def generate_audit_response(
        self,
        audit_report: AuditReport,
        context: AuditContext
    ) -> AgentResponse:
        """
        Generate a comprehensive audit response for the user.
        """
        try:
            # Format the audit report for presentation
            audit_summary = self._format_audit_summary(audit_report)
            
            response_content = f"""# Argument Quality Audit

## Overall Assessment
Quality Score: {audit_report.quality_metrics.overall_score:.1f}/10
Confidence: {audit_report.confidence_score:.1f}/10

{audit_summary}

## Key Recommendations
{chr(10).join(f"• {rec}" for rec in audit_report.recommendations)}

## Strengths Identified
{chr(10).join(f"• {strength}" for strength in audit_report.strengths)}

## Areas for Improvement
{chr(10).join(f"• {area}" for area in audit_report.improvement_areas)}
"""
            
            return AgentResponse(
                success=True,
                content=response_content,
                agent_type=self.agent_role,
                metadata={
                    "audit_quality_score": audit_report.quality_metrics.overall_score,
                    "fallacies_detected": len(audit_report.fallacies_detected),
                    "evidence_items": len(audit_report.evidence_assessments),
                    "reasoning_patterns": len(audit_report.reasoning_patterns),
                    "audit_confidence": audit_report.confidence_score,
                    "message_type": MessageType.AUDIT_REPORT.value,
                    "debate_phase": context.debate_phase
                }
            )

        except Exception as e:
            logger.error(f"Error generating audit response: {e}")
            return AgentResponse(
                success=False,
                content="Failed to generate audit response.",
                error=str(e),
                agent_type=self.agent_role
            )

    def _format_audit_summary(self, report: AuditReport) -> str:
        """
        Format audit report into readable summary.
        """
        sections = []
        
        # Quality metrics summary
        metrics = report.quality_metrics
        sections.append(f"""## Quality Breakdown
- Logical Coherence: {metrics.logical_coherence:.1f}/10
- Evidence Quality: {metrics.evidence_quality:.1f}/10
- Reasoning Clarity: {metrics.reasoning_clarity:.1f}/10
- Argument Strength: {metrics.argument_strength:.1f}/10""")
        
        # Fallacies section
        if report.fallacies_detected:
            fallacy_list = []
            for fallacy in report.fallacies_detected[:3]:  # Top 3
                fallacy_list.append(f"- **{fallacy.fallacy_type.replace('_', ' ').title()}** ({fallacy.severity}): {fallacy.description}")
            sections.append(f"## Logical Issues Detected\n" + "\n".join(fallacy_list))
        
        # Evidence section
        if report.evidence_assessments:
            evidence_summary = f"Analyzed {len(report.evidence_assessments)} pieces of evidence. "
            avg_reliability = sum(e.reliability_score for e in report.evidence_assessments) / len(report.evidence_assessments)
            evidence_summary += f"Average reliability: {avg_reliability:.1f}/10"
            sections.append(f"## Evidence Assessment\n{evidence_summary}")
        
        return "\n\n".join(sections)

    def _create_default_audit_report(self, argument: str) -> AuditReport:
        """
        Create a default audit report when processing fails.
        """
        return AuditReport(
            content_analyzed=argument,
            timestamp=datetime.now(),
            fallacies_detected=[],
            evidence_assessments=[],
            reasoning_patterns=[],
            quality_metrics=QualityMetrics(
                overall_score=5.0, logical_coherence=5.0, evidence_quality=5.0,
                reasoning_clarity=5.0, fallacy_count=0, argument_strength=5.0,
                persuasiveness=5.0, intellectual_honesty=5.0
            ),
            recommendations=["Audit processing failed - please review argument manually"],
            strengths=["Unable to analyze"],
            improvement_areas=["Unable to analyze"],
            confidence_score=0.0
        )

    async def cleanup(self):
        """Clean up agent resources."""
        await super().cleanup()
        logger.info(f"AuditorAgent {self.session_id} cleaned up successfully")

    def get_system_prompt(self) -> str:
        """Get the system prompt for the auditor agent."""
        return """You are an auditor agent in a structured debate. Your role is to:
        1. Identify logical fallacies and reasoning errors
        2. Assess the quality and credibility of evidence presented
        3. Check for inconsistencies and contradictions in arguments
        4. Detect potential bias or emotional manipulation
        5. Provide objective quality assessment of the debate
        
        Be thorough, impartial, and focus on logical rigor and evidence quality."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request and generate an audit response."""
        try:
            if context is None:
                context = {}
            
            # Build the prompt for auditing
            prompt = f"""
            Context: You need to audit the following debate content for quality and logical rigor.
            
            Content to Audit: {request}
            
            Your task: Analyze this content for:
            - Logical fallacies or reasoning errors
            - Quality and credibility of evidence
            - Inconsistencies or contradictions
            - Potential bias or emotional manipulation
            
            Provide a balanced assessment with specific examples.
            """
            
            # Get response from LLM
            response = await self._get_llm_response(prompt, context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=response,
                metadata={
                    "agent_type": "auditor",
                    "timestamp": datetime.now().isoformat(),
                    "context": context
                }
            )
            
        except Exception as e:
            logger.error(f"Error in auditor agent request processing: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="I encountered an error while performing the audit.",
                metadata={"error": str(e)}
            )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the auditor agent."""
        return """You are an auditor agent in a structured debate. Your role is to:
        1. Identify logical fallacies and reasoning errors
        2. Assess the quality and credibility of evidence presented
        3. Check for inconsistencies and contradictions in arguments
        4. Detect potential bias or emotional manipulation
        5. Provide objective quality assessment of the debate
        
        Be thorough, impartial, and focus on logical rigor and evidence quality."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request and generate an audit response."""
        try:
            if context is None:
                context = {}
            
            # Build the prompt for auditing
            prompt = f"""
            Context: You need to audit the following debate content for quality and logical rigor.
            
            Content to Audit: {request}
            
            Your task: Analyze this content for:
            - Logical fallacies or reasoning errors
            - Quality and credibility of evidence
            - Inconsistencies or contradictions
            - Potential bias or emotional manipulation
            
            Provide a balanced assessment with specific examples.
            """
            
            # Get response from LLM
            response = await self._get_llm_response(prompt, context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=response,
                metadata={
                    "agent_type": "auditor",
                    "timestamp": datetime.now().isoformat(),
                    "context": context
                }
            )
            
        except Exception as e:
            logger.error(f"Error in auditor agent request processing: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="I encountered an error while performing the audit.",
                metadata={"error": str(e)}
            )
