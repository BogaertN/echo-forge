import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents.base_agent import BaseAgent, AgentResponse, AgentConfig
from utils import (
    clean_text, extract_key_concepts, calculate_similarity, 
    get_logger, generate_uuid, count_words
)

logger = get_logger(__name__)

class OppositionStrategy(Enum):
    """Strategies for opposing arguments"""
    DIRECT_REFUTATION = "direct_refutation"         # Direct contradiction with evidence
    ALTERNATIVE_EXPLANATION = "alternative_explanation"  # Offer different interpretation
    CONSEQUENCE_CHALLENGE = "consequence_challenge"   # Challenge predicted outcomes
    ASSUMPTION_ATTACK = "assumption_attack"          # Target underlying assumptions
    EVIDENCE_CRITIQUE = "evidence_critique"          # Question evidence quality
    SCOPE_LIMITATION = "scope_limitation"            # Limit applicability
    PRECEDENT_COUNTER = "precedent_counter"          # Use counter-examples
    VALUES_CHALLENGE = "values_challenge"            # Challenge underlying values

class CritiqueType(Enum):
    """Types of critical analysis"""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAULTY_CAUSATION = "faulty_causation"
    OVERGENERALIZATION = "overgeneralization"
    FALSE_DICHOTOMY = "false_dichotomy"
    STRAWMAN_IDENTIFICATION = "strawman_identification"
    CORRELATION_VS_CAUSATION = "correlation_vs_causation"
    SAMPLE_BIAS = "sample_bias"

@dataclass
class CounterArgument:
    """Structure for counter-arguments"""
    claim: str
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    target_weakness: str = ""
    strategy: OppositionStrategy = OppositionStrategy.DIRECT_REFUTATION
    confidence: float = 0.0
    supporting_sources: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CriticalAnalysis:
    """Analysis of proponent's arguments for weaknesses"""
    target_argument: str
    identified_fallacies: List[str] = field(default_factory=list)
    weak_evidence: List[str] = field(default_factory=list)
    questionable_assumptions: List[str] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)
    strength_assessment: float = 0.0
    exploitable_weaknesses: List[str] = field(default_factory=list)

@dataclass
class OppositionContext:
    """Context for opposition strategy"""
    question: str
    position_taken: str = "opposition"
    proponent_arguments: List[str] = field(default_factory=list)
    counter_arguments_made: List[CounterArgument] = field(default_factory=list)
    debate_history: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    identified_weaknesses: List[str] = field(default_factory=list)
    concessions_to_avoid: List[str] = field(default_factory=list)

class OpponentAgent(BaseAgent):
    """
    Agent specialized in building counter-arguments and challenging positions
    through systematic critical analysis and alternative perspective presentation.
    """
    
    def __init__(self, **kwargs):
        # Default configuration optimized for critical analysis
        default_config = AgentConfig(
            model="llama3.1:8b",  # Large model for complex critical reasoning
            temperature=0.6,      # Slightly lower for more focused criticism
            max_tokens=1024,      # Room for detailed counter-arguments
            timeout=90,           # Time for thorough analysis
            enable_tools=True,    # Use tools for fact-checking opponent claims
            enable_memory=True,
            memory_limit=30       # Remember extensive debate context
        )
        
        # Merge with provided config
        config = kwargs.pop('config', default_config)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        super().__init__(config=config, **kwargs)
        
        # Opposition-specific state
        self.opposition_context: Optional[OppositionContext] = None
        self.critique_frameworks = self._initialize_critique_frameworks()
        self.fallacy_patterns = self._initialize_fallacy_patterns()
        self.alternative_perspectives = self._initialize_alternative_perspectives()
        
        logger.info(f"OpponentAgent initialized: {self.agent_id}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the opponent agent"""
        return """You are a skilled opponent agent specializing in critical analysis and counter-argumentation. Your role is to:

## Primary Objectives:
1. **Challenge arguments systematically** through rigorous critical analysis
2. **Identify logical weaknesses** and exploit them constructively
3. **Present alternative perspectives** that are equally valid or stronger
4. **Question assumptions** that may be taken for granted
5. **Provide evidence-based rebuttals** using credible sources
6. **Maintain intellectual honesty** while being a strong advocate for the opposing view

## Critical Analysis Framework:
- **Logical consistency**: Identify contradictions and inconsistencies
- **Evidence quality**: Evaluate source credibility and methodology
- **Assumption analysis**: Challenge unstated premises
- **Scope limitations**: Point out overgeneralizations
- **Alternative explanations**: Offer different interpretations of the same data
- **Unintended consequences**: Highlight potential negative outcomes

## Opposition Strategies:
- **Direct refutation**: Contradict with stronger evidence
- **Alternative explanation**: Reframe the issue differently
- **Consequence challenge**: Question predicted outcomes
- **Assumption attack**: Target foundational beliefs
- **Evidence critique**: Question methodology and sources
- **Scope limitation**: Show where arguments don't apply
- **Precedent counter**: Use historical counter-examples

## Logical Fallacy Detection:
- **Ad hominem**: Attacks on the person rather than argument
- **Straw man**: Misrepresenting opponent's position
- **False dichotomy**: Presenting only two options when more exist
- **Hasty generalization**: Drawing broad conclusions from limited data
- **Appeal to authority**: Relying on authority rather than evidence
- **Correlation vs causation**: Confusing correlation with cause
- **Slippery slope**: Assuming one event will lead to extreme consequences

## Quality Standards:
- **Constructive opposition**: Challenge ideas, not people
- **Evidence-based**: Support counter-arguments with credible sources
- **Intellectually honest**: Acknowledge strong points in opponent's case
- **Proportional response**: Match the strength of criticism to actual weaknesses
- **Alternative solutions**: Don't just tear down - offer better approaches

## Tone and Approach:
- Respectful but firm in disagreement
- Curious and questioning rather than dismissive
- Focused on strengthening overall discourse
- Committed to truth-seeking through rigorous examination
- Professional and constructive in criticism

Remember: Your goal is not to win at all costs, but to ensure that all positions are rigorously tested and that the strongest arguments emerge through constructive opposition."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request for counter-argument generation"""
        try:
            # Determine the type of opposition needed
            if context and context.get("role") == "opening_statement":
                return await self.generate_opening_opposition(request, context)
            elif context and context.get("role") == "debate_response":
                return await self.generate_debate_response(request, context)
            elif context and context.get("role") == "rebuttal":
                return await self.generate_rebuttal(request, context)
            else:
                # Default: general counter-argument
                return await self.generate_counter_argument(request, context)
                
        except Exception as e:
            logger.error(f"Error in opponent process_request: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error while analyzing the opposing position. Let me provide a different perspective.",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def generate_opening_statement(self, 
                                       question: str,
                                       proponent_statement: str,
                                       tools_enabled: bool = None,
                                       tool_manager = None) -> Dict[str, Any]:
        """
        Generate opening opposition statement after hearing proponent.
        
        Args:
            question: The debate question
            proponent_statement: The proponent's opening statement
            tools_enabled: Whether to use external tools
            tool_manager: Tool manager for research
            
        Returns:
            Dictionary with opposition statement and analysis
        """
        try:
            # Initialize opposition context
            if not self.opposition_context:
                self.opposition_context = OppositionContext(
                    question=question,
                    position_taken="opposition"
                )
            
            self.opposition_context.proponent_arguments.append(proponent_statement)
            
            # Set tool manager if provided
            if tool_manager:
                self.set_tool_manager(tool_manager)
            
            # Analyze proponent's opening for weaknesses
            critical_analysis = await self._analyze_proponent_argument(proponent_statement)
            
            # Research counter-evidence if tools enabled
            counter_research = None
            if tools_enabled and self.tool_manager:
                counter_research = await self._conduct_opposition_research(
                    question, proponent_statement, critical_analysis
                )
            
            # Build counter-argument structure
            counter_argument = await self._build_opening_opposition(
                question, proponent_statement, critical_analysis, counter_research
            )
            
            # Generate the opposition statement
            statement_response = await self._generate_opposition_statement(
                counter_argument, "opening"
            )
            
            # Update context
            self.opposition_context.counter_arguments_made.append(counter_argument)
            self.opposition_context.identified_weaknesses.extend(
                critical_analysis.exploitable_weaknesses
            )
            self.opposition_context.current_round = 1
            
            return {
                "statement": statement_response.content,
                "reasoning": statement_response.reasoning or "",
                "key_points": [counter_argument.claim] + counter_argument.evidence[:2],
                "sources": counter_research.get("sources", []) if counter_research else [],
                "weaknesses_identified": critical_analysis.exploitable_weaknesses,
                "strategy_used": counter_argument.strategy.value,
                "fallacies_detected": critical_analysis.identified_fallacies
            }
            
        except Exception as e:
            logger.error(f"Error generating opening opposition: {e}")
            return {
                "statement": f"I respectfully disagree with the position presented and will demonstrate why the opposing view deserves serious consideration.",
                "reasoning": "Error in detailed analysis, providing basic opposition",
                "key_points": [],
                "sources": [],
                "weaknesses_identified": [],
                "strategy_used": "direct_refutation"
            }
    
    async def generate_debate_response(self,
                                     question: str,
                                     debate_history: List[Dict[str, Any]],
                                     round_number: int,
                                     tools_enabled: bool = None,
                                     tool_manager = None) -> Dict[str, Any]:
        """
        Generate opposition response during main debate rounds.
        
        Args:
            question: The debate question
            debate_history: Previous exchanges in the debate
            round_number: Current round number
            tools_enabled: Whether to use external tools
            tool_manager: Tool manager for research
            
        Returns:
            Dictionary with opposition response and analysis
        """
        try:
            # Update opposition context
            if not self.opposition_context:
                self.opposition_context = OppositionContext(
                    question=question,
                    position_taken="opposition"
                )
            
            self.opposition_context.current_round = round_number
            self._update_opposition_context_from_history(debate_history)
            
            # Set tool manager if provided
            if tool_manager:
                self.set_tool_manager(tool_manager)
            
            # Analyze latest proponent arguments
            proponent_analysis = await self._analyze_latest_proponent_arguments(debate_history)
            
            # Determine opposition strategy for this round
            opposition_strategy = await self._determine_opposition_strategy(
                proponent_analysis, round_number
            )
            
            # Fact-check proponent claims if tools available
            fact_check_results = None
            if tools_enabled and self.tool_manager and proponent_analysis.get("claims_to_verify"):
                fact_check_results = await self._fact_check_proponent_claims(
                    proponent_analysis["claims_to_verify"]
                )
            
            # Build response counter-argument
            response_counter = await self._build_response_opposition(
                proponent_analysis, opposition_strategy, fact_check_results
            )
            
            # Generate the response
            response = await self._generate_opposition_statement(
                response_counter, "response"
            )
            
            # Extract sources
            sources = []
            if fact_check_results:
                sources.extend(fact_check_results.get("sources", []))
            
            # Update tracking
            self.opposition_context.counter_arguments_made.append(response_counter)
            
            return {
                "response": response.content,
                "key_points": [response_counter.claim] + response_counter.evidence[:2],
                "sources": sources,
                "strategy_used": response_counter.strategy.value,
                "proponent_weaknesses_targeted": proponent_analysis.get("weaknesses", []),
                "new_evidence_presented": len(response_counter.evidence),
                "confidence": response_counter.confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating opposition response: {e}")
            return {
                "response": "I maintain my opposition and will address the new points raised while strengthening my counter-arguments.",
                "key_points": [],
                "sources": [],
                "strategy_used": "direct_refutation",
                "proponent_weaknesses_targeted": []
            }
    
    async def generate_counter_argument(self, topic: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Generate a general counter-argument for a topic"""
        try:
            prompt = f"""
            I need to construct a strong counter-argument to the following position: {topic}
            
            Please provide:
            1. A clear opposing claim
            2. Evidence or reasoning that challenges the original position
            3. Alternative perspectives or interpretations
            4. Potential weaknesses in the original argument
            5. Constructive criticism that strengthens overall discourse
            
            Approach this as a thoughtful critic seeking truth through rigorous examination.
            """
            
            # Add context if provided
            if context:
                prompt += f"\n\nAdditional context: {context}"
            
            response = await self.chat(prompt)
            
            # Enhance with critical analysis
            key_points = self._extract_critical_points(response.content)
            response.key_points = key_points
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating counter-argument: {e}")
            raise
    
    async def generate_rebuttal(self, argument_to_rebut: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Generate a rebuttal to a specific argument"""
        try:
            # Analyze the argument for specific weaknesses
            analysis = await self._quick_critical_analysis(argument_to_rebut)
            
            prompt = f"""
            I need to construct a strong rebuttal to this argument: {argument_to_rebut}
            
            Based on my analysis, I should focus on:
            - Logical weaknesses: {', '.join(analysis.get('logical_issues', []))}
            - Evidence problems: {', '.join(analysis.get('evidence_issues', []))}
            - Alternative explanations or interpretations
            - Unaddressed complications or consequences
            
            Provide a respectful but thorough rebuttal that challenges the core claims.
            """
            
            response = await self.chat(prompt, context)
            
            # Add rebuttal-specific metadata
            response.metadata.update({
                "rebuttal_targets": analysis.get("weaknesses", []),
                "argument_type": "rebuttal",
                "critical_analysis": analysis
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating rebuttal: {e}")
            raise
    
    async def _analyze_proponent_argument(self, argument: str) -> CriticalAnalysis:
        """Conduct thorough critical analysis of proponent's argument"""
        try:
            analysis = CriticalAnalysis(target_argument=argument)
            
            # Identify logical fallacies
            analysis.identified_fallacies = self._detect_logical_fallacies(argument)
            
            # Identify weak evidence
            analysis.weak_evidence = self._identify_weak_evidence(argument)
            
            # Identify questionable assumptions
            analysis.questionable_assumptions = self._identify_assumptions(argument)
            
            # Generate alternative interpretations
            analysis.alternative_interpretations = self._generate_alternative_interpretations(argument)
            
            # Assess overall strength
            analysis.strength_assessment = self._assess_argument_strength(argument, analysis)
            
            # Identify exploitable weaknesses
            analysis.exploitable_weaknesses = self._identify_exploitable_weaknesses(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in critical analysis: {e}")
            return CriticalAnalysis(
                target_argument=argument,
                exploitable_weaknesses=["general logical concerns"]
            )
    
    def _detect_logical_fallacies(self, argument: str) -> List[str]:
        """Detect logical fallacies in the argument"""
        fallacies = []
        argument_lower = argument.lower()
        
        # Check for common fallacies
        if any(phrase in argument_lower for phrase in ["all", "every", "never", "always", "everyone"]):
            fallacies.append("hasty_generalization")
        
        if any(phrase in argument_lower for phrase in ["either", "only two", "must choose"]):
            fallacies.append("false_dichotomy")
        
        if "because" in argument_lower and "therefore" in argument_lower:
            # Check for circular reasoning (simplified detection)
            sentences = argument.split('.')
            if len(sentences) > 1:
                for i, sentence in enumerate(sentences[:-1]):
                    if calculate_similarity(sentence, sentences[i+1]) > 0.7:
                        fallacies.append("circular_reasoning")
                        break
        
        if any(phrase in argument_lower for phrase in ["studies show", "experts say"]):
            if not any(specific in argument_lower for specific in ["university", "published", "journal"]):
                fallacies.append("appeal_to_authority")
        
        if any(phrase in argument_lower for phrase in ["leads to", "will result in", "slippery slope"]):
            fallacies.append("slippery_slope")
        
        if any(phrase in argument_lower for phrase in ["feel", "terrible", "disaster", "amazing"]):
            emotional_count = sum(1 for word in ["feel", "terrible", "disaster", "amazing", "awful", "wonderful"] 
                                if word in argument_lower)
            if emotional_count > 2:
                fallacies.append("appeal_to_emotion")
        
        return list(set(fallacies))  # Remove duplicates
    
    def _identify_weak_evidence(self, argument: str) -> List[str]:
        """Identify weak or problematic evidence"""
        weak_evidence = []
        argument_lower = argument.lower()
        
        # Unsupported statistical claims
        if re.search(r'\d+%', argument) or re.search(r'\d+', argument):
            if not any(source in argument_lower for source in ["study", "research", "survey", "data from"]):
                weak_evidence.append("unsupported_statistics")
        
        # Vague research references
        if "studies show" in argument_lower or "research proves" in argument_lower:
            if not any(specific in argument_lower for specific in ["university", "journal", "published", "peer-reviewed"]):
                weak_evidence.append("vague_research_claims")
        
        # Anecdotal evidence presented as general truth
        if any(phrase in argument_lower for phrase in ["i know someone", "my experience", "i've seen"]):
            weak_evidence.append("anecdotal_evidence")
        
        # Correlation presented as causation
        if any(phrase in argument_lower for phrase in ["correlation", "associated with", "linked to"]):
            if any(causal in argument_lower for causal in ["causes", "leads to", "results in"]):
                weak_evidence.append("correlation_causation_confusion")
        
        return weak_evidence
    
    def _identify_assumptions(self, argument: str) -> List[str]:
        """Identify questionable assumptions"""
        assumptions = []
        argument_lower = argument.lower()
        
        # Assumptions about human behavior
        if any(phrase in argument_lower for phrase in ["people will", "everyone wants", "humans naturally"]):
            assumptions.append("behavioral_assumptions")
        
        # Economic assumptions
        if any(phrase in argument_lower for phrase in ["market will", "economy will", "costs will"]):
            assumptions.append("economic_assumptions")
        
        # Technology assumptions
        if any(phrase in argument_lower for phrase in ["technology will", "innovation will", "automation will"]):
            assumptions.append("technological_assumptions")
        
        # Causal assumptions
        if any(phrase in argument_lower for phrase in ["will lead to", "results in", "causes"]):
            assumptions.append("causal_assumptions")
        
        # Value assumptions
        if any(phrase in argument_lower for phrase in ["obviously", "clearly", "everyone knows"]):
            assumptions.append("shared_values_assumptions")
        
        return assumptions
    
    def _generate_alternative_interpretations(self, argument: str) -> List[str]:
        """Generate alternative ways to interpret the argument"""
        alternatives = []
        
        # If argument claims benefits, consider costs
        if any(word in argument.lower() for word in ["benefit", "advantage", "improve", "better"]):
            alternatives.append("Focus on hidden costs and unintended consequences")
        
        # If argument uses causation, consider correlation
        if any(word in argument.lower() for word in ["causes", "leads to", "results in"]):
            alternatives.append("Relationship may be correlational rather than causal")
        
        # If argument generalizes, consider specificity
        if any(word in argument.lower() for word in ["all", "every", "always", "never"]):
            alternatives.append("Consider exceptions and specific contexts where this doesn't apply")
        
        # If argument assumes single cause, consider multiple factors
        if "because" in argument.lower():
            alternatives.append("Multiple factors likely contribute to this outcome")
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _assess_argument_strength(self, argument: str, analysis: CriticalAnalysis) -> float:
        """Assess the overall strength of the argument"""
        base_strength = 0.7  # Assume moderate strength initially
        
        # Deduct for fallacies
        fallacy_penalty = len(analysis.identified_fallacies) * 0.15
        
        # Deduct for weak evidence
        evidence_penalty = len(analysis.weak_evidence) * 0.1
        
        # Deduct for questionable assumptions
        assumption_penalty = len(analysis.questionable_assumptions) * 0.1
        
        # Add bonus for length and detail (indicates thought)
        length_bonus = min(0.1, len(argument.split()) / 1000)
        
        final_strength = max(0.0, base_strength - fallacy_penalty - evidence_penalty - assumption_penalty + length_bonus)
        return min(1.0, final_strength)
    
    def _identify_exploitable_weaknesses(self, analysis: CriticalAnalysis) -> List[str]:
        """Identify the most exploitable weaknesses for counter-argument"""
        weaknesses = []
        
        # Prioritize logical fallacies
        if analysis.identified_fallacies:
            for fallacy in analysis.identified_fallacies:
                weaknesses.append(f"logical_fallacy_{fallacy}")
        
        # Add evidence issues
        if analysis.weak_evidence:
            weaknesses.append("insufficient_evidence_quality")
        
        # Add assumption issues
        if len(analysis.questionable_assumptions) >= 2:
            weaknesses.append("multiple_questionable_assumptions")
        
        # Add alternative interpretation opportunities
        if len(analysis.alternative_interpretations) >= 2:
            weaknesses.append("alternative_explanations_available")
        
        return weaknesses[:4]  # Top 4 exploitable weaknesses
    
    async def _conduct_opposition_research(self, 
                                         question: str, 
                                         proponent_argument: str,
                                         analysis: CriticalAnalysis) -> Dict[str, Any]:
        """Conduct research to support opposition arguments"""
        research_results = {"sources": [], "counter_evidence": [], "contradictions": []}
        
        try:
            if not self.tool_manager:
                return research_results
            
            # Extract claims to fact-check
            claims_to_check = self._extract_factual_claims(proponent_argument)
            
            # Fact-check key claims
            for claim in claims_to_check[:3]:
                try:
                    fact_check = await self.tool_manager.fact_check(
                        claim,
                        agent_id=self.agent_id,
                        session_id=self.session_id
                    )
                    
                    if fact_check.verdict in ["false", "mixed"]:
                        research_results["contradictions"].append({
                            "claim": claim,
                            "verdict": fact_check.verdict,
                            "explanation": fact_check.explanation,
                            "confidence": fact_check.confidence
                        })
                    
                    research_results["sources"].extend(fact_check.sources)
                    
                except Exception as e:
                    logger.warning(f"Fact-check failed: {e}")
                    continue
            
            # Search for counter-evidence
            key_concepts = extract_key_concepts(question + " " + proponent_argument, max_concepts=3)
            
            for concept in key_concepts:
                try:
                    search_queries = [
                        f"{concept} problems disadvantages",
                        f"{concept} criticism concerns",
                        f"why {concept} fails"
                    ]
                    
                    for query in search_queries[:2]:  # Limit searches
                        search_results = await self.tool_manager.web_search(
                            query, max_results=2,
                            agent_id=self.agent_id,
                            session_id=self.session_id
                        )
                        
                        for result in search_results:
                            research_results["sources"].append({
                                "title": result.title,
                                "url": result.url,
                                "snippet": result.snippet,
                                "relevance": result.relevance_score,
                                "search_type": "counter_evidence"
                            })
                            
                            # Extract counter-evidence
                            if any(indicator in result.snippet.lower() for indicator in 
                                   ["problem", "concern", "risk", "disadvantage", "criticism", "fails"]):
                                research_results["counter_evidence"].append(result.snippet)
                                
                except Exception as e:
                    logger.warning(f"Counter-evidence search failed: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in opposition research: {e}")
        
        return research_results
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims for verification"""
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                (re.search(r'\d+%', sentence) or  # Percentages
                 re.search(r'\d+', sentence) or   # Numbers
                 any(word in sentence.lower() for word in ["studies", "research", "evidence", "data", "statistics"]))):
                claims.append(sentence)
        
        return claims[:4]  # Limit to 4 claims
    
    async def _build_opening_opposition(self, 
                                      question: str,
                                      proponent_statement: str,
                                      analysis: CriticalAnalysis,
                                      research: Optional[Dict[str, Any]]) -> CounterArgument:
        """Build opening opposition argument"""
        try:
            # Choose primary opposition strategy
            if analysis.identified_fallacies:
                strategy = OppositionStrategy.EVIDENCE_CRITIQUE
            elif len(analysis.questionable_assumptions) >= 2:
                strategy = OppositionStrategy.ASSUMPTION_ATTACK
            elif analysis.alternative_interpretations:
                strategy = OppositionStrategy.ALTERNATIVE_EXPLANATION
            else:
                strategy = OppositionStrategy.DIRECT_REFUTATION
            
            # Formulate main counter-claim
            counter_claim = self._formulate_counter_claim(question, proponent_statement, strategy)
            
            # Build evidence list
            evidence = []
            
            # Add logical critiques
            if analysis.identified_fallacies:
                evidence.append(f"The proponent's argument contains logical fallacies: {', '.join(analysis.identified_fallacies)}")
            
            # Add research contradictions
            if research and research.get("contradictions"):
                for contradiction in research["contradictions"][:2]:
                    evidence.append(f"Fact-checking reveals: {contradiction['explanation'][:100]}...")
            
            # Add counter-evidence
            if research and research.get("counter_evidence"):
                for counter_ev in research["counter_evidence"][:2]:
                    evidence.append(f"Contrary evidence shows: {counter_ev[:100]}...")
            
            # Add alternative interpretations
            if analysis.alternative_interpretations:
                evidence.append(f"Alternative perspective: {analysis.alternative_interpretations[0]}")
            
            # Calculate confidence
            confidence = self._calculate_opposition_confidence(analysis, research)
            
            return CounterArgument(
                claim=counter_claim,
                evidence=evidence,
                reasoning=f"Using {strategy.value} to challenge the proponent's position",
                target_weakness=analysis.exploitable_weaknesses[0] if analysis.exploitable_weaknesses else "general_critique",
                strategy=strategy,
                confidence=confidence,
                supporting_sources=research.get("sources", []) if research else []
            )
            
        except Exception as e:
            logger.error(f"Error building opening opposition: {e}")
            return CounterArgument(
                claim="I respectfully disagree with the proponent's position and will present alternative evidence",
                evidence=["The opposing view deserves equal consideration"],
                strategy=OppositionStrategy.DIRECT_REFUTATION,
                confidence=0.6
            )
    
    def _formulate_counter_claim(self, question: str, proponent_statement: str, strategy: OppositionStrategy) -> str:
        """Formulate the main counter-claim"""
        question_lower = question.lower()
        
        if strategy == OppositionStrategy.DIRECT_REFUTATION:
            if question_lower.startswith("should"):
                action = question[6:].strip().rstrip("?")
                return f"No, {action} would be problematic for several important reasons I will demonstrate."
            else:
                return "I argue against the proponent's position and will show why the alternative view is more compelling."
        
        elif strategy == OppositionStrategy.ASSUMPTION_ATTACK:
            return "The proponent's argument relies on several questionable assumptions that, when examined, significantly weaken their case."
        
        elif strategy == OppositionStrategy.EVIDENCE_CRITIQUE:
            return "While the proponent presents their case confidently, a closer examination of their evidence reveals significant problems."
        
        elif strategy == OppositionStrategy.ALTERNATIVE_EXPLANATION:
            return "There are alternative explanations and perspectives that cast serious doubt on the proponent's conclusions."
        
        else:
            return "I respectfully but firmly disagree with the proponent's position for the following reasons."
    
    def _calculate_opposition_confidence(self, analysis: CriticalAnalysis, research: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in opposition argument"""
        base_confidence = 0.6
        
        # Boost for fallacies detected
        fallacy_boost = len(analysis.identified_fallacies) * 0.1
        
        # Boost for research contradictions
        research_boost = 0.0
        if research and research.get("contradictions"):
            research_boost = len(research["contradictions"]) * 0.15
        
        # Boost for weak evidence identified
        evidence_boost = len(analysis.weak_evidence) * 0.05
        
        # Penalty if proponent's argument is very strong
        strength_penalty = 0.0
        if analysis.strength_assessment > 0.8:
            strength_penalty = 0.2
        
        final_confidence = base_confidence + fallacy_boost + research_boost + evidence_boost - strength_penalty
        return max(0.1, min(1.0, final_confidence))
    
    async def _generate_opposition_statement(self, counter_arg: CounterArgument, role: str) -> AgentResponse:
        """Generate the opposition statement from counter-argument structure"""
        try:
            if role == "opening":
                prompt = f"""
                Generate a strong opening opposition statement that argues: {counter_arg.claim}
                
                Key evidence to present:
                {chr(10).join([f"- {evidence}" for evidence in counter_arg.evidence])}
                
                Strategy: {counter_arg.strategy.value}
                Target weakness: {counter_arg.target_weakness}
                
                Structure the opposition as:
                1. Clear statement of disagreement with reasoning
                2. Systematic presentation of counter-evidence
                3. Challenge to proponent's key assumptions or logic
                4. Confident but respectful tone
                
                Make it compelling and well-reasoned, approximately 200-300 words.
                """
            else:
                prompt = f"""
                Generate a strong opposition response that argues: {counter_arg.claim}
                
                Evidence to present: {'; '.join(counter_arg.evidence)}
                Strategy: {counter_arg.strategy.value}
                
                Build systematic opposition while maintaining constructive discourse.
                """
            
            response = await self.chat(prompt)
            
            # Enhance with opposition metadata
            response.reasoning = f"Using {counter_arg.strategy.value} to target {counter_arg.target_weakness}"
            response.confidence = counter_arg.confidence
            response.metadata.update({
                "opposition_strategy": counter_arg.strategy.value,
                "target_weakness": counter_arg.target_weakness,
                "evidence_count": len(counter_arg.evidence),
                "sources_used": len(counter_arg.supporting_sources)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating opposition statement: {e}")
            return AgentResponse(
                content=f"I oppose the proponent's position because {counter_arg.claim}",
                confidence=0.5,
                reasoning="Basic opposition due to generation error"
            )
    
    def _update_opposition_context_from_history(self, debate_history: List[Dict[str, Any]]):
        """Update opposition context from debate history"""
        if not self.opposition_context:
            return
        
        self.opposition_context.debate_history = debate_history
        self.opposition_context.proponent_arguments.clear()
        
        # Extract proponent arguments
        for entry in debate_history:
            if entry.get("agent") == "proponent":
                self.opposition_context.proponent_arguments.append(entry.get("content", ""))
    
    async def _analyze_latest_proponent_arguments(self, debate_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze latest proponent arguments for new weaknesses"""
        try:
            proponent_entries = [entry for entry in debate_history if entry.get("agent") == "proponent"]
            
            if not proponent_entries:
                return {"main_points": [], "weaknesses": [], "claims_to_verify": []}
            
            latest_proponent = proponent_entries[-1]
            content = latest_proponent.get("content", "")
            
            # Quick critical analysis
            analysis = await self._quick_critical_analysis(content)
            
            return {
                "main_points": self._extract_argument_points(content),
                "weaknesses": analysis.get("weaknesses", []),
                "claims_to_verify": self._extract_factual_claims(content),
                "new_fallacies": analysis.get("logical_issues", []),
                "strength_assessment": analysis.get("strength", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing latest proponent arguments: {e}")
            return {"main_points": [], "weaknesses": [], "claims_to_verify": []}
    
    def _extract_argument_points(self, content: str) -> List[str]:
        """Extract main argument points"""
        points = []
        
        # Look for structured points
        bullet_pattern = r'(?:^|\n)[-•*]\s*(.+)'
        bullets = re.findall(bullet_pattern, content, re.MULTILINE)
        points.extend(bullets)
        
        numbered_pattern = r'(?:^|\n)\d+\.\s*(.+)'
        numbered = re.findall(numbered_pattern, content, re.MULTILINE)
        points.extend(numbered)
        
        # Extract key sentences if no structure
        if not points:
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 20 and 
                    any(indicator in sentence.lower() for indicator in 
                        ["because", "therefore", "evidence", "shows", "proves"])):
                    points.append(sentence)
        
        return points[:5]
    
    async def _determine_opposition_strategy(self, proponent_analysis: Dict[str, Any], round_number: int) -> OppositionStrategy:
        """Determine best opposition strategy for this round"""
        weaknesses = proponent_analysis.get("weaknesses", [])
        new_fallacies = proponent_analysis.get("new_fallacies", [])
        strength = proponent_analysis.get("strength_assessment", 0.5)
        
        # If new logical issues, exploit them
        if new_fallacies:
            return OppositionStrategy.EVIDENCE_CRITIQUE
        
        # If argument is weak, direct refutation
        if strength < 0.4:
            return OppositionStrategy.DIRECT_REFUTATION
        
        # If strong argument, look for alternative explanations
        if strength > 0.7:
            return OppositionStrategy.ALTERNATIVE_EXPLANATION
        
        # Later rounds: focus on consequences
        if round_number > 3:
            return OppositionStrategy.CONSEQUENCE_CHALLENGE
        
        # Default strategy
        return OppositionStrategy.ASSUMPTION_ATTACK
    
    async def _fact_check_proponent_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Fact-check proponent's claims for contradictions"""
        fact_check_results = {"sources": [], "contradictions": [], "confirmations": []}
        
        try:
            if not self.tool_manager:
                return fact_check_results
            
            for claim in claims[:3]:  # Limit to 3 claims
                try:
                    fact_check = await self.tool_manager.fact_check(
                        claim,
                        agent_id=self.agent_id,
                        session_id=self.session_id
                    )
                    
                    if fact_check.verdict in ["false", "mixed"]:
                        fact_check_results["contradictions"].append({
                            "claim": claim,
                            "verdict": fact_check.verdict,
                            "explanation": fact_check.explanation,
                            "confidence": fact_check.confidence
                        })
                    
                    fact_check_results["sources"].extend(fact_check.sources)
                    
                except Exception as e:
                    logger.warning(f"Fact-check failed: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error fact-checking proponent claims: {e}")
        
        return fact_check_results
    
    async def _build_response_opposition(self, 
                                       proponent_analysis: Dict[str, Any],
                                       strategy: OppositionStrategy,
                                       fact_check: Optional[Dict[str, Any]]) -> CounterArgument:
        """Build opposition argument for response"""
        try:
            evidence = []
            
            # Address new weaknesses
            weaknesses = proponent_analysis.get("weaknesses", [])
            for weakness in weaknesses[:2]:
                evidence.append(f"The proponent's latest argument shows {weakness}")
            
            # Use fact-check contradictions
            if fact_check and fact_check.get("contradictions"):
                for contradiction in fact_check["contradictions"][:2]:
                    evidence.append(f"Fact-checking contradicts their claim: {contradiction['explanation'][:100]}...")
            
            # Strategic responses based on strategy
            if strategy == OppositionStrategy.CONSEQUENCE_CHALLENGE:
                evidence.append("The proponent fails to address serious negative consequences of their position")
            elif strategy == OppositionStrategy.ALTERNATIVE_EXPLANATION:
                evidence.append("There are compelling alternative explanations that better fit the evidence")
            
            # Ensure minimum evidence
            if len(evidence) < 2:
                evidence.append("The proponent's arguments remain unconvincing for the reasons I've outlined")
            
            claim = "The proponent's latest arguments actually strengthen my opposition case"
            confidence = self._calculate_response_confidence(proponent_analysis, fact_check)
            
            return CounterArgument(
                claim=claim,
                evidence=evidence,
                reasoning=f"Responding with {strategy.value} based on proponent's weaknesses",
                strategy=strategy,
                confidence=confidence,
                supporting_sources=fact_check.get("sources", []) if fact_check else []
            )
            
        except Exception as e:
            logger.error(f"Error building response opposition: {e}")
            return CounterArgument(
                claim="I maintain my opposition with additional supporting arguments",
                evidence=["The proponent has not adequately addressed my concerns"],
                strategy=OppositionStrategy.DIRECT_REFUTATION,
                confidence=0.6
            )
    
    def _calculate_response_confidence(self, proponent_analysis: Dict[str, Any], fact_check: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in response opposition"""
        base_confidence = 0.6
        
        # Boost for new weaknesses found
        weakness_boost = len(proponent_analysis.get("weaknesses", [])) * 0.1
        
        # Boost for fact-check contradictions
        fact_check_boost = 0.0
        if fact_check and fact_check.get("contradictions"):
            fact_check_boost = len(fact_check["contradictions"]) * 0.2
        
        # Adjustment based on proponent strength
        proponent_strength = proponent_analysis.get("strength_assessment", 0.5)
        strength_adjustment = (1.0 - proponent_strength) * 0.3
        
        final_confidence = base_confidence + weakness_boost + fact_check_boost + strength_adjustment
        return max(0.2, min(1.0, final_confidence))
    
    async def _quick_critical_analysis(self, argument: str) -> Dict[str, Any]:
        """Quick critical analysis for immediate response"""
        return {
            "logical_issues": self._detect_logical_fallacies(argument),
            "evidence_issues": self._identify_weak_evidence(argument),
            "weaknesses": self._identify_quick_weaknesses(argument),
            "strength": self._quick_strength_assessment(argument)
        }
    
    def _identify_quick_weaknesses(self, argument: str) -> List[str]:
        """Quick identification of argument weaknesses"""
        weaknesses = []
        argument_lower = argument.lower()
        
        if any(word in argument_lower for word in ["all", "every", "never", "always"]):
            weaknesses.append("overgeneralization")
        
        if "because" in argument_lower and not any(evidence in argument_lower for evidence in ["study", "data", "research"]):
            weaknesses.append("unsupported_causation")
        
        if any(word in argument_lower for word in ["feel", "believe"]) and not any(fact in argument_lower for fact in ["fact", "evidence", "data"]):
            weaknesses.append("opinion_over_fact")
        
        return weaknesses
    
    def _quick_strength_assessment(self, argument: str) -> float:
        """Quick strength assessment"""
        strength = 0.5
        
        # Evidence indicators
        if any(word in argument.lower() for word in ["study", "research", "data", "evidence"]):
            strength += 0.2
        
        # Logical structure
        if "because" in argument.lower() and "therefore" in argument.lower():
            strength += 0.1
        
        # Length and detail
        if len(argument.split()) > 100:
            strength += 0.1
        
        # Qualification and nuance
        if any(word in argument.lower() for word in ["however", "although", "while", "may"]):
            strength += 0.1
        
        return min(1.0, strength)
    
    def _extract_critical_points(self, content: str) -> List[str]:
        """Extract critical analysis points from content"""
        points = []
        
        # Look for critical phrases
        critical_patterns = [
            r'however[,\s]+(.+?)(?:\.|$)',
            r'but[,\s]+(.+?)(?:\.|$)', 
            r'although[,\s]+(.+?)(?:\.|$)',
            r'the problem[,\s]+(.+?)(?:\.|$)',
            r'the issue[,\s]+(.+?)(?:\.|$)'
        ]
        
        for pattern in critical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            points.extend([match.strip() for match in matches])
        
        return points[:5]
    
    def _initialize_critique_frameworks(self) -> Dict[str, List[str]]:
        """Initialize frameworks for critical analysis"""
        return {
            "logical_structure": [
                "Are the premises clearly stated?",
                "Does the conclusion follow from the premises?",
                "Are there any logical leaps or gaps?"
            ],
            "evidence_quality": [
                "Is the evidence credible and recent?",
                "Are sources properly cited and verifiable?",
                "Is the sample size appropriate?"
            ],
            "assumption_analysis": [
                "What unstated assumptions underlie this argument?",
                "Are these assumptions reasonable and justified?",
                "How would the argument change if assumptions were different?"
            ]
        }
    
    def _initialize_fallacy_patterns(self) -> Dict[str, List[str]]:
        """Initialize logical fallacy detection patterns"""
        return {
            "ad_hominem": ["attacks", "personally", "character"],
            "straw_man": ["misrepresents", "distorts", "oversimplifies"],
            "false_dichotomy": ["only two", "either", "must choose"],
            "appeal_to_authority": ["expert says", "authority claims"],
            "hasty_generalization": ["all", "every", "always", "never"],
            "slippery_slope": ["leads to", "will result in", "inevitably"]
        }
    
    def _initialize_alternative_perspectives(self) -> Dict[str, List[str]]:
        """Initialize alternative perspective templates"""
        return {
            "economic": ["cost-benefit analysis", "economic impact", "resource allocation"],
            "social": ["social implications", "community effects", "cultural considerations"],
            "environmental": ["environmental impact", "sustainability", "long-term effects"],
            "ethical": ["moral implications", "rights and responsibilities", "justice considerations"],
            "practical": ["implementation challenges", "feasibility", "real-world constraints"]
        }
    
    def get_opposition_summary(self) -> Dict[str, Any]:
        """Get summary of opposition strategy and progress"""
        if not self.opposition_context:
            return {"active": False}
        
        return {
            "active": True,
            "question": self.opposition_context.question,
            "position": self.opposition_context.position_taken,
            "current_round": self.opposition_context.current_round,
            "counter_arguments_made": len(self.opposition_context.counter_arguments_made),
            "weaknesses_identified": len(self.opposition_context.identified_weaknesses),
            "proponent_arguments_tracked": len(self.opposition_context.proponent_arguments)
        }
    
    async def reset_opposition_context(self):
        """Reset opposition context for new debate"""
        self.opposition_context = None
        self.clear_conversation_history(keep_system=True)
        logger.info(f"Opposition context reset for opponent agent {self.agent_id}")
