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

class ArgumentType(Enum):
    """Types of arguments that can be constructed"""
    DEDUCTIVE = "deductive"           # Logical conclusion from premises
    INDUCTIVE = "inductive"           # Probable conclusion from evidence
    ABDUCTIVE = "abductive"           # Best explanation for observations
    ANALOGICAL = "analogical"         # Similarity-based reasoning
    CAUSAL = "causal"                # Cause-and-effect relationships
    CONSEQUENTIALIST = "consequentialist"  # Based on outcomes
    PRECEDENT = "precedent"           # Based on historical examples
    EMPIRICAL = "empirical"           # Based on data and evidence

class ArgumentStrategy(Enum):
    """Strategic approaches to argumentation"""
    LOGICAL = "logical"               # Pure logical reasoning
    EMPIRICAL = "empirical"           # Data and evidence focused
    ETHICAL = "ethical"               # Moral and values-based
    PRACTICAL = "practical"           # Pragmatic and utilitarian
    EMOTIONAL = "emotional"           # Appeal to emotions and values
    AUTHORITY = "authority"           # Expert opinion and credibility
    CONSENSUS = "consensus"           # Common agreement and acceptance
    HISTORICAL = "historical"         # Lessons from the past

class DebateRole(Enum):
    """Role in the debate context"""
    OPENING = "opening"
    RESPONSE = "response"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"

@dataclass
class ArgumentPremise:
    """Individual premise supporting an argument"""
    statement: str
    evidence_type: str  # fact, study, example, expert_opinion, etc.
    credibility_score: float = 0.0
    source: Optional[str] = None
    verification_status: str = "unverified"  # verified, disputed, unverified

@dataclass
class ArgumentStructure:
    """Complete argument structure"""
    claim: str
    premises: List[ArgumentPremise] = field(default_factory=list)
    argument_type: ArgumentType = ArgumentType.INDUCTIVE
    strategy: ArgumentStrategy = ArgumentStrategy.LOGICAL
    strength_score: float = 0.0
    potential_weaknesses: List[str] = field(default_factory=list)
    rebuttals_anticipated: List[str] = field(default_factory=list)

@dataclass
class DebateContext:
    """Context for the ongoing debate"""
    question: str
    position_taken: str
    debate_history: List[Dict[str, Any]] = field(default_factory=list)
    opponent_arguments: List[str] = field(default_factory=list)
    key_points_made: List[str] = field(default_factory=list)
    current_round: int = 0
    argument_threads: Dict[str, List[str]] = field(default_factory=dict)
    concessions_made: List[str] = field(default_factory=list)

class ProponentAgent(BaseAgent):
    """
    Agent specialized in building strong affirmative arguments and supporting debate positions.
    
    Uses evidence-based reasoning, multiple argumentation strategies, and systematic
    approach to construct compelling cases for any given position.
    """
    
    def __init__(self, **kwargs):
        # Default configuration optimized for argumentation
        default_config = AgentConfig(
            model="llama3.1:8b",  # Larger model for complex reasoning
            temperature=0.7,      # Balanced creativity and consistency
            max_tokens=1024,      # Room for detailed arguments
            timeout=90,           # More time for complex reasoning
            enable_tools=True,    # Use tools for evidence gathering
            enable_memory=True,
            memory_limit=30       # Remember extensive debate context
        )
        
        # Merge with provided config
        config = kwargs.pop('config', default_config)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        super().__init__(config=config, **kwargs)
        
        # Proponent-specific state
        self.debate_context: Optional[DebateContext] = None
        self.argument_frameworks = self._initialize_argument_frameworks()
        self.persuasion_techniques = self._initialize_persuasion_techniques()
        self.fallacy_checklist = self._initialize_fallacy_checklist()
        
        logger.info(f"ProponentAgent initialized: {self.agent_id}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the proponent agent"""
        return """You are a skilled proponent agent specializing in building strong, evidence-based arguments. Your role is to:

## Primary Objectives:
1. **Construct compelling affirmative arguments** for the assigned position
2. **Support claims with credible evidence** and logical reasoning
3. **Anticipate and address counterarguments** proactively
4. **Maintain logical consistency** throughout the debate
5. **Present arguments persuasively** while remaining intellectually honest

## Argumentation Principles:
- **Evidence-based reasoning**: Always support claims with credible sources
- **Logical structure**: Use clear premise-conclusion relationships
- **Intellectual honesty**: Acknowledge limitations and uncertainties
- **Strategic thinking**: Build arguments that are difficult to refute
- **Progressive development**: Build upon previous points systematically

## Argument Types to Master:
- **Deductive**: Logical conclusions from accepted premises
- **Inductive**: Probable conclusions from patterns and evidence
- **Analogical**: Reasoning from similar cases or situations
- **Causal**: Demonstrating cause-and-effect relationships
- **Empirical**: Data-driven arguments with statistical support
- **Precedent**: Arguments based on historical examples

## Debate Strategies:
- **Opening**: Establish strong foundational arguments
- **Response**: Address opponent points while advancing your case
- **Rebuttal**: Systematically counter opposing arguments
- **Building**: Layer arguments to create cumulative strength

## Quality Standards:
- Avoid logical fallacies (ad hominem, straw man, false dichotomy, etc.)
- Use credible, recent sources when possible
- Acknowledge when evidence is mixed or uncertain
- Distinguish between facts, interpretations, and opinions
- Present strongest arguments first, then supporting points

## Persuasion Techniques:
- Use concrete examples and case studies
- Appeal to shared values and common ground
- Present evidence in compelling, accessible ways
- Use rhetorical devices appropriately (but not manipulatively)
- Structure arguments for maximum clarity and impact

Remember: Your goal is to make the strongest possible case for your position while maintaining intellectual integrity and respect for rational discourse."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request for argument generation"""
        try:
            # Determine the type of argument needed
            if context and context.get("role") == "opening_statement":
                return await self.generate_opening_statement(request, context)
            elif context and context.get("role") == "debate_response":
                return await self.generate_debate_response(request, context)
            elif context and context.get("role") == "rebuttal":
                return await self.generate_rebuttal(request, context)
            else:
                # Default: general argument generation
                return await self.generate_argument(request, context)
                
        except Exception as e:
            logger.error(f"Error in proponent process_request: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error while constructing the argument. Let me try a different approach.",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def generate_opening_statement(self, 
                                       question: str, 
                                       context: Dict[str, Any] = None,
                                       tools_enabled: bool = None,
                                       tool_manager = None) -> Dict[str, Any]:
        """
        Generate a strong opening statement for the debate.
        
        Args:
            question: The debate question/topic
            context: Additional context and parameters
            tools_enabled: Whether to use external tools
            tool_manager: Tool manager for research
            
        Returns:
            Dictionary with opening statement and supporting information
        """
        try:
            # Initialize or update debate context
            if not self.debate_context:
                self.debate_context = DebateContext(
                    question=question,
                    position_taken="affirmative"
                )
            
            # Set tool manager if provided
            if tool_manager:
                self.set_tool_manager(tool_manager)
            
            # Analyze the question to understand what we're arguing for
            question_analysis = await self._analyze_debate_question(question)
            
            # Research if tools are enabled
            research_findings = None
            if tools_enabled and self.tool_manager:
                research_findings = await self._conduct_research(question, question_analysis)
            
            # Construct the opening argument
            argument_structure = await self._build_opening_argument(
                question, question_analysis, research_findings
            )
            
            # Generate the actual statement
            statement_response = await self._generate_structured_argument(
                argument_structure, DebateRole.OPENING
            )
            
            # Extract key information
            statement = statement_response.content
            reasoning = statement_response.reasoning or ""
            sources = research_findings.get("sources", []) if research_findings else []
            
            # Update debate context
            self.debate_context.key_points_made.extend(
                argument_structure.premises[:3]  # Track main premises
            )
            self.debate_context.current_round = 1
            
            return {
                "statement": statement,
                "reasoning": reasoning,
                "key_points": [premise.statement for premise in argument_structure.premises],
                "sources": sources,
                "argument_type": argument_structure.argument_type.value,
                "strategy": argument_structure.strategy.value,
                "strength_score": argument_structure.strength_score,
                "anticipated_rebuttals": argument_structure.rebuttals_anticipated
            }
            
        except Exception as e:
            logger.error(f"Error generating opening statement: {e}")
            return {
                "statement": f"I believe the affirmative position on '{question}' has merit and I'll work to demonstrate why this perspective deserves serious consideration.",
                "reasoning": "Error in detailed analysis, providing basic opening",
                "key_points": [],
                "sources": [],
                "argument_type": "inductive",
                "strategy": "logical"
            }
    
    async def generate_debate_response(self,
                                     question: str,
                                     debate_history: List[Dict[str, Any]],
                                     round_number: int,
                                     tools_enabled: bool = None,
                                     tool_manager = None) -> Dict[str, Any]:
        """
        Generate a response during the main debate rounds.
        
        Args:
            question: The debate question
            debate_history: Previous exchanges in the debate
            round_number: Current round number
            tools_enabled: Whether to use external tools
            tool_manager: Tool manager for research
            
        Returns:
            Dictionary with debate response and supporting information
        """
        try:
            # Update debate context
            if not self.debate_context:
                self.debate_context = DebateContext(
                    question=question,
                    position_taken="affirmative"
                )
            
            self.debate_context.current_round = round_number
            self._update_debate_context_from_history(debate_history)
            
            # Set tool manager if provided
            if tool_manager:
                self.set_tool_manager(tool_manager)
            
            # Analyze opponent's latest arguments
            opponent_analysis = await self._analyze_opponent_arguments(debate_history)
            
            # Determine response strategy
            response_strategy = await self._determine_response_strategy(
                opponent_analysis, round_number
            )
            
            # Research if needed and tools available
            additional_research = None
            if tools_enabled and self.tool_manager and opponent_analysis.get("claims_to_verify"):
                additional_research = await self._verify_opponent_claims(
                    opponent_analysis["claims_to_verify"]
                )
            
            # Build response argument
            response_argument = await self._build_response_argument(
                opponent_analysis, response_strategy, additional_research
            )
            
            # Generate the response
            response = await self._generate_structured_argument(
                response_argument, DebateRole.RESPONSE
            )
            
            # Extract sources from research
            sources = []
            if additional_research:
                sources.extend(additional_research.get("sources", []))
            
            # Update debate tracking
            self.debate_context.key_points_made.extend([
                premise.statement for premise in response_argument.premises[:2]
            ])
            
            return {
                "response": response.content,
                "key_points": [premise.statement for premise in response_argument.premises],
                "sources": sources,
                "strategy_used": response_strategy,
                "opponent_points_addressed": opponent_analysis.get("main_points", []),
                "strength_score": response_argument.strength_score
            }
            
        except Exception as e:
            logger.error(f"Error generating debate response: {e}")
            return {
                "response": "Let me build upon my previous arguments while addressing the points raised by my opponent.",
                "key_points": [],
                "sources": [],
                "strategy_used": "logical",
                "opponent_points_addressed": []
            }
    
    async def generate_argument(self, topic: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Generate a general argument for a topic"""
        try:
            # Create a simple argument structure
            prompt = f"""
            I need to construct a strong affirmative argument for the following topic: {topic}
            
            Please provide:
            1. A clear main claim
            2. 2-3 supporting premises with reasoning
            3. Evidence or examples where applicable
            4. Acknowledgment of potential counterarguments
            
            Structure the argument logically and persuasively.
            """
            
            # Add context if provided
            if context:
                prompt += f"\n\nAdditional context: {context}"
            
            response = await self.chat(prompt)
            
            # Enhance with argument analysis
            key_points = self._extract_argument_points(response.content)
            response.key_points = key_points
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating general argument: {e}")
            raise
    
    async def generate_rebuttal(self, opponent_argument: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Generate a rebuttal to an opponent's argument"""
        try:
            # Analyze the opponent's argument
            analysis = await self._quick_argument_analysis(opponent_argument)
            
            prompt = f"""
            I need to construct a strong rebuttal to the following argument: {opponent_argument}
            
            Based on my analysis, I should address:
            - Logical weaknesses or fallacies
            - Unsupported claims or weak evidence
            - Alternative interpretations
            - Counterexamples or contradictory evidence
            
            Provide a respectful but firm rebuttal that maintains my affirmative position.
            """
            
            response = await self.chat(prompt, context)
            
            # Add rebuttal-specific metadata
            response.metadata.update({
                "rebuttal_targets": analysis.get("weaknesses", []),
                "argument_type": "rebuttal"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating rebuttal: {e}")
            raise
    
    async def _analyze_debate_question(self, question: str) -> Dict[str, Any]:
        """Analyze the debate question to understand structure and implications"""
        try:
            # Extract key concepts and identify question type
            key_concepts = extract_key_concepts(question, max_concepts=8)
            
            # Identify question patterns
            question_lower = question.lower()
            question_patterns = {
                "should": ["should", "ought to", "must"],
                "comparison": ["better than", "worse than", "versus", "compared to"],
                "causal": ["causes", "leads to", "results in", "because of"],
                "evaluative": ["good", "bad", "effective", "successful", "beneficial"],
                "policy": ["government", "law", "policy", "regulation", "ban"],
                "ethical": ["right", "wrong", "moral", "ethical", "just", "fair"]
            }
            
            identified_types = []
            for pattern_type, keywords in question_patterns.items():
                if any(keyword in question_lower for keyword in keywords):
                    identified_types.append(pattern_type)
            
            # Identify scope and complexity
            word_count = len(question.split())
            complexity = "high" if word_count > 20 else "medium" if word_count > 10 else "low"
            
            # Identify potential argument angles
            argument_angles = self._identify_argument_angles(question, key_concepts)
            
            return {
                "key_concepts": key_concepts,
                "question_types": identified_types,
                "complexity": complexity,
                "argument_angles": argument_angles,
                "main_claim_direction": self._extract_affirmative_position(question)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing debate question: {e}")
            return {
                "key_concepts": [],
                "question_types": ["general"],
                "complexity": "medium",
                "argument_angles": ["logical", "practical"],
                "main_claim_direction": "affirmative"
            }
    
    def _identify_argument_angles(self, question: str, key_concepts: List[str]) -> List[str]:
        """Identify potential angles for argumentation"""
        angles = []
        
        question_lower = question.lower()
        
        # Check for different argument opportunities
        if any(word in question_lower for word in ["benefit", "advantage", "improve", "help"]):
            angles.append("benefits")
        
        if any(word in question_lower for word in ["cost", "expensive", "afford", "economic"]):
            angles.append("economic")
        
        if any(word in question_lower for word in ["right", "wrong", "moral", "ethical"]):
            angles.append("ethical")
        
        if any(word in question_lower for word in ["evidence", "study", "research", "data"]):
            angles.append("empirical")
        
        if any(word in question_lower for word in ["history", "past", "tradition", "precedent"]):
            angles.append("historical")
        
        if any(word in question_lower for word in ["practical", "feasible", "workable", "realistic"]):
            angles.append("practical")
        
        # Default angles if none identified
        if not angles:
            angles = ["logical", "practical", "empirical"]
        
        return angles[:4]  # Limit to top 4 angles
    
    def _extract_affirmative_position(self, question: str) -> str:
        """Extract what the affirmative position should argue for"""
        # Simple heuristics to understand the affirmative stance
        question_lower = question.lower()
        
        if question_lower.startswith("should"):
            return "yes, this action should be taken"
        elif "better than" in question_lower:
            # Extract the first option as what we're arguing for
            parts = question.split("better than")
            if len(parts) > 1:
                return f"the first option is indeed better"
        elif "?" in question:
            return "affirmative answer to the question"
        else:
            return "support for the stated position"
    
    async def _conduct_research(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct research using available tools"""
        research_results = {"sources": [], "evidence": [], "statistics": []}
        
        try:
            if not self.tool_manager:
                return research_results
            
            # Generate search queries based on key concepts
            key_concepts = analysis.get("key_concepts", [])
            search_queries = []
            
            # Create targeted search queries
            for concept in key_concepts[:3]:  # Top 3 concepts
                search_queries.append(f"{concept} benefits evidence")
                search_queries.append(f"{concept} research studies")
            
            # Add question-specific searches
            if "should" in question.lower():
                search_queries.append(f"why {question.replace('Should', '').replace('should', '').strip()}")
            
            # Perform searches
            for query in search_queries[:3]:  # Limit to 3 searches
                try:
                    search_results = await self.tool_manager.web_search(
                        query, max_results=3,
                        agent_id=self.agent_id,
                        session_id=self.session_id
                    )
                    
                    for result in search_results:
                        research_results["sources"].append({
                            "title": result.title,
                            "url": result.url,
                            "snippet": result.snippet,
                            "relevance": result.relevance_score,
                            "query": query
                        })
                        
                        # Extract potential evidence from snippets
                        if any(indicator in result.snippet.lower() for indicator in 
                               ["study", "research", "found", "shows", "evidence", "data"]):
                            research_results["evidence"].append(result.snippet)
                            
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            # Fact-check key claims if possible
            potential_claims = self._extract_factual_claims(question)
            for claim in potential_claims[:2]:  # Limit fact-checks
                try:
                    fact_check = await self.tool_manager.fact_check(
                        claim,
                        agent_id=self.agent_id,
                        session_id=self.session_id
                    )
                    
                    if fact_check.verdict in ["true", "mixed"]:
                        research_results["evidence"].append(fact_check.explanation)
                        
                except Exception as e:
                    logger.warning(f"Fact-check failed for claim '{claim}': {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error conducting research: {e}")
        
        return research_results
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims that can be verified"""
        # Simple extraction of potential factual claims
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences with numbers, statistics, or definitive statements
            if (len(sentence) > 20 and 
                (re.search(r'\d+%', sentence) or  # Percentages
                 re.search(r'\d+', sentence) or   # Numbers
                 any(word in sentence.lower() for word in ["studies show", "research", "evidence"]))):
                claims.append(sentence)
        
        return claims[:3]  # Limit to 3 claims
    
    async def _build_opening_argument(self, 
                                    question: str, 
                                    analysis: Dict[str, Any], 
                                    research: Optional[Dict[str, Any]]) -> ArgumentStructure:
        """Build a structured opening argument"""
        try:
            # Determine primary argument strategy
            question_types = analysis.get("question_types", [])
            if "ethical" in question_types:
                primary_strategy = ArgumentStrategy.ETHICAL
            elif "policy" in question_types:
                primary_strategy = ArgumentStrategy.PRACTICAL
            elif research and research.get("evidence"):
                primary_strategy = ArgumentStrategy.EMPIRICAL
            else:
                primary_strategy = ArgumentStrategy.LOGICAL
            
            # Create main claim
            main_claim = self._formulate_main_claim(question, analysis)
            
            # Build supporting premises
            premises = []
            
            # Add evidence-based premises if research available
            if research and research.get("evidence"):
                for evidence in research["evidence"][:2]:  # Top 2 pieces of evidence
                    premise = ArgumentPremise(
                        statement=f"Research evidence supports this position: {evidence[:100]}...",
                        evidence_type="research",
                        credibility_score=0.8,
                        verification_status="researched"
                    )
                    premises.append(premise)
            
            # Add logical premises
            logical_premises = self._generate_logical_premises(question, analysis)
            premises.extend(logical_premises)
            
            # Calculate argument strength
            strength_score = self._calculate_argument_strength(premises, research)
            
            # Anticipate potential rebuttals
            anticipated_rebuttals = self._anticipate_rebuttals(question, premises)
            
            return ArgumentStructure(
                claim=main_claim,
                premises=premises,
                argument_type=ArgumentType.INDUCTIVE,  # Most flexible for opening
                strategy=primary_strategy,
                strength_score=strength_score,
                rebuttals_anticipated=anticipated_rebuttals
            )
            
        except Exception as e:
            logger.error(f"Error building opening argument: {e}")
            # Return basic argument structure
            return ArgumentStructure(
                claim=f"I argue in favor of the affirmative position on: {question}",
                premises=[
                    ArgumentPremise(
                        statement="This position has logical merit that deserves consideration",
                        evidence_type="logical",
                        credibility_score=0.5
                    )
                ],
                strength_score=0.5
            )
    
    def _formulate_main_claim(self, question: str, analysis: Dict[str, Any]) -> str:
        """Formulate the main claim for the argument"""
        direction = analysis.get("main_claim_direction", "affirmative")
        
        if question.lower().startswith("should"):
            # Remove "should" and affirm the action
            action = question[6:].strip().rstrip("?")
            return f"Yes, {action} and here's why this is the right course of action."
        elif "better than" in question.lower():
            parts = question.split("better than")
            if len(parts) > 1:
                first_option = parts[0].replace("Is", "").replace("is", "").strip()
                return f"I argue that {first_option} is indeed the superior choice."
        else:
            return f"I support the affirmative position on this question: {question}"
    
    def _generate_logical_premises(self, question: str, analysis: Dict[str, Any]) -> List[ArgumentPremise]:
        """Generate logical premises based on question analysis"""
        premises = []
        key_concepts = analysis.get("key_concepts", [])
        
        # Create premises based on argument angles
        argument_angles = analysis.get("argument_angles", [])
        
        if "benefits" in argument_angles:
            premises.append(ArgumentPremise(
                statement=f"The proposed approach offers significant benefits including improved outcomes for stakeholders",
                evidence_type="logical",
                credibility_score=0.7
            ))
        
        if "practical" in argument_angles:
            premises.append(ArgumentPremise(
                statement="This position is not only theoretically sound but also practically implementable",
                evidence_type="practical",
                credibility_score=0.6
            ))
        
        if "ethical" in argument_angles:
            premises.append(ArgumentPremise(
                statement="From an ethical standpoint, this position aligns with fundamental principles of fairness and justice",
                evidence_type="ethical",
                credibility_score=0.7
            ))
        
        # Ensure we have at least 2 premises
        if len(premises) < 2:
            premises.append(ArgumentPremise(
                statement="The weight of reasoning and available evidence supports this conclusion",
                evidence_type="logical",
                credibility_score=0.6
            ))
        
        return premises[:3]  # Limit to 3 premises for clarity
    
    def _calculate_argument_strength(self, premises: List[ArgumentPremise], research: Optional[Dict[str, Any]]) -> float:
        """Calculate overall argument strength score"""
        if not premises:
            return 0.0
        
        # Base score from premise credibility
        premise_score = sum(premise.credibility_score for premise in premises) / len(premises)
        
        # Bonus for research support
        research_bonus = 0.2 if research and research.get("evidence") else 0.0
        
        # Bonus for multiple premise types
        evidence_types = set(premise.evidence_type for premise in premises)
        diversity_bonus = 0.1 * (len(evidence_types) - 1)
        
        return min(1.0, premise_score + research_bonus + diversity_bonus)
    
    def _anticipate_rebuttals(self, question: str, premises: List[ArgumentPremise]) -> List[str]:
        """Anticipate potential rebuttals to the argument"""
        rebuttals = []
        
        # Generic rebuttals based on question type
        question_lower = question.lower()
        
        if "should" in question_lower:
            rebuttals.append("Opponent may argue about negative consequences or implementation challenges")
        
        if "better than" in question_lower:
            rebuttals.append("Opponent will likely highlight advantages of the alternative option")
        
        if any(word in question_lower for word in ["cost", "expensive"]):
            rebuttals.append("Economic arguments about cost-effectiveness may be challenged")
        
        # Premise-specific rebuttals
        for premise in premises:
            if premise.evidence_type == "research" and premise.credibility_score < 0.9:
                rebuttals.append("Research credibility or methodology may be questioned")
            elif premise.evidence_type == "logical":
                rebuttals.append("Logical assumptions may be challenged")
        
        return rebuttals[:3]  # Limit to top 3 anticipated rebuttals
    
    async def _generate_structured_argument(self, argument: ArgumentStructure, role: DebateRole) -> AgentResponse:
        """Generate the actual argument text from the structure"""
        try:
            # Create structured prompt based on role
            if role == DebateRole.OPENING:
                prompt = f"""
                Generate a compelling opening statement that argues: {argument.claim}
                
                Structure the argument with:
                1. Strong opening that establishes the position
                2. Main supporting points: {'; '.join([p.statement for p in argument.premises])}
                3. Clear logical flow and transitions
                4. Confident but respectful tone
                
                Strategy: {argument.strategy.value}
                Expected strength: {argument.strength_score:.1f}/1.0
                
                Make the argument persuasive and well-reasoned, approximately 200-300 words.
                """
            else:
                prompt = f"""
                Generate a strong debate response that argues: {argument.claim}
                
                Key points to make: {'; '.join([p.statement for p in argument.premises])}
                Strategy: {argument.strategy.value}
                
                Build upon previous arguments while addressing new points.
                """
            
            response = await self.chat(prompt)
            
            # Enhance response with argument metadata
            response.reasoning = f"Using {argument.strategy.value} strategy with {argument.argument_type.value} reasoning"
            response.confidence = argument.strength_score
            response.metadata.update({
                "argument_type": argument.argument_type.value,
                "strategy": argument.strategy.value,
                "premise_count": len(argument.premises),
                "anticipated_rebuttals": argument.rebuttals_anticipated
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured argument: {e}")
            return AgentResponse(
                content=f"I support the position that {argument.claim}",
                confidence=0.5,
                reasoning="Basic argument due to generation error"
            )
    
    def _update_debate_context_from_history(self, debate_history: List[Dict[str, Any]]):
        """Update internal debate context from history"""
        if not self.debate_context:
            return
        
        # Clear and rebuild from history
        self.debate_context.debate_history = debate_history
        self.debate_context.opponent_arguments.clear()
        
        # Extract opponent arguments
        for entry in debate_history:
            if entry.get("agent") == "opponent":
                self.debate_context.opponent_arguments.append(entry.get("content", ""))
    
    async def _analyze_opponent_arguments(self, debate_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze opponent's arguments to inform response strategy"""
        try:
            # Get opponent's latest arguments
            opponent_entries = [entry for entry in debate_history if entry.get("agent") == "opponent"]
            
            if not opponent_entries:
                return {"main_points": [], "weaknesses": [], "claims_to_verify": []}
            
            latest_opponent = opponent_entries[-1]
            opponent_content = latest_opponent.get("content", "")
            
            # Extract main points
            main_points = self._extract_argument_points(opponent_content)
            
            # Identify potential weaknesses
            weaknesses = self._identify_argument_weaknesses(opponent_content)
            
            # Extract claims that could be fact-checked
            claims_to_verify = self._extract_factual_claims(opponent_content)
            
            return {
                "main_points": main_points,
                "weaknesses": weaknesses,
                "claims_to_verify": claims_to_verify,
                "argument_style": self._classify_argument_style(opponent_content)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing opponent arguments: {e}")
            return {"main_points": [], "weaknesses": [], "claims_to_verify": []}
    
    def _extract_argument_points(self, content: str) -> List[str]:
        """Extract main argument points from content"""
        points = []
        
        # Look for structured points (numbered, bulleted)
        bullet_pattern = r'(?:^|\n)[-•*]\s*(.+)'
        bullets = re.findall(bullet_pattern, content, re.MULTILINE)
        points.extend(bullets)
        
        numbered_pattern = r'(?:^|\n)\d+\.\s*(.+)'
        numbered = re.findall(numbered_pattern, content, re.MULTILINE)
        points.extend(numbered)
        
        # If no structured points, extract key sentences
        if not points:
            sentences = re.split(r'[.!?]+', content)
            # Look for sentences with argument indicators
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 20 and 
                    any(indicator in sentence.lower() for indicator in 
                        ["because", "therefore", "since", "as a result", "this shows", "evidence"])):
                    points.append(sentence)
        
        return points[:5]  # Limit to top 5 points
    
    def _identify_argument_weaknesses(self, content: str) -> List[str]:
        """Identify potential weaknesses in opponent's argument"""
        weaknesses = []
        
        # Check for common logical issues
        content_lower = content.lower()
        
        # Overgeneralization
        if any(word in content_lower for word in ["all", "every", "never", "always", "everyone"]):
            weaknesses.append("potential overgeneralization")
        
        # Unsupported claims
        if "studies show" in content_lower or "research proves" in content_lower:
            if not any(source in content_lower for source in ["university", "journal", "study by"]):
                weaknesses.append("unsupported research claims")
        
        # Emotional appeals without logic
        emotional_words = ["terrible", "disaster", "amazing", "perfect", "outrageous"]
        if sum(1 for word in emotional_words if word in content_lower) > 2:
            weaknesses.append("heavy reliance on emotional appeals")
        
        # False dichotomies
        if any(phrase in content_lower for phrase in ["only way", "must choose", "either", "no other option"]):
            weaknesses.append("possible false dichotomy")
        
        return weaknesses
    
    def _classify_argument_style(self, content: str) -> str:
        """Classify the opponent's argument style"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["data", "statistics", "study", "research", "evidence"]):
            return "empirical"
        elif any(word in content_lower for word in ["moral", "ethical", "right", "wrong", "should"]):
            return "ethical"
        elif any(word in content_lower for word in ["practical", "feasible", "cost", "benefit", "efficient"]):
            return "practical"
        elif any(word in content_lower for word in ["feel", "believe", "values", "important"]):
            return "emotional"
        else:
            return "logical"
    
    async def _determine_response_strategy(self, opponent_analysis: Dict[str, Any], round_number: int) -> str:
        """Determine the best response strategy"""
        weaknesses = opponent_analysis.get("weaknesses", [])
        opponent_style = opponent_analysis.get("argument_style", "logical")
        
        # Counter their strength with our strength
        if opponent_style == "empirical" and weaknesses:
            return "challenge_evidence"
        elif opponent_style == "emotional":
            return "logical_counter"
        elif opponent_style == "practical":
            return "broader_perspective"
        elif round_number <= 2:
            return "build_foundation"
        else:
            return "consolidate_position"
    
    async def _verify_opponent_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Verify opponent's factual claims using tools"""
        verification_results = {"sources": [], "contradictions": [], "confirmations": []}
        
        try:
            if not self.tool_manager:
                return verification_results
            
            for claim in claims[:2]:  # Limit to 2 claims
                try:
                    fact_check = await self.tool_manager.fact_check(
                        claim,
                        agent_id=self.agent_id,
                        session_id=self.session_id
                    )
                    
                    if fact_check.verdict == "false":
                        verification_results["contradictions"].append({
                            "claim": claim,
                            "verdict": fact_check.verdict,
                            "explanation": fact_check.explanation
                        })
                    elif fact_check.verdict == "true":
                        verification_results["confirmations"].append({
                            "claim": claim,
                            "verdict": fact_check.verdict
                        })
                        
                    verification_results["sources"].extend(fact_check.sources)
                    
                except Exception as e:
                    logger.warning(f"Fact-check failed for claim: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error verifying opponent claims: {e}")
        
        return verification_results
    
    async def _build_response_argument(self, 
                                     opponent_analysis: Dict[str, Any], 
                                     strategy: str,
                                     research: Optional[Dict[str, Any]]) -> ArgumentStructure:
        """Build argument structure for response"""
        try:
            premises = []
            
            # Address opponent weaknesses
            weaknesses = opponent_analysis.get("weaknesses", [])
            for weakness in weaknesses[:2]:
                premises.append(ArgumentPremise(
                    statement=f"The opposing argument shows {weakness} which undermines its credibility",
                    evidence_type="logical",
                    credibility_score=0.7
                ))
            
            # Add contradictory evidence if found
            if research and research.get("contradictions"):
                for contradiction in research["contradictions"][:1]:
                    premises.append(ArgumentPremise(
                        statement=f"Fact-checking reveals: {contradiction['explanation'][:100]}...",
                        evidence_type="fact_check",
                        credibility_score=0.9,
                        verification_status="verified"
                    ))
            
            # Build positive case
            if strategy == "build_foundation":
                premises.append(ArgumentPremise(
                    statement="Let me strengthen my position with additional supporting evidence",
                    evidence_type="logical",
                    credibility_score=0.8
                ))
            
            # Ensure minimum premises
            if len(premises) < 2:
                premises.append(ArgumentPremise(
                    statement="My previous arguments remain strong and unrefuted by the opposition",
                    evidence_type="logical",
                    credibility_score=0.6
                ))
            
            main_claim = "My position is strengthened by addressing these points while the opposition's arguments show significant weaknesses"
            
            return ArgumentStructure(
                claim=main_claim,
                premises=premises,
                strategy=ArgumentStrategy.LOGICAL,
                strength_score=self._calculate_argument_strength(premises, research)
            )
            
        except Exception as e:
            logger.error(f"Error building response argument: {e}")
            return ArgumentStructure(
                claim="I maintain my position and will address the points raised",
                premises=[ArgumentPremise(
                    statement="My arguments remain valid",
                    evidence_type="logical",
                    credibility_score=0.5
                )],
                strength_score=0.5
            )
    
    async def _quick_argument_analysis(self, argument: str) -> Dict[str, Any]:
        """Quick analysis of an argument for rebuttal purposes"""
        return {
            "weaknesses": self._identify_argument_weaknesses(argument),
            "main_claims": self._extract_factual_claims(argument),
            "style": self._classify_argument_style(argument)
        }
    
    def _initialize_argument_frameworks(self) -> Dict[str, List[str]]:
        """Initialize argument frameworks and templates"""
        return {
            "deductive": [
                "Major premise: {general_principle}",
                "Minor premise: {specific_case}",
                "Conclusion: {logical_result}"
            ],
            "inductive": [
                "Evidence: {supporting_examples}",
                "Pattern: {observed_pattern}",
                "Conclusion: {probable_result}"
            ],
            "causal": [
                "Cause: {identified_cause}",
                "Mechanism: {causal_mechanism}",
                "Effect: {resulting_effect}"
            ]
        }
    
    def _initialize_persuasion_techniques(self) -> Dict[str, str]:
        """Initialize persuasion techniques"""
        return {
            "ethos": "Establish credibility and trustworthiness",
            "pathos": "Appeal to emotions and values",
            "logos": "Use logical reasoning and evidence",
            "reciprocity": "Highlight mutual benefits",
            "scarcity": "Emphasize urgency or limited opportunities",
            "consensus": "Show widespread agreement or support"
        }
    
    def _initialize_fallacy_checklist(self) -> List[str]:
        """Initialize logical fallacy checklist"""
        return [
            "ad_hominem", "straw_man", "false_dichotomy", "slippery_slope",
            "appeal_to_authority", "appeal_to_emotion", "circular_reasoning",
            "hasty_generalization", "red_herring", "false_cause"
        ]
    
    def get_debate_summary(self) -> Dict[str, Any]:
        """Get summary of current debate state"""
        if not self.debate_context:
            return {"active": False}
        
        return {
            "active": True,
            "question": self.debate_context.question,
            "position": self.debate_context.position_taken,
            "current_round": self.debate_context.current_round,
            "key_points_made": len(self.debate_context.key_points_made),
            "opponent_arguments_tracked": len(self.debate_context.opponent_arguments)
        }
    
    async def reset_debate_context(self):
        """Reset debate context for new debate"""
        self.debate_context = None
        self.clear_conversation_history(keep_system=True)
        logger.info(f"Debate context reset for proponent agent {self.agent_id}")
