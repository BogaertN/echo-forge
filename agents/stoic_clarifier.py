import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents.base_agent import BaseAgent, AgentResponse, AgentConfig, ConversationMessage, MessageRole
from utils import (
    clean_text, extract_key_concepts, calculate_similarity, 
    get_logger, generate_uuid
)

logger = get_logger(__name__)

class ClarificationStage(Enum):
    """Stages of the clarification process"""
    INITIAL_UNDERSTANDING = "initial_understanding"
    ASSUMPTION_EXPLORATION = "assumption_exploration"
    DEFINITION_CLARIFICATION = "definition_clarification"
    CONTEXT_GATHERING = "context_gathering"
    SCOPE_REFINEMENT = "scope_refinement"
    GOAL_CLARIFICATION = "goal_clarification"
    COMPLETION_ASSESSMENT = "completion_assessment"

class QuestionType(Enum):
    """Types of Socratic questions"""
    CLARIFICATION = "clarification"          # "What do you mean by...?"
    ASSUMPTION = "assumption"                # "What assumptions are you making?"
    EVIDENCE = "evidence"                    # "What evidence supports this?"
    PERSPECTIVE = "perspective"              # "How might others view this?"
    IMPLICATION = "implication"              # "What are the implications?"
    META_QUESTION = "meta_question"          # "Why is this question important?"
    DEFINITION = "definition"                # "How would you define...?"
    EXAMPLE = "example"                      # "Can you give me an example?"

@dataclass
class ClarificationContext:
    """Context for the clarification process"""
    original_question: str
    current_understanding: str = ""
    identified_assumptions: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)
    refinement_history: List[str] = field(default_factory=list)
    stage: ClarificationStage = ClarificationStage.INITIAL_UNDERSTANDING
    rounds_completed: int = 0
    max_rounds: int = 5
    completion_signals: List[str] = field(default_factory=list)

@dataclass
class SocraticQuestion:
    """Structure for a Socratic question"""
    question: str
    question_type: QuestionType
    purpose: str
    follow_up_hints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    expected_outcome: str = ""

class StoicClarifier(BaseAgent):
    """
    Socratic clarification agent that helps users refine and deepen their questions
    through systematic philosophical questioning techniques.
    """
    
    def __init__(self, **kwargs):
        # Default configuration optimized for clarification
        default_config = AgentConfig(
            model="gemma2:2b",  # Smaller model for focused questioning
            temperature=0.6,    # Balanced creativity and consistency
            max_tokens=512,     # Shorter, focused responses
            timeout=45,
            enable_tools=False, # Pure reasoning, no external tools
            enable_memory=True,
            memory_limit=20     # Keep focused conversation history
        )
        
        # Merge with provided config
        config = kwargs.pop('config', default_config)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        super().__init__(config=config, **kwargs)
        
        # Clarification-specific state
        self.clarification_context: Optional[ClarificationContext] = None
        self.question_strategies = self._initialize_question_strategies()
        self.completion_criteria = self._initialize_completion_criteria()
        
        logger.info(f"StoicClarifier agent initialized: {self.agent_id}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Socratic clarifier"""
        return """You are a Socratic clarification agent designed to help users refine and deepen their questions through philosophical questioning. Your role is to:

1. **Guide users through systematic questioning** to uncover the true nature of their inquiry
2. **Challenge assumptions** gently but persistently to reveal underlying beliefs
3. **Clarify definitions** of key terms and concepts to ensure shared understanding
4. **Explore context** to understand the situational factors affecting the question
5. **Refine scope** to make questions more precise and answerable
6. **Never provide direct answers** - your job is to ask better questions, not to solve problems

## Your Approach:
- Ask ONE focused question at a time
- Build on previous responses to go deeper
- Use the Socratic method: question assumptions, definitions, implications
- Be patient, curious, and genuinely interested in understanding
- Help users discover insights through their own reasoning
- Recognize when clarification is sufficient for productive debate

## Question Types to Use:
- **Clarification**: "What do you mean when you say...?"
- **Assumptions**: "What are you assuming about...?"
- **Evidence**: "What leads you to think...?"
- **Perspective**: "How might someone who disagrees view this?"
- **Implications**: "If that's true, what would it mean for...?"
- **Definitions**: "How would you define...?"
- **Examples**: "Can you give me a specific example of...?"
- **Meta-questions**: "Why is this question important to you?"

## Signs of Completion:
- User has clearly defined key terms
- Main assumptions have been identified and examined
- Context and scope are well-understood
- Question is specific and focused enough for productive debate
- User shows confidence in their refined question

Remember: You are helping them think more clearly, not thinking for them. Be the midwife of ideas, not the parent."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a clarification request"""
        try:
            # Parse the request type
            if context and context.get("action") == "start_clarification":
                return await self.start_clarification(request, context)
            elif context and context.get("action") == "continue_clarification":
                return await self.continue_clarification(request, context)
            else:
                # Default: treat as start of clarification
                return await self.start_clarification(request, context)
                
        except Exception as e:
            logger.error(f"Error in clarification process: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error in the clarification process. Let's try again.",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def start_clarification(self, question: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Start the clarification process with an initial question.
        
        Args:
            question: The user's initial question
            context: Additional context information
            
        Returns:
            AgentResponse with first clarifying question
        """
        try:
            # Initialize clarification context
            self.clarification_context = ClarificationContext(
                original_question=clean_text(question),
                max_rounds=context.get("max_rounds", 5) if context else 5
            )
            
            # Clear any previous conversation for fresh start
            self.clear_conversation_history(keep_system=True)
            
            # Analyze the initial question
            analysis = await self._analyze_initial_question(question)
            
            # Generate first clarifying question
            first_question = await self._generate_first_question(question, analysis)
            
            # Update context
            self.clarification_context.rounds_completed = 1
            self.clarification_context.current_understanding = analysis.get("understanding", "")
            self.clarification_context.key_terms.extend(analysis.get("key_terms", []))
            
            return AgentResponse(
                content=first_question.question,
                confidence=0.8,
                reasoning=first_question.purpose,
                key_points=first_question.follow_up_hints,
                metadata={
                    "question_type": first_question.question_type.value,
                    "stage": self.clarification_context.stage.value,
                    "round": self.clarification_context.rounds_completed,
                    "analysis": analysis,
                    "examples": first_question.examples,
                    "clarification_complete": False
                }
            )
            
        except Exception as e:
            logger.error(f"Error starting clarification: {e}")
            raise
    
    async def continue_clarification(self, response: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Continue the clarification process with user response.
        
        Args:
            response: User's response to previous question
            context: Additional context information
            
        Returns:
            AgentResponse with next question or completion signal
        """
        try:
            if not self.clarification_context:
                # Context was lost, restart
                return await self.start_clarification(response, context)
            
            # Process the user's response
            await self._process_user_response(response)
            
            # Check if clarification is complete
            completion_check = await self._assess_completion()
            
            if completion_check["complete"]:
                # Generate final clarified question
                final_question = await self._generate_final_question()
                
                return AgentResponse(
                    content=f"Excellent! Based on our discussion, here's your refined question:\n\n\"{final_question}\"\n\nThis question is now clear, focused, and ready for productive debate. You've identified the key assumptions, defined important terms, and established the context for meaningful discussion.",
                    confidence=0.9,
                    key_points=completion_check["summary_points"],
                    metadata={
                        "clarification_complete": True,
                        "final_question": final_question,
                        "original_question": self.clarification_context.original_question,
                        "rounds_completed": self.clarification_context.rounds_completed,
                        "key_concepts": self.clarification_context.key_terms,
                        "assumptions_identified": self.clarification_context.identified_assumptions,
                        "context_factors": self.clarification_context.context_factors,
                        "completion_reason": completion_check["reason"]
                    }
                )
            
            # Generate next question
            next_question = await self._generate_next_question()
            
            self.clarification_context.rounds_completed += 1
            
            return AgentResponse(
                content=next_question.question,
                confidence=0.8,
                reasoning=next_question.purpose,
                key_points=next_question.follow_up_hints,
                metadata={
                    "question_type": next_question.question_type.value,
                    "stage": self.clarification_context.stage.value,
                    "round": self.clarification_context.rounds_completed,
                    "examples": next_question.examples,
                    "clarification_complete": False,
                    "progress": self._get_progress_summary()
                }
            )
            
        except Exception as e:
            logger.error(f"Error continuing clarification: {e}")
            raise
    
    async def get_clarification_question(self, question: str, history: List[Dict] = None) -> Dict[str, Any]:
        """
        Get a clarification question for the given question and history.
        Used by the orchestrator for step-by-step clarification.
        
        Args:
            question: Current question or user input
            history: Previous clarification exchanges
            
        Returns:
            Dictionary with clarification question and metadata
        """
        try:
            # If this is the first question, start clarification
            if not history or len(history) == 0:
                response = await self.start_clarification(question)
                return {
                    "question": response.content,
                    "clarification_complete": response.metadata.get("clarification_complete", False),
                    "context": response.metadata.get("analysis", {}),
                    "suggestions": response.key_points,
                    "round": 1
                }
            
            # Continue clarification with the latest user response
            latest_response = history[-1].get("user_response", question)
            response = await self.continue_clarification(latest_response)
            
            return {
                "question": response.content,
                "clarification_complete": response.metadata.get("clarification_complete", False),
                "final_question": response.metadata.get("final_question", ""),
                "context": response.metadata.get("progress", {}),
                "suggestions": response.key_points,
                "round": len(history) + 1,
                "summary": response.metadata.get("completion_reason", "")
            }
            
        except Exception as e:
            logger.error(f"Error getting clarification question: {e}")
            return {
                "question": "I apologize, but I encountered an error. Could you please rephrase your question?",
                "clarification_complete": False,
                "context": {"error": str(e)},
                "suggestions": [],
                "round": len(history) + 1 if history else 1
            }
    
    async def process_clarification_response(self, user_response: str, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user response and return next step in clarification.
        Used by the orchestrator for streamlined clarification flow.
        
        Args:
            user_response: User's response to clarification question
            session_context: Session context from orchestrator
            
        Returns:
            Dictionary with next question or completion signal
        """
        try:
            # Continue clarification process
            response = await self.continue_clarification(user_response, session_context)
            
            return {
                "clarification_complete": response.metadata.get("clarification_complete", False),
                "question": response.content,
                "final_question": response.metadata.get("final_question", ""),
                "key_concepts": response.metadata.get("key_concepts", []),
                "context": response.metadata.get("progress", {}),
                "summary": response.reasoning or "",
                "suggestions": response.key_points
            }
            
        except Exception as e:
            logger.error(f"Error processing clarification response: {e}")
            return {
                "clarification_complete": False,
                "question": "I'm having trouble processing your response. Could you try rephrasing it?",
                "final_question": "",
                "key_concepts": [],
                "context": {"error": str(e)},
                "summary": "",
                "suggestions": []
            }
    
    async def _analyze_initial_question(self, question: str) -> Dict[str, Any]:
        """Analyze the initial question to understand its structure and content"""
        try:
            # Extract basic information
            key_terms = extract_key_concepts(question, max_concepts=10)
            
            # Identify question type patterns
            question_indicators = {
                "what": ["what", "which", "where"],
                "how": ["how"],
                "why": ["why", "because"],
                "should": ["should", "ought", "must"],
                "comparison": ["better", "worse", "versus", "vs", "compared"],
                "future": ["will", "future", "prediction", "forecast"],
                "opinion": ["think", "believe", "opinion", "view"]
            }
            
            question_lower = question.lower()
            detected_types = []
            for category, indicators in question_indicators.items():
                if any(indicator in question_lower for indicator in indicators):
                    detected_types.append(category)
            
            # Identify potential assumptions
            assumption_patterns = [
                r"all\s+\w+\s+are",
                r"everyone\s+\w+",
                r"never\s+\w+",
                r"always\s+\w+",
                r"must\s+be",
                r"obviously",
                r"clearly"
            ]
            
            potential_assumptions = []
            for pattern in assumption_patterns:
                matches = re.findall(pattern, question_lower)
                potential_assumptions.extend(matches)
            
            # Assess complexity and scope
            word_count = len(question.split())
            complexity = "high" if word_count > 20 else "medium" if word_count > 10 else "low"
            
            return {
                "understanding": f"This appears to be a {'/'.join(detected_types) if detected_types else 'general'} question about {', '.join(key_terms[:3])}.",
                "key_terms": key_terms,
                "question_types": detected_types,
                "potential_assumptions": potential_assumptions,
                "complexity": complexity,
                "word_count": word_count,
                "scope": "broad" if len(key_terms) > 5 else "focused"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing initial question: {e}")
            return {
                "understanding": "I need to understand this question better.",
                "key_terms": [],
                "question_types": [],
                "potential_assumptions": [],
                "complexity": "unknown",
                "word_count": 0,
                "scope": "unclear"
            }
    
    async def _generate_first_question(self, question: str, analysis: Dict[str, Any]) -> SocraticQuestion:
        """Generate the first clarifying question based on initial analysis"""
        try:
            # Choose strategy based on analysis
            if analysis.get("complexity") == "high" or analysis.get("scope") == "broad":
                # Start with scope clarification
                return SocraticQuestion(
                    question=f"I can see this is a complex question about {', '.join(analysis.get('key_terms', [])[:2])}. To help me understand better, what's the most important aspect you'd like to explore first?",
                    question_type=QuestionType.CLARIFICATION,
                    purpose="Narrow down scope and identify priority focus",
                    follow_up_hints=[
                        "Think about what outcome you're hoping for",
                        "Consider which part is most urgent or interesting to you",
                        "Focus on one main aspect rather than trying to cover everything"
                    ]
                )
            
            elif analysis.get("potential_assumptions"):
                # Challenge assumptions early
                assumption = analysis["potential_assumptions"][0]
                return SocraticQuestion(
                    question=f"I notice you mentioned '{assumption}' in your question. What leads you to believe this is always the case?",
                    question_type=QuestionType.ASSUMPTION,
                    purpose="Identify and examine underlying assumptions",
                    follow_up_hints=[
                        "Think about exceptions to this rule",
                        "Consider different contexts where this might not apply",
                        "Reflect on where this belief comes from"
                    ]
                )
            
            elif analysis.get("key_terms"):
                # Start with definition clarification
                key_term = analysis["key_terms"][0]
                return SocraticQuestion(
                    question=f"Let's start by clarifying what you mean by '{key_term}'. How would you define this term in the context of your question?",
                    question_type=QuestionType.DEFINITION,
                    purpose="Establish clear definitions for key concepts",
                    follow_up_hints=[
                        "Think about how others might define this differently",
                        "Consider specific examples that illustrate your definition",
                        "Reflect on the boundaries of this concept"
                    ]
                )
            
            else:
                # General understanding question
                return SocraticQuestion(
                    question="Help me understand what's driving this question for you. What situation or experience led you to ask this?",
                    question_type=QuestionType.META_QUESTION,
                    purpose="Understand the context and motivation behind the question",
                    follow_up_hints=[
                        "Think about the personal or practical importance",
                        "Consider what you hope to achieve with the answer",
                        "Reflect on what prompted this inquiry"
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error generating first question: {e}")
            return SocraticQuestion(
                question="To help me understand your question better, could you tell me what's most important to you about this topic?",
                question_type=QuestionType.CLARIFICATION,
                purpose="Basic understanding",
                follow_up_hints=["Focus on what matters most to you"]
            )
    
    async def _process_user_response(self, response: str):
        """Process and integrate user response into clarification context"""
        try:
            response = clean_text(response)
            
            # Extract new information from response
            new_concepts = extract_key_concepts(response, max_concepts=5)
            self.clarification_context.key_terms.extend(new_concepts)
            self.clarification_context.key_terms = list(set(self.clarification_context.key_terms))  # Remove duplicates
            
            # Look for assumptions in the response
            assumption_patterns = [
                r"i think (\w+(?:\s+\w+)*)",
                r"i believe (\w+(?:\s+\w+)*)",
                r"obviously (\w+(?:\s+\w+)*)",
                r"clearly (\w+(?:\s+\w+)*)",
                r"everyone knows (\w+(?:\s+\w+)*)",
                r"it's common sense (\w+(?:\s+\w+)*)"
            ]
            
            for pattern in assumption_patterns:
                matches = re.findall(pattern, response.lower())
                for match in matches:
                    if match not in self.clarification_context.identified_assumptions:
                        self.clarification_context.identified_assumptions.append(match)
            
            # Look for context clues
            context_patterns = [
                r"in my experience (\w+(?:\s+\w+)*)",
                r"at work (\w+(?:\s+\w+)*)",
                r"in (\w+(?:\s+\w+)*) context",
                r"when (\w+(?:\s+\w+)*)",
                r"because (\w+(?:\s+\w+)*)"
            ]
            
            for pattern in context_patterns:
                matches = re.findall(pattern, response.lower())
                for match in matches:
                    if match not in self.clarification_context.context_factors:
                        self.clarification_context.context_factors.append(match)
            
            # Update current understanding
            self.clarification_context.current_understanding += f" User clarified: {response[:100]}..."
            
            # Track refinement
            self.clarification_context.refinement_history.append(response)
            
            # Update stage based on progress
            self._update_clarification_stage()
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
    
    def _update_clarification_stage(self):
        """Update the current clarification stage based on progress"""
        try:
            rounds = self.clarification_context.rounds_completed
            has_definitions = len(self.clarification_context.key_terms) >= 2
            has_assumptions = len(self.clarification_context.identified_assumptions) >= 1
            has_context = len(self.clarification_context.context_factors) >= 1
            
            if rounds >= 4 or (has_definitions and has_assumptions and has_context):
                self.clarification_context.stage = ClarificationStage.COMPLETION_ASSESSMENT
            elif has_context:
                self.clarification_context.stage = ClarificationStage.SCOPE_REFINEMENT
            elif has_assumptions:
                self.clarification_context.stage = ClarificationStage.CONTEXT_GATHERING
            elif has_definitions:
                self.clarification_context.stage = ClarificationStage.ASSUMPTION_EXPLORATION
            elif rounds >= 2:
                self.clarification_context.stage = ClarificationStage.DEFINITION_CLARIFICATION
            else:
                self.clarification_context.stage = ClarificationStage.INITIAL_UNDERSTANDING
                
        except Exception as e:
            logger.error(f"Error updating clarification stage: {e}")
    
    async def _generate_next_question(self) -> SocraticQuestion:
        """Generate the next question based on current stage and context"""
        try:
            stage = self.clarification_context.stage
            
            if stage == ClarificationStage.DEFINITION_CLARIFICATION:
                return await self._generate_definition_question()
            elif stage == ClarificationStage.ASSUMPTION_EXPLORATION:
                return await self._generate_assumption_question()
            elif stage == ClarificationStage.CONTEXT_GATHERING:
                return await self._generate_context_question()
            elif stage == ClarificationStage.SCOPE_REFINEMENT:
                return await self._generate_scope_question()
            elif stage == ClarificationStage.GOAL_CLARIFICATION:
                return await self._generate_goal_question()
            else:
                # Default to clarification
                return await self._generate_clarification_question()
                
        except Exception as e:
            logger.error(f"Error generating next question: {e}")
            return SocraticQuestion(
                question="Could you help me understand this better by giving me a specific example?",
                question_type=QuestionType.EXAMPLE,
                purpose="Gain concrete understanding",
                follow_up_hints=["Think of a real situation where this applies"]
            )
    
    async def _generate_definition_question(self) -> SocraticQuestion:
        """Generate a question focused on defining key terms"""
        undefined_terms = [term for term in self.clarification_context.key_terms 
                          if term not in self.clarification_context.current_understanding]
        
        if undefined_terms:
            term = undefined_terms[0]
            return SocraticQuestion(
                question=f"You mentioned '{term}' - this seems important to your question. How would you distinguish '{term}' from similar concepts?",
                question_type=QuestionType.DEFINITION,
                purpose="Clarify definition and boundaries of key concepts",
                follow_up_hints=[
                    f"Think about what makes '{term}' unique",
                    "Consider edge cases or borderline examples",
                    "Reflect on how others might define this differently"
                ]
            )
        else:
            return SocraticQuestion(
                question="Are there any other important terms or concepts in your question that might be interpreted differently by different people?",
                question_type=QuestionType.CLARIFICATION,
                purpose="Identify additional terms needing definition",
                follow_up_hints=[
                    "Look for words that could have multiple meanings",
                    "Consider technical or specialized terms",
                    "Think about concepts that might be controversial"
                ]
            )
    
    async def _generate_assumption_question(self) -> SocraticQuestion:
        """Generate a question focused on exploring assumptions"""
        if self.clarification_context.identified_assumptions:
            assumption = self.clarification_context.identified_assumptions[0]
            return SocraticQuestion(
                question=f"You seem to assume that {assumption}. What if this weren't true? How would that change your question?",
                question_type=QuestionType.ASSUMPTION,
                purpose="Challenge and examine underlying assumptions",
                follow_up_hints=[
                    "Consider alternative possibilities",
                    "Think about exceptions to this assumption",
                    "Reflect on the source of this belief"
                ]
            )
        else:
            return SocraticQuestion(
                question="What assumptions are you making about this situation that might not be obvious?",
                question_type=QuestionType.ASSUMPTION,
                purpose="Identify hidden assumptions",
                follow_up_hints=[
                    "Think about what you're taking for granted",
                    "Consider what someone from a different background might assume",
                    "Reflect on unstated premises in your thinking"
                ]
            )
    
    async def _generate_context_question(self) -> SocraticQuestion:
        """Generate a question focused on gathering context"""
        return SocraticQuestion(
            question="Help me understand the broader context. What circumstances or constraints are shaping this question for you?",
            question_type=QuestionType.CLARIFICATION,
            purpose="Gather relevant contextual information",
            follow_up_hints=[
                "Think about time constraints or deadlines",
                "Consider organizational or social factors",
                "Reflect on resources and limitations",
                "Think about stakeholders or affected parties"
            ]
        )
    
    async def _generate_scope_question(self) -> SocraticQuestion:
        """Generate a question focused on refining scope"""
        return SocraticQuestion(
            question="Your question touches on several important areas. If you had to choose the single most important aspect to focus on, what would it be and why?",
            question_type=QuestionType.CLARIFICATION,
            purpose="Narrow scope to most essential elements",
            follow_up_hints=[
                "Think about what would have the biggest impact",
                "Consider what's most urgent or time-sensitive",
                "Reflect on what you have the most control over",
                "Focus on what's most actionable"
            ]
        )
    
    async def _generate_goal_question(self) -> SocraticQuestion:
        """Generate a question focused on clarifying goals"""
        return SocraticQuestion(
            question="What would a good answer to your question help you achieve? What would you do differently if you had clarity on this?",
            question_type=QuestionType.META_QUESTION,
            purpose="Clarify the purpose and desired outcomes",
            follow_up_hints=[
                "Think about the practical applications",
                "Consider how this fits into your larger goals",
                "Reflect on what decisions this would inform",
                "Think about who else might benefit from the answer"
            ]
        )
    
    async def _generate_clarification_question(self) -> SocraticQuestion:
        """Generate a general clarification question"""
        return SocraticQuestion(
            question="What aspect of this question is most important to you, and what makes it challenging to answer?",
            question_type=QuestionType.CLARIFICATION,
            purpose="Understand core concerns and challenges",
            follow_up_hints=[
                "Think about what keeps you up at night about this",
                "Consider what makes this question difficult",
                "Reflect on what you're hoping to understand better"
            ]
        )
    
    async def _assess_completion(self) -> Dict[str, Any]:
        """Assess whether clarification is complete"""
        try:
            context = self.clarification_context
            
            # Completion criteria
            has_clear_definitions = len(context.key_terms) >= 2
            has_examined_assumptions = len(context.identified_assumptions) >= 1
            has_sufficient_context = len(context.context_factors) >= 1
            reached_max_rounds = context.rounds_completed >= context.max_rounds
            
            # Quality indicators
            recent_responses = context.refinement_history[-2:]
            shows_clarity = any("clear" in response.lower() or "understand" in response.lower() 
                              for response in recent_responses)
            
            # Check for completion signals
            completion_keywords = ["ready", "clear", "understand", "enough", "good", "proceed"]
            shows_readiness = any(keyword in response.lower() 
                                for response in recent_responses 
                                for keyword in completion_keywords)
            
            # Decision logic
            if reached_max_rounds:
                complete = True
                reason = "Maximum rounds reached"
            elif has_clear_definitions and has_examined_assumptions and has_sufficient_context:
                complete = True
                reason = "All key criteria satisfied"
            elif shows_readiness and (has_clear_definitions or has_examined_assumptions):
                complete = True
                reason = "User indicates readiness and basic criteria met"
            else:
                complete = False
                reason = "More clarification needed"
            
            # Generate summary points
            summary_points = []
            if context.key_terms:
                summary_points.append(f"Key concepts identified: {', '.join(context.key_terms[:3])}")
            if context.identified_assumptions:
                summary_points.append(f"Assumptions examined: {len(context.identified_assumptions)} identified")
            if context.context_factors:
                summary_points.append(f"Context factors: {', '.join(context.context_factors[:2])}")
            
            return {
                "complete": complete,
                "reason": reason,
                "summary_points": summary_points,
                "criteria_met": {
                    "definitions": has_clear_definitions,
                    "assumptions": has_examined_assumptions,
                    "context": has_sufficient_context,
                    "clarity": shows_clarity,
                    "readiness": shows_readiness
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing completion: {e}")
            return {
                "complete": True,  # Default to complete on error
                "reason": "Error in assessment, proceeding to debate",
                "summary_points": [],
                "criteria_met": {}
            }
    
    async def _generate_final_question(self) -> str:
        """Generate the final clarified question"""
        try:
            context = self.clarification_context
            
            # Start with original question
            base_question = context.original_question
            
            # Incorporate key refinements
            refinements = []
            
            # Add key terms that were clarified
            if context.key_terms:
                key_terms_str = ", ".join(context.key_terms[:3])
                refinements.append(f"specifically regarding {key_terms_str}")
            
            # Add context factors
            if context.context_factors:
                context_str = ", ".join(context.context_factors[:2])
                refinements.append(f"in the context of {context_str}")
            
            # Add assumption considerations if significant
            if len(context.identified_assumptions) >= 2:
                refinements.append("taking into account the assumptions we've identified")
            
            # Construct refined question
            if refinements:
                refined_question = f"{base_question.rstrip('?')}, {', and '.join(refinements)}?"
            else:
                refined_question = base_question
            
            # Clean up the question
            refined_question = clean_text(refined_question)
            
            return refined_question
            
        except Exception as e:
            logger.error(f"Error generating final question: {e}")
            return self.clarification_context.original_question
    
    def _get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of clarification progress"""
        if not self.clarification_context:
            return {}
        
        context = self.clarification_context
        
        return {
            "stage": context.stage.value,
            "rounds_completed": context.rounds_completed,
            "max_rounds": context.max_rounds,
            "key_terms_identified": len(context.key_terms),
            "assumptions_examined": len(context.identified_assumptions),
            "context_factors": len(context.context_factors),
            "completion_percentage": min(100, (context.rounds_completed / context.max_rounds) * 100)
        }
    
    def _initialize_question_strategies(self) -> Dict[str, List[str]]:
        """Initialize question strategies for different scenarios"""
        return {
            "broad_questions": [
                "What's the most important aspect to focus on?",
                "If you had to narrow this down to one key issue, what would it be?",
                "What would success look like in addressing this question?"
            ],
            "vague_terms": [
                "What do you mean by '{term}'?",
                "How would you define '{term}' in this context?",
                "What distinguishes '{term}' from similar concepts?"
            ],
            "assumptions": [
                "What are you assuming about this situation?",
                "What if that assumption weren't true?",
                "Where does that belief come from?"
            ],
            "context": [
                "What circumstances led to this question?",
                "What constraints are you working within?",
                "Who else is affected by this issue?"
            ]
        }
    
    def _initialize_completion_criteria(self) -> Dict[str, Any]:
        """Initialize criteria for determining completion"""
        return {
            "minimum_rounds": 2,
            "maximum_rounds": 5,
            "required_elements": ["key_terms", "assumptions_or_context"],
            "completion_signals": ["ready", "clear", "understand", "proceed", "enough"]
        }
    
    async def reset_clarification(self):
        """Reset clarification context for new session"""
        self.clarification_context = None
        self.clear_conversation_history(keep_system=True)
        logger.info(f"Clarification context reset for agent {self.agent_id}")
    
    def get_clarification_summary(self) -> Dict[str, Any]:
        """Get summary of current clarification state"""
        if not self.clarification_context:
            return {"active": False}
        
        return {
            "active": True,
            "original_question": self.clarification_context.original_question,
            "current_stage": self.clarification_context.stage.value,
            "rounds_completed": self.clarification_context.rounds_completed,
            "key_terms": self.clarification_context.key_terms,
            "assumptions_identified": self.clarification_context.identified_assumptions,
            "context_factors": self.clarification_context.context_factors,
            "progress": self._get_progress_summary()
        }
