#!/usr/bin/env python3
"""
EchoForge Journal Assistant Agent
================================
Specialized agent that helps users process and reflect on their debate experiences through guided journaling.

Features:
- Guided reflection on debate experiences
- Insight extraction and pattern recognition
- Emotional processing support
- Learning objective identification
- Growth tracking and progress monitoring
- Metacognitive development assistance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from agents.base_agent import BaseAgent, ConversationMessage, AgentResponse, AgentConfig
from models import DebatePhase, MessageType, JournalEntryType, ReflectionDepth
from utils import extract_key_concepts, calculate_similarity, clean_text

logger = logging.getLogger(__name__)


@dataclass
class ReflectionPrompt:
    """Structured prompt for guided reflection."""
    category: str  # emotional, cognitive, metacognitive, learning
    question: str
    follow_ups: List[str]
    depth_level: str  # surface, intermediate, deep
    context_dependent: bool = False


@dataclass
class InsightExtraction:
    """Extracted insight from reflection."""
    insight_type: str  # learning, pattern, emotional, strategic
    content: str
    confidence_level: float
    supporting_evidence: List[str]
    potential_applications: List[str]
    growth_area: Optional[str] = None


@dataclass
class DebateReflection:
    """Comprehensive reflection on a debate experience."""
    session_id: str
    debate_question: str
    user_position: str
    key_moments: List[str]
    emotional_journey: Dict[str, str]
    learning_points: List[str]
    challenges_faced: List[str]
    strengths_demonstrated: List[str]
    growth_opportunities: List[str]
    overall_satisfaction: float
    confidence_change: float
    insights_gained: List[InsightExtraction]


@dataclass
class JournalContext:
    """Context for journaling operations."""
    session_id: str
    user_id: str
    debate_history: List[Dict[str, Any]]
    previous_reflections: List[DebateReflection]
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    reflection_depth: str = "intermediate"
    focus_areas: List[str] = field(default_factory=list)
    emotional_state: Optional[str] = None


class JournalAssistantAgent(BaseAgent):
    """
    Agent specialized in guiding users through reflective journaling about their debate experiences.
    
    Helps users process emotions, extract insights, identify learning opportunities,
    and track their growth in critical thinking and debate skills through
    structured reflection and supportive guidance.
    """

    def __init__(self, **kwargs):
        # Default configuration optimized for journaling
        default_config = AgentConfig(
            model="tinyllama:1.1b",  # Lightweight model for journaling
            temperature=0.6,         # Slightly creative for reflection
            max_tokens=512,          # Moderate length for journal prompts
            timeout=60,              # Quick turnaround for journaling
            enable_tools=False,      # Pure reasoning for reflection
            enable_memory=True,
            memory_limit=20          # Focused journaling context
        )
        
        # Merge with provided config
        config = default_config.__dict__.copy()
        config.update(kwargs)
        final_config = AgentConfig(**config)
        
        super().__init__(final_config)
        self.agent_role = "journal_assistant"
        
        # Reflection prompt categories
        self.reflection_prompts = {
            "emotional": [
                ReflectionPrompt("emotional", "How did you feel at different points during the debate?", 
                               ["What emotions surprised you?", "When did you feel most/least confident?"], "surface"),
                ReflectionPrompt("emotional", "What emotional patterns do you notice in your debate experiences?", 
                               ["How do these patterns affect your reasoning?", "What triggers strong emotional responses?"], "deep"),
                ReflectionPrompt("emotional", "How did your feelings about the topic change during the debate?", 
                               ["What caused these changes?", "How did emotions influence your arguments?"], "intermediate")
            ],
            "cognitive": [
                ReflectionPrompt("cognitive", "What was the strongest argument you made? Why?", 
                               ["What made it effective?", "How could it be improved?"], "surface"),
                ReflectionPrompt("cognitive", "Which of your assumptions were challenged during the debate?", 
                               ["How did you respond to these challenges?", "What new perspectives emerged?"], "intermediate"),
                ReflectionPrompt("cognitive", "How did your understanding of the topic evolve throughout the debate?", 
                               ["What evidence most influenced your thinking?", "Where did you change your mind?"], "deep")
            ],
            "metacognitive": [
                ReflectionPrompt("metacognitive", "How did you monitor your own thinking during the debate?", 
                               ["When did you catch yourself making assumptions?", "How did you check your reasoning?"], "intermediate"),
                ReflectionPrompt("metacognitive", "What patterns do you notice in how you approach complex questions?", 
                               ["What are your cognitive strengths and blindspots?", "How do you handle uncertainty?"], "deep"),
                ReflectionPrompt("metacognitive", "How effective were your debate strategies?", 
                               ["What worked well?", "What would you do differently?"], "surface")
            ],
            "learning": [
                ReflectionPrompt("learning", "What did you learn about the topic that you didn't know before?", 
                               ["What surprised you most?", "What questions do you still have?"], "surface"),
                ReflectionPrompt("learning", "What did you learn about your own thinking and debate style?", 
                               ["What strengths did you discover?", "What areas need development?"], "intermediate"),
                ReflectionPrompt("learning", "How will this debate experience influence your future thinking?", 
                               ["What principles will you apply going forward?", "How has your approach evolved?"], "deep")
            ]
        }

    async def initiate_post_debate_reflection(
        self, 
        context: JournalContext
    ) -> AgentResponse:
        """
        Initiate guided reflection immediately after a debate concludes.
        """
        try:
            # Assess user's likely emotional state and energy level
            emotional_state = await self._assess_post_debate_state(context)
            context.emotional_state = emotional_state
            
            # Select appropriate reflection approach based on state
            reflection_approach = self._select_reflection_approach(emotional_state, context)
            
            initial_prompt = f"""
            Your debate on "{context.debate_history[-1].get('question', 'the topic')}" has concluded. 
            
            I'm here to help you reflect on this experience and capture any insights while they're fresh. 
            This reflection is entirely for your benefit - there are no right or wrong answers, just 
            your honest thoughts and feelings.

            Let's start with a simple check-in:

            **How are you feeling right now about the debate you just had?**

            Take a moment to notice:
            - Your energy level (high, medium, low)
            - Your emotional state (excited, frustrated, curious, satisfied, etc.)
            - Your overall sense of how it went

            Share whatever feels right to you. We'll go at your pace and focus on what's 
            most helpful for your learning and growth.
            """

            # Track initiation
            await self._track_performance("reflection_initiation", {
                "emotional_state": emotional_state,
                "approach": reflection_approach,
                "debate_length": len(context.debate_history)
            })

            return AgentResponse(
                success=True,
                content=initial_prompt,
                agent_type=self.agent_role,
                metadata={
                    "reflection_phase": "initiation",
                    "approach": reflection_approach,
                    "emotional_state": emotional_state,
                    "message_type": MessageType.REFLECTION_PROMPT.value
                }
            )

        except Exception as e:
            logger.error(f"Error initiating reflection: {e}")
            return AgentResponse(
                success=False,
                content="I'm having trouble starting the reflection process. Would you like to share how you're feeling about the debate?",
                error=str(e),
                agent_type=self.agent_role
            )

    async def guide_reflection_conversation(
        self,
        user_response: str,
        context: JournalContext,
        current_phase: str = "exploration"
    ) -> AgentResponse:
        """
        Guide an ongoing reflection conversation with adaptive prompts.
        """
        try:
            # Analyze user's response for emotional tone and content
            response_analysis = await self._analyze_user_response(user_response, context)
            
            # Select next prompt based on conversation flow
            next_prompt = await self._select_next_prompt(response_analysis, context, current_phase)
            
            # Generate contextual response
            response_content = await self._generate_reflection_response(
                user_response, next_prompt, response_analysis, context
            )

            return AgentResponse(
                success=True,
                content=response_content,
                agent_type=self.agent_role,
                metadata={
                    "reflection_phase": current_phase,
                    "prompt_category": next_prompt.category if next_prompt else "synthesis",
                    "emotional_tone": response_analysis.get("emotional_tone", "neutral"),
                    "depth_level": next_prompt.depth_level if next_prompt else "intermediate",
                    "message_type": MessageType.REFLECTION_GUIDANCE.value
                }
            )

        except Exception as e:
            logger.error(f"Error in reflection guidance: {e}")
            return AgentResponse(
                success=False,
                content="I want to make sure I understand your experience. Could you tell me more about what stood out to you in the debate?",
                error=str(e),
                agent_type=self.agent_role
            )

    async def extract_insights_from_reflection(
        self,
        reflection_content: str,
        context: JournalContext
    ) -> List[InsightExtraction]:
        """
        Extract meaningful insights from user's reflection content.
        """
        try:
            insight_prompt = f"""
            Analyze this reflection content and extract meaningful insights for the user's growth and learning.

            REFLECTION CONTENT:
            {reflection_content}

            CONTEXT:
            - Debate Topic: {context.debate_history[-1].get('question', 'Unknown') if context.debate_history else 'Unknown'}
            - Previous Reflections: {len(context.previous_reflections)} sessions

            Extract insights in these categories:

            1. LEARNING INSIGHTS:
               - What new knowledge or understanding did they gain?
               - What misconceptions were corrected?
               - What questions emerged for future exploration?

            2. PATTERN INSIGHTS:
               - What recurring themes appear in their thinking?
               - What cognitive or emotional patterns are evident?
               - How do these patterns help or hinder their reasoning?

            3. EMOTIONAL INSIGHTS:
               - How do emotions affect their debate performance?
               - What emotional growth or awareness is evident?
               - What emotional patterns need attention?

            4. STRATEGIC INSIGHTS:
               - What debate strategies are they developing?
               - What communication skills are improving?
               - What tactical adjustments would be beneficial?

            For each insight:
            - State it clearly and specifically
            - Assess confidence level (1-10)
            - Identify supporting evidence from their reflection
            - Suggest potential applications or next steps
            - Identify any growth areas this reveals

            Focus on insights that are:
            - Actionable and constructive
            - Based on evidence in their reflection
            - Supportive of their intellectual and emotional growth
            - Honest but encouraging

            Avoid insights that are:
            - Overly critical or discouraging
            - Based on assumptions rather than evidence
            - Focused on what they did "wrong" rather than growth opportunities
            """

            response = await self._get_completion(insight_prompt, context_key="insight_extraction")
            
            if not response.success:
                logger.warning(f"Insight extraction failed: {response.error}")
                return []

            return self._parse_extracted_insights(response.content)

        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []

    async def generate_growth_recommendations(
        self,
        debate_reflection: DebateReflection,
        context: JournalContext
    ) -> List[str]:
        """
        Generate personalized growth recommendations based on reflection analysis.
        """
        try:
            recommendations_prompt = f"""
            Based on this user's debate reflection and growth patterns, generate supportive 
            and actionable recommendations for their continued development.

            REFLECTION SUMMARY:
            - Topic: {debate_reflection.debate_question}
            - Position: {debate_reflection.user_position}
            - Satisfaction: {debate_reflection.overall_satisfaction}/10
            - Confidence Change: {debate_reflection.confidence_change}

            STRENGTHS DEMONSTRATED:
            {chr(10).join(f"- {strength}" for strength in debate_reflection.strengths_demonstrated)}

            CHALLENGES FACED:
            {chr(10).join(f"- {challenge}" for challenge in debate_reflection.challenges_faced)}

            GROWTH OPPORTUNITIES:
            {chr(10).join(f"- {opportunity}" for opportunity in debate_reflection.growth_opportunities)}

            INSIGHTS GAINED:
            {chr(10).join(f"- {insight.content}" for insight in debate_reflection.insights_gained)}

            Generate 3-5 specific, actionable recommendations that:

            1. BUILD ON STRENGTHS:
               - Acknowledge what they're doing well
               - Suggest ways to further develop these strengths
               - Show how to leverage strengths in new areas

            2. ADDRESS GROWTH AREAS:
               - Focus on 1-2 most impactful areas for improvement
               - Provide specific, achievable steps
               - Frame challenges as opportunities for growth

            3. SUPPORT CONTINUED LEARNING:
               - Suggest resources or practices for ongoing development
               - Recommend ways to apply insights to future debates
               - Encourage experimentation with new approaches

            GUIDELINES:
            - Be encouraging and supportive while being honest
            - Focus on process improvements rather than outcomes
            - Make recommendations specific and actionable
            - Consider their current skill level and emotional state
            - Emphasize growth mindset and learning from experience
            """

            response = await self._get_completion(recommendations_prompt, context_key="growth_recommendations")
            
            if not response.success:
                logger.warning(f"Growth recommendations failed: {response.error}")
                return ["Continue practicing active reflection after debates", "Focus on understanding different perspectives"]

            return self._parse_recommendations(response.content)

        except Exception as e:
            logger.error(f"Error generating growth recommendations: {e}")
            return ["Continue engaging in thoughtful reflection", "Practice applying insights to future discussions"]

    async def create_journal_entry_draft(
        self,
        debate_reflection: DebateReflection,
        context: JournalContext
    ) -> str:
        """
        Create a structured journal entry draft based on the reflection.
        """
        try:
            entry_prompt = f"""
            Create a thoughtful journal entry based on this user's debate reflection.
            Write in their voice, incorporating their insights and experiences.

            REFLECTION DATA:
            Topic: {debate_reflection.debate_question}
            Position: {debate_reflection.user_position}
            Key Moments: {', '.join(debate_reflection.key_moments)}
            
            Learning Points: {', '.join(debate_reflection.learning_points)}
            Challenges: {', '.join(debate_reflection.challenges_faced)}
            Strengths: {', '.join(debate_reflection.strengths_demonstrated)}

            Emotional Journey:
            {chr(10).join(f"{moment}: {emotion}" for moment, emotion in debate_reflection.emotional_journey.items())}

            Key Insights:
            {chr(10).join(f"- {insight.content}" for insight in debate_reflection.insights_gained)}

            Create a journal entry that:

            1. CAPTURES THE EXPERIENCE:
               - Describes the debate topic and their position
               - Highlights key moments and turning points
               - Acknowledges the emotional journey

            2. DOCUMENTS LEARNING:
               - Records specific insights and new understanding
               - Notes challenges faced and how they were handled
               - Identifies strengths that emerged

            3. REFLECTS ON GROWTH:
               - Connects this experience to broader learning patterns
               - Considers implications for future thinking and debates
               - Sets intentions for continued development

            4. MAINTAINS AUTHENTIC VOICE:
               - Use first person perspective
               - Match their level of reflection and introspection
               - Be honest about both successes and struggles

            Aim for 300-500 words that feel genuine and personally meaningful.
            Include specific details that will help them remember this experience.
            """

            response = await self._get_completion(entry_prompt, context_key="journal_entry")
            
            if response.success:
                return response.content
            else:
                logger.warning(f"Journal entry creation failed: {response.error}")
                return self._create_fallback_journal_entry(debate_reflection)

        except Exception as e:
            logger.error(f"Error creating journal entry: {e}")
            return self._create_fallback_journal_entry(debate_reflection)

    async def _assess_post_debate_state(self, context: JournalContext) -> str:
        """
        Assess the user's likely emotional and cognitive state after the debate.
        """
        # This would ideally use more sophisticated analysis
        # For now, use simple heuristics based on debate characteristics
        
        if not context.debate_history:
            return "neutral"
        
        last_debate = context.debate_history[-1]
        debate_length = len(last_debate.get('exchanges', []))
        
        # Simple heuristic assessment
        if debate_length > 8:
            return "potentially_tired"
        elif debate_length < 3:
            return "possibly_frustrated"
        else:
            return "engaged"

    def _select_reflection_approach(self, emotional_state: str, context: JournalContext) -> str:
        """
        Select appropriate reflection approach based on user state.
        """
        if emotional_state == "potentially_tired":
            return "gentle"
        elif emotional_state == "possibly_frustrated":
            return "supportive"
        elif emotional_state == "engaged":
            return "exploratory"
        else:
            return "standard"

    async def _analyze_user_response(
        self, 
        response: str, 
        context: JournalContext
    ) -> Dict[str, Any]:
        """
        Analyze user's response for emotional tone and content themes.
        """
        analysis = {
            "emotional_tone": "neutral",
            "engagement_level": "medium",
            "reflection_depth": "surface",
            "key_themes": [],
            "needs_support": False,
            "ready_for_deeper_reflection": True
        }
        
        response_lower = response.lower()
        
        # Emotional tone detection
        positive_indicators = ["good", "great", "excited", "learned", "enjoyed", "satisfied"]
        negative_indicators = ["frustrated", "confused", "difficult", "challenging", "upset"]
        
        positive_count = sum(1 for word in positive_indicators if word in response_lower)
        negative_count = sum(1 for word in negative_indicators if word in response_lower)
        
        if positive_count > negative_count:
            analysis["emotional_tone"] = "positive"
        elif negative_count > positive_count:
            analysis["emotional_tone"] = "negative"
            analysis["needs_support"] = True
        
        # Engagement level
        if len(response.split()) > 50:
            analysis["engagement_level"] = "high"
        elif len(response.split()) < 10:
            analysis["engagement_level"] = "low"
        
        # Reflection depth
        reflection_indicators = ["because", "realized", "noticed", "learned", "understood", "felt"]
        if sum(1 for word in reflection_indicators if word in response_lower) > 2:
            analysis["reflection_depth"] = "deep"
        
        # Extract key themes
        analysis["key_themes"] = extract_key_concepts(response)[:5]
        
        return analysis

    async def _select_next_prompt(
        self, 
        response_analysis: Dict[str, Any], 
        context: JournalContext, 
        current_phase: str
    ) -> Optional[ReflectionPrompt]:
        """
        Select the next reflection prompt based on conversation analysis.
        """
        # Consider emotional state and engagement level
        if response_analysis.get("needs_support"):
            # Use gentler, more supportive prompts
            return ReflectionPrompt(
                "emotional", 
                "That sounds challenging. What part of the experience felt most difficult?",
                ["How did you handle that difficulty?", "What would help you feel more confident next time?"],
                "surface"
            )
        
        # Select based on current phase and depth
        if current_phase == "exploration":
            if response_analysis.get("reflection_depth") == "surface":
                # Encourage deeper reflection
                return self._get_prompt_by_criteria("cognitive", "intermediate")
            else:
                # Continue with current depth
                return self._get_prompt_by_criteria("learning", "intermediate")
        
        elif current_phase == "deepening":
            return self._get_prompt_by_criteria("metacognitive", "deep")
        
        else:  # synthesis phase
            return None  # No more prompts needed

    def _get_prompt_by_criteria(self, category: str, depth: str) -> ReflectionPrompt:
        """
        Get a reflection prompt matching specified criteria.
        """
        category_prompts = self.reflection_prompts.get(category, [])
        matching_prompts = [p for p in category_prompts if p.depth_level == depth]
        
        if matching_prompts:
            return matching_prompts[0]  # Could randomize in future
        elif category_prompts:
            return category_prompts[0]  # Fallback to any prompt in category
        else:
            # Fallback to a general prompt
            return ReflectionPrompt(
                "general",
                "What stands out most to you about this debate experience?",
                ["Why was that particularly significant?"],
                "intermediate"
            )

    async def _generate_reflection_response(
        self,
        user_response: str,
        next_prompt: Optional[ReflectionPrompt],
        response_analysis: Dict[str, Any],
        context: JournalContext
    ) -> str:
        """
        Generate a contextual response that acknowledges the user and provides next prompt.
        """
        try:
            # Acknowledge their response
            acknowledgment = self._create_acknowledgment(user_response, response_analysis)
            
            if next_prompt:
                # Provide the next prompt
                transition = self._create_transition(response_analysis, next_prompt)
                prompt_text = f"**{next_prompt.question}**"
                
                # Add follow-ups if appropriate
                if response_analysis.get("engagement_level") == "high":
                    follow_up = f"\n\nAs you think about that: {', '.join(next_prompt.follow_ups[:2])}"
                else:
                    follow_up = ""
                
                return f"{acknowledgment}\n\n{transition}\n\n{prompt_text}{follow_up}"
            
            else:
                # Synthesis phase - wrap up reflection
                return f"{acknowledgment}\n\nThank you for sharing your thoughtful reflections. Would you like me to help you capture these insights in a journal entry, or would you prefer to continue exploring any particular aspect of your experience?"

        except Exception as e:
            logger.error(f"Error generating reflection response: {e}")
            return "I appreciate you sharing that. What else stands out to you about your debate experience?"

    def _create_acknowledgment(self, user_response: str, analysis: Dict[str, Any]) -> str:
        """
        Create an appropriate acknowledgment of the user's response.
        """
        emotional_tone = analysis.get("emotional_tone", "neutral")
        
        if emotional_tone == "positive":
            acknowledgments = [
                "It sounds like you had a valuable experience.",
                "I can sense your engagement with the topic.",
                "That's a thoughtful observation."
            ]
        elif emotional_tone == "negative":
            acknowledgments = [
                "I appreciate you being honest about the challenges.",
                "It takes courage to reflect on difficult experiences.",
                "Those feelings are completely understandable."
            ]
        else:
            acknowledgments = [
                "Thank you for sharing that perspective.",
                "I can see you're thinking carefully about this.",
                "That's an interesting point to consider."
            ]
        
        return acknowledgments[0]  # Could randomize

    def _create_transition(self, analysis: Dict[str, Any], next_prompt: ReflectionPrompt) -> str:
        """
        Create a smooth transition to the next prompt.
        """
        if next_prompt.category == "emotional":
            return "Let's explore the emotional side of your experience:"
        elif next_prompt.category == "cognitive":
            return "I'm curious about your thinking process:"
        elif next_prompt.category == "metacognitive":
            return "Let's step back and look at how you approached this:"
        elif next_prompt.category == "learning":
            return "Thinking about what you've gained from this:"
        else:
            return "Building on what you've shared:"

    def _parse_extracted_insights(self, insight_text: str) -> List[InsightExtraction]:
        """
        Parse extracted insights from LLM response.
        """
        insights = []
        lines = insight_text.split('\n')
        
        current_insight = {}
        current_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for insight type headers
            if any(insight_type in line.lower() for insight_type in ["learning", "pattern", "emotional", "strategic"]):
                if current_insight and current_type:
                    insight = self._create_insight_from_data(current_insight, current_type)
                    if insight:
                        insights.append(insight)
                current_insight = {}
                current_type = self._extract_insight_type(line)
            
            # Look for specific fields
            elif "insight:" in line.lower() or "content:" in line.lower():
                current_insight["content"] = line.split(":", 1)[1].strip()
            elif "confidence:" in line.lower():
                try:
                    confidence_text = line.split(":", 1)[1].strip()
                    current_insight["confidence"] = float(re.search(r'(\d+)', confidence_text).group(1)) / 10.0
                except:
                    current_insight["confidence"] = 0.7
            elif "evidence:" in line.lower():
                current_insight["evidence"] = [line.split(":", 1)[1].strip()]
            elif "application:" in line.lower():
                current_insight["applications"] = [line.split(":", 1)[1].strip()]
        
        # Don't forget the last insight
        if current_insight and current_type:
            insight = self._create_insight_from_data(current_insight, current_type)
            if insight:
                insights.append(insight)
        
        return insights[:5]  # Limit to top 5

    def _extract_insight_type(self, line: str) -> str:
        """
        Extract insight type from line.
        """
        line_lower = line.lower()
        if "learning" in line_lower:
            return "learning"
        elif "pattern" in line_lower:
            return "pattern"
        elif "emotional" in line_lower:
            return "emotional"
        elif "strategic" in line_lower:
            return "strategic"
        else:
            return "general"

    def _create_insight_from_data(self, data: Dict[str, Any], insight_type: str) -> Optional[InsightExtraction]:
        """
        Create InsightExtraction from parsed data.
        """
        if "content" not in data:
            return None
        
        return InsightExtraction(
            insight_type=insight_type,
            content=data["content"],
            confidence_level=data.get("confidence", 0.7),
            supporting_evidence=data.get("evidence", []),
            potential_applications=data.get("applications", []),
            growth_area=data.get("growth_area")
        )

    def _parse_recommendations(self, recommendations_text: str) -> List[str]:
        """
        Parse recommendations from LLM response.
        """
        recommendations = []
        lines = recommendations_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendation = line[1:].strip()
                if recommendation:
                    recommendations.append(recommendation)
            elif line and any(char.isdigit() for char in line[:3]) and ':' in line:
                # Handle numbered recommendations
                recommendation = line.split(':', 1)[1].strip()
                if recommendation:
                    recommendations.append(recommendation)
        
        return recommendations[:5]  # Limit to top 5

    def _create_fallback_journal_entry(self, debate_reflection: DebateReflection) -> str:
        """
        Create a simple journal entry when LLM generation fails.
        """
        return f"""# Debate Reflection: {debate_reflection.debate_question}

Today I engaged in a debate about {debate_reflection.debate_question}, taking the position that {debate_reflection.user_position}.

## Key Experiences:
- {chr(10).join(f"- {moment}" for moment in debate_reflection.key_moments[:3])}

## What I Learned:
- {chr(10).join(f"- {learning}" for learning in debate_reflection.learning_points[:3])}

## Challenges I Faced:
- {chr(10).join(f"- {challenge}" for challenge in debate_reflection.challenges_faced[:3])}

## Strengths I Demonstrated:
- {chr(10).join(f"- {strength}" for strength in debate_reflection.strengths_demonstrated[:3])}

Overall, I feel {debate_reflection.overall_satisfaction}/10 satisfied with this debate experience. 
It was a valuable opportunity for learning and growth.
"""

    async def cleanup(self):
        """Clean up agent resources."""
        await super().cleanup()
        logger.info(f"JournalAssistantAgent {self.session_id} cleaned up successfully")

    def get_system_prompt(self) -> str:
        """Get the system prompt for the journal assistant agent."""
        return """You are a journal assistant agent. Your role is to:
        1. Help users reflect on their debate experiences
        2. Extract meaningful insights and learning outcomes
        3. Generate thoughtful questions for self-reflection
        4. Assist with organizing thoughts and experiences
        5. Support personal growth through guided journaling
        
        Be supportive, insightful, and focus on helping users learn from their experiences."""

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a request and generate a journal assistant response."""
        try:
            if context is None:
                context = {}
            
            # Build the prompt for journaling assistance
            prompt = f"""
            Context: You are helping a user reflect on their debate experience.
            
            User's Experience: {request}
            
            Your task: Help the user reflect on this experience by:
            - Identifying key insights and learning moments
            - Asking thoughtful follow-up questions
            - Helping organize their thoughts
            - Supporting their personal growth journey
            
            Be supportive and encouraging while helping them think deeply.
            """
            
            # Get response from LLM
            response = await self._get_llm_response(prompt, context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=response,
                metadata={
                    "agent_type": "journal_assistant",
                    "timestamp": datetime.now().isoformat(),
                    "context": context
                }
            )
            
        except Exception as e:
            logger.error(f"Error in journal assistant request processing: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="I encountered an error while helping with your reflection.",
                metadata={"error": str(e)}
            )
