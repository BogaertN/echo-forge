import logging
import json
import uuid
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import ollama
import whisper
import re

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all EchoForge agents.
    Provides context isolation and consistent interface.
    """
    
    def __init__(self, model: str, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.agent_id = str(uuid.uuid4())
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = self._get_system_prompt()
        self.created_at = datetime.now().isoformat()
        
        # Agent-specific configuration
        self.temperature = self.config.get('temperature', 0.4)
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.context_window = self.config.get('context_window', 4096)
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent role"""
        pass
    
    def _generate_response(self, prompt: str, system_override: str = None) -> str:
        """Generate response using Ollama with isolated context"""
        try:
            # Use system override if provided, otherwise use agent's system prompt
            system_prompt = system_override or self.system_prompt
            
            # Build messages with conversation history for context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history (limit to prevent context overflow)
            recent_history = self.conversation_history[-10:]  # Last 10 exchanges
            messages.extend(recent_history)
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_ctx': self.context_window,
                    'num_predict': self.max_tokens
                }
            )
            
            content = response['message']['content']
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Trim history if it gets too long
            if len(self.conversation_history) > 20:  # Keep last 20 messages
                self.conversation_history = self.conversation_history[-20:]
            
            return content
            
        except Exception as e:
            logger.error(f"Response generation failed for {self.__class__.__name__}: {str(e)}")
            raise
    
    def reset_context(self):
        """Reset the agent's conversation history"""
        self.conversation_history = []
        logger.info(f"Context reset for {self.__class__.__name__}")
    
    def get_context_summary(self) -> Dict:
        """Get summary of agent's current context"""
        return {
            'agent_type': self.__class__.__name__,
            'model': self.model,
            'conversation_length': len(self.conversation_history),
            'agent_id': self.agent_id,
            'created_at': self.created_at
        }

class StoicClarifier(BaseAgent):
    """
    Socratic clarification agent that asks probing questions
    to refine user input without providing direct answers.
    """
    
    def _get_system_prompt(self) -> str:
        return """You are a Stoic clarifier using the Socratic method. Your ONLY role is to ask clarifying questions to help refine and focus the user's thinking.

RULES:
- NEVER provide direct answers or solutions
- NEVER give advice or recommendations  
- ONLY ask probing questions that help clarify intent, assumptions, constraints, or goals
- Ask ONE focused question at a time
- Help the user think more deeply about their question
- When the question is crystal clear and well-defined, respond with exactly: "CLARIFICATION_COMPLETE"

Focus on clarifying:
- What specific outcome they want
- What constraints or limitations exist
- What assumptions they're making
- What they've already tried
- What success looks like to them"""
    
    def start_clarification(self, initial_question: str) -> str:
        """Start the clarification process with the initial question"""
        clarification_prompt = f"""A user has presented this question: "{initial_question}"

Ask ONE clarifying question to help them refine their thinking. Focus on the most important ambiguity or assumption in their question."""
        
        return self._generate_response(clarification_prompt)
    
    def continue_clarification(self, user_response: str, conversation_history: List[Dict]) -> Dict:
        """Continue clarification dialogue based on user response and history"""
        
        # Build context from conversation history
        context = "Clarification conversation so far:\n"
        for exchange in conversation_history[:-1]:  # Exclude current incomplete exchange
            if 'question' in exchange:
                context += f"Clarifier: {exchange['question']}\n"
            if 'response' in exchange:
                context += f"User: {exchange['response']}\n"
        
        context += f"User's latest response: {user_response}\n\n"
        
        prompt = f"""{context}

Based on this conversation, either:
1. Ask the next clarifying question to further refine their thinking, OR
2. If their intent and constraints are now crystal clear, respond with exactly: "CLARIFICATION_COMPLETE"

Remember: ONLY ask questions, never give answers."""
        
        response = self._generate_response(prompt)
        
        # Check if clarification is complete
        if "CLARIFICATION_COMPLETE" in response.upper():
            # Generate final clarified prompt
            final_prompt = self._generate_clarified_prompt(conversation_history, user_response)
            return {
                'complete': True,
                'clarified_prompt': final_prompt
            }
        else:
            return {
                'complete': False,
                'next_question': response
            }
    
    def _generate_clarified_prompt(self, history: List[Dict], final_response: str) -> str:
        """Generate the final clarified prompt based on the full conversation"""
        
        context = "Full clarification conversation:\n"
        for exchange in history:
            if 'question' in exchange:
                context += f"Q: {exchange['question']}\n"
            if 'response' in exchange:
                context += f"A: {exchange['response']}\n"
        context += f"Final response: {final_response}\n\n"
        
        system_override = """Based on this clarification dialogue, write a clear, specific, well-defined question that captures the user's true intent. This will be used for a structured debate. Make it precise and unambiguous."""
        
        return self._generate_response(context, system_override)

class ProponentAgent(BaseAgent):
    """
    Agent that argues in favor of a given position with strong supporting arguments.
    """
    
    def _get_system_prompt(self) -> str:
        return """You are a proponent in a structured debate. Your role is to build the strongest possible case supporting the given position.

Your approach:
- Present logical, well-reasoned arguments in favor of the position
- Use evidence, examples, and sound reasoning
- Consider multiple angles and perspectives that support your side
- Address potential counterarguments proactively when relevant
- Build upon previous rounds while adding new supporting points
- Be persuasive but intellectually honest

Structure your arguments clearly and build compelling cases that advocate for the position."""
    
    def generate_argument(self, context: str) -> str:
        """Generate a proponent argument based on debate context"""
        return self._generate_response(context)
    
    def should_use_tools(self, argument: str) -> bool:
        """Determine if this argument would benefit from additional tool research"""
        tool_indicators = [
            "recent data", "current statistics", "latest research", 
            "up-to-date information", "recent examples", "current trends"
        ]
        return any(indicator in argument.lower() for indicator in tool_indicators)
    
    def extract_search_query(self, argument: str) -> Optional[str]:
        """Extract a search query from the argument for tool use"""
        # Simple extraction - could be enhanced with NLP
        sentences = argument.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["data", "statistics", "research", "examples"]):
                # Extract key terms for search
                words = sentence.split()
                key_words = [word for word in words if len(word) > 3 and word.isalpha()]
                if key_words:
                    return ' '.join(key_words[:5])  # Take first 5 meaningful words
        return None
    
    def enhance_argument_with_data(self, base_argument: str, tool_data: Dict) -> str:
        """Enhance argument with data from tools"""
        enhancement_prompt = f"""Original argument: {base_argument}

Additional research data: {json.dumps(tool_data, indent=2)}

Enhance the original argument by incorporating relevant information from the research data. Maintain the same position but strengthen it with the additional evidence."""
        
        return self._generate_response(enhancement_prompt)

class OpponentAgent(BaseAgent):
    """
    Agent that argues against a given position with strong counterarguments.
    """
    
    def _get_system_prompt(self) -> str:
        return """You are an opponent in a structured debate. Your role is to provide the strongest possible critique and counterarguments against the given position.

Your approach:
- Challenge the position with logical counterarguments
- Identify weaknesses, flaws, or limitations in the proponent's reasoning
- Present alternative perspectives and solutions
- Use evidence and examples that undermine the position
- Point out unintended consequences or overlooked factors
- Build upon previous critique rounds while introducing new challenges
- Be rigorous but fair in your analysis

Structure your counterarguments clearly and build compelling cases against the position."""
    
    def generate_argument(self, context: str) -> str:
        """Generate an opponent argument based on debate context"""
        return self._generate_response(context)
    
    def should_use_tools(self, argument: str) -> bool:
        """Determine if this counterargument would benefit from additional research"""
        tool_indicators = [
            "studies show", "research indicates", "data suggests",
            "evidence points to", "statistics reveal", "examples include"
        ]
        return any(indicator in argument.lower() for indicator in tool_indicators)
    
    def extract_search_query(self, argument: str) -> Optional[str]:
        """Extract search query for tool-enhanced counterarguments"""
        # Look for claims that could be fact-checked or supported with data
        sentences = argument.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["research", "study", "data", "evidence"]):
                words = sentence.split()
                key_words = [word for word in words if len(word) > 3 and word.isalpha()]
                if key_words:
                    return ' '.join(key_words[:5])
        return None
    
    def enhance_argument_with_data(self, base_argument: str, tool_data: Dict) -> str:
        """Enhance counterargument with research data"""
        enhancement_prompt = f"""Original counterargument: {base_argument}

Additional research data: {json.dumps(tool_data, indent=2)}

Strengthen the counterargument by incorporating relevant contradictory evidence or alternative perspectives from the research data."""
        
        return self._generate_response(enhancement_prompt)

class DeciderAgent(BaseAgent):
    """
    Agent that synthesizes debate arguments into balanced insights.
    """
    
    def _get_system_prompt(self) -> str:
        return """You are a neutral decider synthesizing structured debates. Your role is to create balanced, insightful summaries that fairly represent all perspectives.

Your approach:
- Provide objective synthesis of the key arguments from all sides
- Identify areas of agreement and fundamental disagreements
- Highlight the strongest points made by each side
- Note any unresolved questions or areas needing further exploration
- Present nuanced insights that acknowledge complexity
- Avoid taking sides while providing clear analysis
- Reference specific arguments from the debate when relevant

Create synthesis that helps readers understand the full landscape of the issue."""
    
    def synthesize_debate(self, clarified_prompt: str, transcript: List[Dict], 
                         auditor_findings: Dict, tone: str = "neutral") -> str:
        """Synthesize completed debate into balanced insight"""
        
        # Build synthesis context
        context = f"Original question: {clarified_prompt}\n\n"
        context += "=== DEBATE TRANSCRIPT ===\n"
        
        for round_data in transcript:
            context += f"\nRound {round_data['round']}:\n"
            for role, arg_data in round_data.get('arguments', {}).items():
                context += f"{role.title()}: {arg_data['content']}\n"
        
        if auditor_findings:
            context += f"\n=== AUDITOR ANALYSIS ===\n{json.dumps(auditor_findings, indent=2)}\n"
        
        # Tone-specific instructions
        tone_instructions = {
            "neutral": "Provide a balanced, objective synthesis. Present both sides fairly.",
            "optimistic": "Focus on positive aspects, opportunities, and constructive elements.",
            "cautious": "Emphasize risks, uncertainties, and areas requiring careful consideration.",
            "skeptical": "Highlight weaknesses in arguments, unresolved issues, and remaining doubts."
        }
        
        synthesis_prompt = f"""{context}

Synthesize this debate with a {tone} tone. {tone_instructions.get(tone, tone_instructions['neutral'])}

Your synthesis should:
1. Summarize the key arguments from each side
2. Identify areas of agreement and disagreement
3. Note any unresolved questions or ghost loops
4. Provide clear, actionable insights
5. Reference specific arguments from the transcript

Be concise but comprehensive."""
        
        return self._generate_response(synthesis_prompt)

class AuditorAgent(BaseAgent):
    """
    Agent that checks debates for contradictions, drift, and logical issues.
    """
    
    def _get_system_prompt(self) -> str:
        return """You are an auditor checking debate quality and logical consistency. Your role is to identify issues that could undermine the debate's integrity.

Check for:
- Logical contradictions within or between arguments
- Factual inconsistencies or unsupported claims
- Topic drift away from the original question
- Hallucinations or obviously false statements
- Circular reasoning or logical fallacies
- Unaddressed counterarguments

Provide structured analysis with specific citations to round numbers and arguments. Be thorough but constructive in identifying issues."""
    
    def analyze_debate(self, full_transcript: str) -> Dict:
        """Analyze completed debate for issues"""
        
        analysis_prompt = f"""Analyze this debate transcript for logical and factual issues:

{full_transcript}

Provide analysis in the following structure:
1. Contradictions found (if any)
2. Factual concerns (if any) 
3. Topic drift assessment
4. Logical fallacies identified (if any)
5. Overall coherence rating (1-10)
6. Recommendations for improvement

Be specific and cite round numbers when identifying issues."""
        
        analysis_text = self._generate_response(analysis_prompt)
        
        # Parse analysis into structured format
        try:
            # Extract coherence rating
            coherence_match = re.search(r'coherence rating.*?(\d+)', analysis_text.lower())
            coherence_rating = int(coherence_match.group(1)) if coherence_match else 7
            
            # Check for serious issues
            serious_issues = any(keyword in analysis_text.lower() for keyword in [
                'contradiction', 'hallucination', 'false', 'unsupported', 'drift'
            ])
            
            return {
                'analysis': analysis_text,
                'coherence_rating': coherence_rating,
                'has_serious_issues': serious_issues,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Auditor analysis parsing failed: {str(e)}")
            return {
                'analysis': analysis_text,
                'coherence_rating': 5,  # Default middle rating
                'has_serious_issues': False,
                'timestamp': datetime.now().isoformat()
            }

class JournalingAssistant(BaseAgent):
    """
    Agent specialized for journaling, metadata generation, and voice transcription.
    """
    
    def __init__(self, model: str, config: Dict = None):
        super().__init__(model, config)
        self.whisper_model = None  # Lazy load Whisper
    
    def _get_system_prompt(self) -> str:
        return """You are a journaling assistant helping users organize and enrich their thoughts. Your role is to enhance journal entries with structured metadata and clear organization.

Your tasks:
- Rephrase content for clarity and journal-appropriate format
- Generate relevant tags based on content themes
- Assign appropriate weights for relevance, emotion, and priority (1-10 scale)
- Identify if entries represent unresolved "ghost loops" needing future attention
- Suggest connections to previous thoughts or themes
- Maintain the user's voice while improving organization

Always respond with structured metadata in JSON format when requested."""
    
    def generate_metadata(self, content: str, context: Dict = None) -> Dict:
        """Generate rich metadata for journal entry"""
        
        metadata_prompt = f"""Analyze this journal entry content and generate metadata:

Content: {content}

Context: {json.dumps(context, indent=2) if context else 'None provided'}

Generate metadata in this exact JSON format:
{{
    "title": "Brief descriptive title (3-8 words)",
    "tags": ["tag1", "tag2", "tag3"],
    "weights": {{
        "relevance": 8,    // 1-10: how relevant to user's goals/interests
        "emotion": 6,      // 1-10: emotional intensity/significance
        "priority": 7      // 1-10: how urgent/important for follow-up
    }},
    "ghost_loop": false,   // true if unresolved/needs future attention
    "ghost_loop_reason": "explanation if flagged as ghost loop",
    "summary": "One sentence summary of the main insight"
}}

Respond ONLY with valid JSON."""
        
        response = self._generate_response(metadata_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                
                # Validate and clamp weights
                weights = metadata.get('weights', {})
                for key in ['relevance', 'emotion', 'priority']:
                    if key in weights:
                        weights[key] = max(1, min(10, int(weights[key])))
                
                return metadata
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Metadata parsing failed: {str(e)}, using defaults")
            return self._generate_default_metadata(content)
    
    def _generate_default_metadata(self, content: str) -> Dict:
        """Generate default metadata when parsing fails"""
        return {
            'title': 'Journal Entry',
            'tags': ['general'],
            'weights': {'relevance': 5, 'emotion': 5, 'priority': 5},
            'ghost_loop': False,
            'ghost_loop_reason': '',
            'summary': content[:100] + '...' if len(content) > 100 else content
        }
    
    def rephrase_for_journal(self, content: str) -> str:
        """Rephrase content for journal entry format"""
        
        rephrase_prompt = f"""Rephrase this content for a personal journal entry. Make it clear, well-organized, and appropriate for future reflection:

Original content: {content}

Maintain the original meaning and the user's voice, but improve clarity and organization. Make it suitable for a private journal."""
        
        return self._generate_response(rephrase_prompt)
    
    def transcribe_voice(self, audio_path: str) -> str:
        """Transcribe audio file using local Whisper model"""
        try:
            if self.whisper_model is None:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("tiny")  # Fast, local transcription
            
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"].strip()
            
            logger.info(f"Voice transcribed: {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            logger.error(f"Voice transcription failed: {str(e)}")
            raise

class SpecialistAgent(BaseAgent):
    """
    Specialist agent for domain-specific expertise (ethics, statistics, etc.).
    """
    
    def __init__(self, specialist_type: str, model: str, config: Dict = None):
        self.specialist_type = specialist_type
        super().__init__(model, config)
    
    def _get_system_prompt(self) -> str:
        specialist_prompts = {
            'ethics': """You are an ethics specialist providing moral and ethical analysis. Focus on:
- Ethical implications and moral considerations
- Rights, duties, and obligations involved
- Potential harm or benefit to stakeholders
- Fairness and justice considerations
- Long-term ethical consequences""",
            
            'statistics': """You are a statistics and data analysis specialist. Focus on:
- Statistical validity and significance
- Data interpretation and methodology
- Quantitative analysis and metrics
- Research design and sampling issues
- Numerical evidence and trends""",
            
            'creativity': """You are a creativity and innovation specialist. Focus on:
- Alternative approaches and novel solutions
- Creative thinking and brainstorming
- Out-of-the-box perspectives
- Innovation opportunities
- Unconventional possibilities""",
            
            'technical': """You are a technical specialist providing technical analysis. Focus on:
- Technical feasibility and implementation
- Systems thinking and architecture
- Technical risks and constraints
- Engineering considerations
- Technology implications"""
        }
        
        return specialist_prompts.get(self.specialist_type, 
            f"You are a {self.specialist_type} specialist providing expert domain knowledge in your field.")
    
    def generate_argument(self, context: str) -> str:
        """Generate specialist perspective on the debate"""
        
        specialist_prompt = f"""{context}

As a {self.specialist_type} specialist, provide your expert perspective on this debate. Focus specifically on {self.specialist_type}-related aspects and considerations that others might miss."""
        
        return self._generate_response(specialist_prompt)

# Agent factory function
def create_agent(agent_type: str, model: str, config: Dict = None, **kwargs) -> BaseAgent:
    """Factory function to create agent instances"""
    
    agent_classes = {
        'clarifier': StoicClarifier,
        'proponent': ProponentAgent,
        'opponent': OpponentAgent,
        'decider': DeciderAgent,
        'auditor': AuditorAgent,
        'journaling_assistant': JournalingAssistant,
        'specialist': SpecialistAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agent_classes[agent_type]
    
    # Handle specialist agents with additional parameters
    if agent_type == 'specialist':
        specialist_type = kwargs.get('specialist_type', 'general')
        return agent_class(specialist_type, model, config)
    else:
        return agent_class(model, config)

# Agent management utilities
class AgentPool:
    """Manages a pool of agent instances for reuse and efficiency"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.usage_stats = {}
    
    def get_or_create_agent(self, agent_type: str, model: str, config: Dict = None, **kwargs) -> BaseAgent:
        """Get existing agent or create new one"""
        agent_key = f"{agent_type}_{model}"
        
        if agent_key not in self.agents:
            self.agents[agent_key] = create_agent(agent_type, model, config, **kwargs)
            self.usage_stats[agent_key] = {'created_at': datetime.now(), 'usage_count': 0}
        
        self.usage_stats[agent_key]['usage_count'] += 1
        self.usage_stats[agent_key]['last_used'] = datetime.now()
        
        return self.agents[agent_key]
    
    def reset_agent_context(self, agent_type: str, model: str):
        """Reset specific agent's context"""
        agent_key = f"{agent_type}_{model}"
        if agent_key in self.agents:
            self.agents[agent_key].reset_context()
    
    def get_pool_stats(self) -> Dict:
        """Get statistics about the agent pool"""
        return {
            'total_agents': len(self.agents),
            'agent_types': list(set(key.split('_')[0] for key in self.agents.keys())),
            'usage_stats': self.usage_stats
        }
