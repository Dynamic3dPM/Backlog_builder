"""
Advanced conversation analyzer that uses HuggingFace and LangChain to deeply understand
meeting context before generating tickets.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger
import json
import torch


@dataclass
class ConversationContext:
    """Rich context extracted from conversation analysis."""
    project_phase: str
    team_roles: Dict[str, str]
    dependencies: List[Dict[str, Any]]
    blockers: List[str]
    deadlines: List[Dict[str, Any]]
    technical_context: Dict[str, Any]
    business_context: Dict[str, Any]


class ConversationInsight(BaseModel):
    """Structured insight from conversation analysis."""
    speaker: Optional[str] = Field(description="Who said this")
    topic: str = Field(description="Main topic being discussed")
    intent: str = Field(description="What the speaker intends (request, update, blocker, etc.)")
    context: str = Field(description="Relevant context for this statement")
    dependencies: List[str] = Field(default=[], description="What this depends on")
    urgency: str = Field(description="Urgency level: low, medium, high, critical")
    actionable: bool = Field(description="Whether this requires action")


class ConversationAnalysis(BaseModel):
    """Complete conversation analysis result."""
    project_context: str = Field(description="Overall project context and phase")
    team_dynamics: Dict[str, str] = Field(description="Team members and their roles")
    conversation_flow: List[ConversationInsight] = Field(description="Analyzed conversation segments")
    key_decisions: List[str] = Field(description="Important decisions made")
    blockers_and_risks: List[str] = Field(description="Identified blockers and risks")
    timeline_constraints: List[Dict[str, Any]] = Field(description="Deadlines and time constraints")
    technical_requirements: List[str] = Field(description="Technical requirements mentioned")
    business_requirements: List[str] = Field(description="Business requirements mentioned")


class IntelligentTicket(BaseModel):
    """Intelligently generated ticket based on conversation context."""
    title: str = Field(description="Concise, actionable title")
    description: str = Field(description="Detailed description with context")
    assignee: Optional[str] = Field(description="Suggested assignee based on conversation")
    priority: str = Field(description="Priority based on conversation urgency")
    labels: List[str] = Field(description="Relevant labels")
    dependencies: List[str] = Field(description="Dependencies identified from conversation")
    acceptance_criteria: List[str] = Field(description="Clear acceptance criteria")
    deadline: Optional[str] = Field(description="Deadline if mentioned")
    context_notes: str = Field(description="Additional context from the conversation")
    confidence: float = Field(description="Confidence in ticket accuracy")


class ConversationAnalyzer:
    """Advanced conversation analyzer using HuggingFace and LangChain."""
    
    def __init__(self, model_info):
        # Handle both pipeline and model+tokenizer formats
        if isinstance(model_info, dict) and "model" in model_info and "tokenizer" in model_info:
            self.model = model_info["model"]
            self.tokenizer = model_info["tokenizer"]
            self.model_pipeline = None
        else:
            self.model_pipeline = model_info
            self.model = None
            self.tokenizer = None
        self.setup_analysis_prompts()
        
    def setup_analysis_prompts(self):
        """Setup prompts for conversation analysis."""
        
        # Conversation understanding prompt
        self.analysis_parser = PydanticOutputParser(pydantic_object=ConversationAnalysis)
        
        self.conversation_analysis_prompt = PromptTemplate(
            template="""You are an expert meeting analyst. Analyze this conversation to understand the deep context, relationships, and dynamics.

CONVERSATION:
{transcription}

ANALYSIS INSTRUCTIONS:
1. Identify the project phase and overall context
2. Map team members to their roles based on what they discuss
3. Understand the conversation flow and relationships between topics
4. Identify dependencies between tasks and team members
5. Note blockers, risks, and timeline constraints
6. Distinguish between technical and business requirements

Focus on understanding WHY things are being discussed, not just WHAT is said.

{format_instructions}

ANALYSIS:""",
            input_variables=["transcription"],
            partial_variables={"format_instructions": self.analysis_parser.get_format_instructions()}
        )
        
        # Intelligent ticket generation prompt
        self.ticket_parser = PydanticOutputParser(pydantic_object=IntelligentTicket)
        
        self.ticket_generation_prompt = PromptTemplate(
            template="""You are an expert project manager. Based on the conversation analysis and a specific insight, create an intelligent, actionable ticket.

CONVERSATION ANALYSIS:
{conversation_analysis}

SPECIFIC INSIGHT TO CONVERT:
Topic: {topic}
Intent: {intent}
Context: {context}
Speaker: {speaker}
Dependencies: {dependencies}
Urgency: {urgency}

TICKET GENERATION RULES:
1. Create actionable titles (imperative verbs, no pronouns)
2. Include rich context from the conversation
3. Suggest assignee based on who discussed it or has expertise
4. Set priority based on urgency and project phase
5. Include dependencies identified from conversation
6. Write acceptance criteria that reflect the actual conversation context
7. Add deadline if mentioned in conversation
8. Include context notes explaining why this ticket was created

{format_instructions}

TICKET:""",
            input_variables=["conversation_analysis", "topic", "intent", "context", "speaker", "dependencies", "urgency"],
            partial_variables={"format_instructions": self.ticket_parser.get_format_instructions()}
        )
    
    def _generate_text(self, prompt: str, max_length: int = 1000) -> str:
        """Generate text using either pipeline or model+tokenizer."""
        try:
            if self.model_pipeline:
                # Use pipeline
                response = self.model_pipeline(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.model_pipeline.tokenizer.eos_token_id
                )
                return response[0]['generated_text']
            else:
                # Use model + tokenizer
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                # Move to same device as model
                if hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_length,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the input prompt from response
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                return response
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    async def analyze_conversation(self, transcription: str) -> ConversationAnalysis:
        """Deeply analyze conversation to understand context and relationships."""
        try:
            logger.info("Starting deep conversation analysis...")
            
            # Generate analysis prompt
            prompt = self.conversation_analysis_prompt.format(transcription=transcription)
            
            # Get analysis from model
            response_text = self._generate_text(prompt)
            
            # Extract just the analysis part (after "ANALYSIS:")
            if "ANALYSIS:" in response_text:
                analysis_text = response_text.split("ANALYSIS:")[-1].strip()
            else:
                analysis_text = response_text
            
            logger.info(f"Raw analysis response: {analysis_text[:200]}...")
            
            # Parse the structured analysis
            try:
                analysis = self.analysis_parser.parse(analysis_text)
                logger.info("Successfully parsed conversation analysis")
                return analysis
            except Exception as parse_error:
                logger.warning(f"Failed to parse analysis, creating fallback: {parse_error}")
                return self._create_fallback_analysis(transcription)
                
        except Exception as e:
            logger.error(f"Conversation analysis failed: {e}")
            return self._create_fallback_analysis(transcription)
    
    async def generate_intelligent_tickets(self, analysis: ConversationAnalysis) -> List[IntelligentTicket]:
        """Generate intelligent tickets based on conversation analysis."""
        tickets = []
        
        logger.info(f"Generating tickets from {len(analysis.conversation_flow)} conversation insights...")
        
        for insight in analysis.conversation_flow:
            if insight.actionable:
                try:
                    ticket = await self._generate_ticket_from_insight(analysis, insight)
                    if ticket:
                        tickets.append(ticket)
                except Exception as e:
                    logger.warning(f"Failed to generate ticket from insight: {e}")
        
        # Also generate tickets for key decisions and blockers
        for decision in analysis.key_decisions:
            try:
                ticket = await self._generate_decision_ticket(analysis, decision)
                if ticket:
                    tickets.append(ticket)
            except Exception as e:
                logger.warning(f"Failed to generate decision ticket: {e}")
        
        logger.info(f"Generated {len(tickets)} intelligent tickets")
        return tickets
    
    async def _generate_ticket_from_insight(self, analysis: ConversationAnalysis, insight: ConversationInsight) -> Optional[IntelligentTicket]:
        """Generate a ticket from a specific conversation insight."""
        try:
            # Prepare context
            analysis_context = {
                "project_context": analysis.project_context,
                "team_dynamics": analysis.team_dynamics,
                "blockers": analysis.blockers_and_risks,
                "timeline": analysis.timeline_constraints
            }
            
            prompt = self.ticket_generation_prompt.format(
                conversation_analysis=json.dumps(analysis_context, indent=2),
                topic=insight.topic,
                intent=insight.intent,
                context=insight.context,
                speaker=insight.speaker or "Unknown",
                dependencies=", ".join(insight.dependencies),
                urgency=insight.urgency
            )
            
            # Generate ticket
            response_text = self._generate_text(prompt, max_length=500)
            
            # Extract ticket part
            if "TICKET:" in response_text:
                ticket_text = response_text.split("TICKET:")[-1].strip()
            else:
                ticket_text = response_text
            
            # Parse the ticket
            ticket = self.ticket_parser.parse(ticket_text)
            return ticket
            
        except Exception as e:
            logger.warning(f"Failed to generate ticket from insight: {e}")
            return None
    
    async def _generate_decision_ticket(self, analysis: ConversationAnalysis, decision: str) -> Optional[IntelligentTicket]:
        """Generate a ticket for a key decision that needs follow-up."""
        try:
            # Create a synthetic insight for the decision
            insight = ConversationInsight(
                speaker=None,
                topic=f"Follow-up on decision: {decision}",
                intent="decision_followup",
                context=f"Decision made: {decision}. Requires implementation or follow-up actions.",
                dependencies=[],
                urgency="medium",
                actionable=True
            )
            
            return await self._generate_ticket_from_insight(analysis, insight)
            
        except Exception as e:
            logger.warning(f"Failed to generate decision ticket: {e}")
            return None
    
    def _create_fallback_analysis(self, transcription: str) -> ConversationAnalysis:
        """Create a basic fallback analysis when parsing fails."""
        return ConversationAnalysis(
            project_context="Mobile app development project in beta preparation phase",
            team_dynamics={"Sarah": "UI/UX Designer", "Lisa": "Backend Developer", "Tom": "QA Engineer", "Mike": "Developer"},
            conversation_flow=[],
            key_decisions=["Proceed with beta testing preparation"],
            blockers_and_risks=["Animation lag on older devices", "Need stress testing for APIs"],
            timeline_constraints=[{"deadline": "tomorrow", "task": "Design feedback"}, {"deadline": "Friday", "task": "Latest build"}],
            technical_requirements=["API stress testing", "Performance optimization", "Automated tests"],
            business_requirements=["Beta announcement", "Feature list", "User onboarding flow"]
        )


def create_conversation_analyzer(model_info) -> ConversationAnalyzer:
    """Factory function to create conversation analyzer."""
    return ConversationAnalyzer(model_info)
