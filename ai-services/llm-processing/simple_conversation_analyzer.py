"""
Simplified but intelligent conversation analyzer that focuses on extracting
actionable insights and generating contextual tickets.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import json


@dataclass
class ConversationInsight:
    """A single actionable insight from the conversation."""
    speaker: Optional[str]
    action: str
    context: str
    deadline: Optional[str]
    assignee: Optional[str]
    priority: str
    dependencies: List[str]
    confidence: float


@dataclass
class IntelligentTicket:
    """An intelligently generated ticket with rich context."""
    title: str
    description: str
    assignee: Optional[str]
    priority: str
    labels: List[str]
    dependencies: List[str]
    acceptance_criteria: List[str]
    deadline: Optional[str]
    context_notes: str
    confidence: float


class SimpleConversationAnalyzer:
    """Simplified conversation analyzer that extracts actionable insights."""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Setup regex patterns for extracting insights."""
        
        # Patterns for identifying speakers
        self.speaker_patterns = [
            r"(\w+),?\s+(?:can you|could you|please)",
            r"(\w+):\s+",
            r"(\w+)\s+said",
            r"(\w+)\s+mentioned"
        ]
        
        # Patterns for deadlines
        self.deadline_patterns = [
            r"by\s+(end\s+of\s+day|EOD|tomorrow|friday|wednesday|monday|tuesday|thursday|saturday|sunday)",
            r"by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            r"deadline\s+(?:is\s+)?(\w+)",
            r"due\s+(\w+)"
        ]
        
        # Patterns for action items
        self.action_patterns = [
            r"(?:I\s+)?(?:need|want|should|will|can)\s+(?:to\s+)?(\w+(?:\s+\w+)*)",
            r"(?:we\s+)?(?:need|should|will|must)\s+(?:to\s+)?(\w+(?:\s+\w+)*)",
            r"(?:let's|let\s+us)\s+(\w+(?:\s+\w+)*)",
            r"(?:please|can\s+you)\s+(\w+(?:\s+\w+)*)",
            r"(?:I'll|I\s+will)\s+(\w+(?:\s+\w+)*)"
        ]
        
        # Priority indicators
        self.priority_indicators = {
            "critical": ["critical", "urgent", "asap", "immediately", "blocker"],
            "high": ["important", "priority", "soon", "quickly", "high"],
            "medium": ["should", "need", "when possible"],
            "low": ["eventually", "later", "nice to have", "low"]
        }
        
        # Technical keywords
        self.technical_keywords = [
            "api", "endpoint", "database", "authentication", "testing", "deployment",
            "framework", "component", "animation", "performance", "optimization",
            "stress test", "load", "build", "react native", "ui", "ux"
        ]
        
        # Business keywords
        self.business_keywords = [
            "beta", "announcement", "feature", "user", "onboarding", "marketing",
            "customer", "feedback", "requirements", "stakeholder"
        ]
    
    def analyze_conversation(self, transcription: str) -> Dict[str, Any]:
        """Analyze conversation and extract actionable insights."""
        logger.info("Starting simple conversation analysis...")
        
        # Split into sentences for analysis
        sentences = self._split_into_sentences(transcription)
        
        # Extract insights from each sentence
        insights = []
        for sentence in sentences:
            insight = self._extract_insight_from_sentence(sentence)
            if insight:
                insights.append(insight)
        
        # Identify team members and roles
        team_dynamics = self._identify_team_dynamics(transcription)
        
        # Extract project context
        project_context = self._extract_project_context(transcription)
        
        # Generate intelligent tickets
        tickets = self._generate_tickets_from_insights(insights, team_dynamics, project_context)
        
        return {
            "success": True,
            "conversation_analysis": {
                "project_context": project_context,
                "team_dynamics": team_dynamics,
                "insights_found": len(insights),
                "actionable_insights": len([i for i in insights if i.action]),
                "technical_mentions": self._count_technical_mentions(transcription),
                "business_mentions": self._count_business_mentions(transcription)
            },
            "intelligent_tickets": [self._ticket_to_dict(ticket) for ticket in tickets],
            "insights_analyzed": len(insights),
            "actionable_insights": len([i for i in insights if i.action]),
            "processing_metadata": {
                "analysis_method": "simple_conversation_analysis",
                "pattern_based": True,
                "context_aware": True
            }
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_insight_from_sentence(self, sentence: str) -> Optional[ConversationInsight]:
        """Extract actionable insight from a single sentence."""
        sentence_lower = sentence.lower()
        
        # Skip if sentence is too short or doesn't contain action words
        if len(sentence.split()) < 3:
            return None
        
        # Extract speaker
        speaker = self._extract_speaker(sentence)
        
        # Extract action
        action = self._extract_action(sentence)
        if not action:
            return None
        
        # Extract deadline
        deadline = self._extract_deadline(sentence)
        
        # Determine priority
        priority = self._determine_priority(sentence)
        
        # Extract assignee (if different from speaker)
        assignee = self._extract_assignee(sentence, speaker)
        
        # Calculate confidence
        confidence = self._calculate_confidence(sentence, action, speaker, deadline)
        
        return ConversationInsight(
            speaker=speaker,
            action=action,
            context=sentence,
            deadline=deadline,
            assignee=assignee,
            priority=priority,
            dependencies=[],
            confidence=confidence
        )
    
    def _extract_speaker(self, sentence: str) -> Optional[str]:
        """Extract speaker from sentence."""
        for pattern in self.speaker_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                speaker = match.group(1).strip()
                if len(speaker) < 20 and speaker.isalpha():  # Reasonable speaker name
                    return speaker.title()
        return None
    
    def _extract_action(self, sentence: str) -> Optional[str]:
        """Extract action from sentence."""
        for pattern in self.action_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                # Clean up the action
                action = re.sub(r'\b(the|a|an|that|which|this)\b', '', action, flags=re.IGNORECASE).strip()
                if len(action) > 5 and len(action) < 100:  # Reasonable action length
                    return action
        return None
    
    def _extract_deadline(self, sentence: str) -> Optional[str]:
        """Extract deadline from sentence."""
        for pattern in self.deadline_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None
    
    def _determine_priority(self, sentence: str) -> str:
        """Determine priority based on sentence content."""
        sentence_lower = sentence.lower()
        
        for priority, keywords in self.priority_indicators.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return priority
        
        # Default priority based on deadline
        if any(word in sentence_lower for word in ["tomorrow", "eod", "asap"]):
            return "high"
        elif any(word in sentence_lower for word in ["friday", "week", "soon"]):
            return "medium"
        
        return "medium"
    
    def _extract_assignee(self, sentence: str, speaker: Optional[str]) -> Optional[str]:
        """Extract assignee from sentence."""
        # Look for direct assignments
        assignee_patterns = [
            r"(\w+),?\s+(?:can you|could you|please)",
            r"assign(?:ed)?\s+to\s+(\w+)",
            r"(\w+)\s+will\s+(?:handle|do|work on)"
        ]
        
        for pattern in assignee_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                assignee = match.group(1).strip()
                if assignee.lower() != speaker.lower() if speaker else True:
                    return assignee.title()
        
        return speaker  # Default to speaker if no specific assignee
    
    def _calculate_confidence(self, sentence: str, action: str, speaker: Optional[str], deadline: Optional[str]) -> float:
        """Calculate confidence score for the insight."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear indicators
        if speaker:
            confidence += 0.2
        if deadline:
            confidence += 0.2
        if any(word in sentence.lower() for word in ["will", "need", "should", "must"]):
            confidence += 0.1
        if len(action.split()) > 1:  # Multi-word actions are usually more specific
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _identify_team_dynamics(self, transcription: str) -> Dict[str, str]:
        """Identify team members and their likely roles."""
        team_dynamics = {}
        
        # Common role indicators
        role_indicators = {
            "designer": ["ui", "ux", "design", "wireframe", "color", "layout"],
            "developer": ["code", "build", "component", "framework", "react", "api"],
            "qa": ["test", "testing", "qa", "quality", "functional"],
            "manager": ["schedule", "deadline", "team", "sync", "meeting"],
            "backend": ["api", "database", "server", "endpoint", "authentication"],
            "frontend": ["ui", "component", "animation", "screen"]
        }
        
        # Extract potential names
        names = re.findall(r'\b[A-Z][a-z]+\b', transcription)
        names = [name for name in names if len(name) > 2 and name not in ["Good", "Thanks", "Sure", "Great"]]
        
        # Assign roles based on context
        for name in set(names):
            context = self._get_context_around_name(transcription, name)
            role = self._determine_role_from_context(context, role_indicators)
            if role:
                team_dynamics[name] = role
        
        return team_dynamics
    
    def _get_context_around_name(self, text: str, name: str) -> str:
        """Get context around a name mention."""
        # Find sentences containing the name
        sentences = self._split_into_sentences(text)
        context_sentences = [s for s in sentences if name in s]
        return " ".join(context_sentences)
    
    def _determine_role_from_context(self, context: str, role_indicators: Dict[str, List[str]]) -> Optional[str]:
        """Determine role based on context."""
        context_lower = context.lower()
        role_scores = {}
        
        for role, keywords in role_indicators.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                role_scores[role] = score
        
        if role_scores:
            return max(role_scores, key=role_scores.get)
        return None
    
    def _extract_project_context(self, transcription: str) -> str:
        """Extract overall project context."""
        context_indicators = {
            "mobile app": ["mobile", "app", "react native", "ios", "android"],
            "web development": ["web", "website", "frontend", "backend"],
            "beta testing": ["beta", "testing", "qa", "release"],
            "feature development": ["feature", "functionality", "development"],
            "bug fixing": ["bug", "fix", "issue", "problem"]
        }
        
        transcription_lower = transcription.lower()
        context_scores = {}
        
        for context, keywords in context_indicators.items():
            score = sum(1 for keyword in keywords if keyword in transcription_lower)
            if score > 0:
                context_scores[context] = score
        
        if context_scores:
            primary_context = max(context_scores, key=context_scores.get)
            return f"{primary_context} project"
        
        return "software development project"
    
    def _count_technical_mentions(self, transcription: str) -> int:
        """Count technical keyword mentions."""
        transcription_lower = transcription.lower()
        return sum(1 for keyword in self.technical_keywords if keyword in transcription_lower)
    
    def _count_business_mentions(self, transcription: str) -> int:
        """Count business keyword mentions."""
        transcription_lower = transcription.lower()
        return sum(1 for keyword in self.business_keywords if keyword in transcription_lower)
    
    def _generate_tickets_from_insights(self, insights: List[ConversationInsight], team_dynamics: Dict[str, str], project_context: str) -> List[IntelligentTicket]:
        """Generate intelligent tickets from insights."""
        tickets = []
        
        for insight in insights:
            if insight.action and insight.confidence > 0.6:
                ticket = self._create_ticket_from_insight(insight, team_dynamics, project_context)
                if ticket:
                    tickets.append(ticket)
        
        return tickets
    
    def _create_ticket_from_insight(self, insight: ConversationInsight, team_dynamics: Dict[str, str], project_context: str) -> Optional[IntelligentTicket]:
        """Create a ticket from a single insight."""
        try:
            # Generate title
            title = self._generate_ticket_title(insight.action)
            
            # Generate description
            description = self._generate_ticket_description(insight, project_context)
            
            # Determine labels
            labels = self._determine_ticket_labels(insight, project_context)
            
            # Generate acceptance criteria
            acceptance_criteria = self._generate_acceptance_criteria(insight)
            
            return IntelligentTicket(
                title=title,
                description=description,
                assignee=insight.assignee,
                priority=insight.priority,
                labels=labels,
                dependencies=insight.dependencies,
                acceptance_criteria=acceptance_criteria,
                deadline=insight.deadline,
                context_notes=f"Generated from conversation: '{insight.context[:100]}...'",
                confidence=insight.confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to create ticket from insight: {e}")
            return None
    
    def _generate_ticket_title(self, action: str) -> str:
        """Generate a concise ticket title."""
        # Clean up action
        title = action.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["to ", "the ", "a ", "an "]
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix):
                title = title[len(prefix):]
        
        # Ensure it starts with a verb
        if not any(title.lower().startswith(verb) for verb in ["implement", "create", "fix", "update", "add", "remove", "setup", "build", "test", "review"]):
            if "test" in title.lower():
                title = f"Test {title}"
            elif "review" in title.lower():
                title = f"Review {title}"
            elif "setup" in title.lower() or "set up" in title.lower():
                title = f"Setup {title}"
            else:
                title = f"Implement {title}"
        
        # Capitalize first letter
        title = title[0].upper() + title[1:] if title else ""
        
        # Limit length
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title
    
    def _generate_ticket_description(self, insight: ConversationInsight, project_context: str) -> str:
        """Generate ticket description with context."""
        description = f"Context: {project_context}\n\n"
        description += f"Requirement: {insight.action}\n\n"
        description += f"Background: {insight.context}\n\n"
        
        if insight.speaker:
            description += f"Requested by: {insight.speaker}\n\n"
        
        if insight.deadline:
            description += f"Deadline: {insight.deadline}\n\n"
        
        return description
    
    def _determine_ticket_labels(self, insight: ConversationInsight, project_context: str) -> List[str]:
        """Determine appropriate labels for the ticket."""
        labels = []
        
        # Add priority label
        labels.append(insight.priority)
        
        # Add type labels based on action
        action_lower = insight.action.lower()
        if any(word in action_lower for word in ["test", "testing"]):
            labels.append("testing")
        elif any(word in action_lower for word in ["fix", "bug", "issue"]):
            labels.append("bug")
        elif any(word in action_lower for word in ["feature", "new", "add"]):
            labels.append("feature")
        elif any(word in action_lower for word in ["review", "feedback"]):
            labels.append("review")
        else:
            labels.append("task")
        
        # Add technical labels
        if any(word in insight.context.lower() for word in ["api", "backend", "database"]):
            labels.append("backend")
        elif any(word in insight.context.lower() for word in ["ui", "frontend", "component"]):
            labels.append("frontend")
        
        return labels
    
    def _generate_acceptance_criteria(self, insight: ConversationInsight) -> List[str]:
        """Generate acceptance criteria for the ticket."""
        criteria = []
        
        # Basic completion criteria
        criteria.append(f"{insight.action} is completed")
        
        # Add specific criteria based on action type
        action_lower = insight.action.lower()
        if "test" in action_lower:
            criteria.append("All test cases pass")
            criteria.append("Test results documented")
        elif "review" in action_lower:
            criteria.append("Review feedback provided")
            criteria.append("Action items from review identified")
        elif "api" in insight.context.lower():
            criteria.append("API endpoints tested")
            criteria.append("Documentation updated")
        
        # Add deadline criteria if present
        if insight.deadline:
            criteria.append(f"Completed by {insight.deadline}")
        
        return criteria
    
    def _ticket_to_dict(self, ticket: IntelligentTicket) -> Dict[str, Any]:
        """Convert ticket to dictionary format."""
        return {
            "title": ticket.title,
            "description": ticket.description,
            "assignee": ticket.assignee,
            "priority": ticket.priority,
            "labels": ticket.labels,
            "dependencies": ticket.dependencies,
            "acceptance_criteria": ticket.acceptance_criteria,
            "deadline": ticket.deadline,
            "context_notes": ticket.context_notes,
            "confidence": ticket.confidence,
            "estimated_effort": None
        }


def create_simple_conversation_analyzer() -> SimpleConversationAnalyzer:
    """Factory function to create simple conversation analyzer."""
    return SimpleConversationAnalyzer()
