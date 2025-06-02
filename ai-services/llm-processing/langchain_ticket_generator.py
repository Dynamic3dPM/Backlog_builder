"""
LangChain-based Ticket Generator for improved prompt engineering and output parsing.
Solves issues with inconsistent LLM outputs and poor title extraction.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import BaseOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
import json
import re
from loguru import logger


class TicketOutput(BaseModel):
    """Structured output model for generated tickets."""
    title: str = Field(description="Short, actionable title (5-7 words max)")
    description: str = Field(description="User story description")
    acceptance_criteria: List[str] = Field(description="List of acceptance criteria")
    priority: str = Field(description="Priority level: High, Medium, or Low")
    labels: List[str] = Field(description="List of relevant labels")
    estimated_effort: Optional[str] = Field(description="Effort estimation", default=None)


class LangChainTicketGenerator:
    """LangChain-based ticket generator with structured output parsing."""
    
    def __init__(self, model_pipeline=None):
        self.model_pipeline = model_pipeline
        self.llm = None
        self.setup_langchain()
        
    def setup_langchain(self):
        """Initialize LangChain components."""
        if self.model_pipeline:
            # Wrap HuggingFace pipeline in LangChain
            self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)
        
        # Setup output parser
        self.output_parser = PydanticOutputParser(pydantic_object=TicketOutput)
        
        # Create fixing parser to handle malformed outputs
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.output_parser,
            llm=self.llm
        ) if self.llm else self.output_parser
        
        # Setup few-shot examples
        self.examples = [
            {
                "action_item": "Implement user authentication",
                "context": "Users need secure login",
                "output": '{"title": "Implement User Authentication", "description": "As a User, I want to authenticate securely so that I can access my account.", "acceptance_criteria": ["User can register", "User can login", "Password reset works"], "priority": "High", "labels": ["authentication", "security"]}'
            }
        ]
        
        # Create few-shot prompt template
        example_template = """Action: {action_item}
Context: {context}
Output: {output}"""
        
        example_prompt = PromptTemplate(
            input_variables=["action_item", "context", "output"],
            template=example_template
        )
        
        # Main prompt template - simplified for T5 models
        self.prompt_template = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix="""Create a structured ticket from action items. Output valid JSON only.

{format_instructions}

Example:""",
            suffix="""Action: {action_item}
Context: {context}
Output:""",
            input_variables=["action_item", "context"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        # Create the chain
        if self.llm:
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                verbose=True
            )
    
    def generate_ticket(self, action_item: str, context: str, ticket_type: str = "task") -> Dict[str, Any]:
        """Generate a structured ticket using LangChain."""
        try:
            if not self.llm:
                logger.warning("No LLM available, using fallback generation")
                return self._fallback_generation(action_item, context, ticket_type)
            
            # Generate response using the chain
            response = self.chain.run(
                action_item=action_item,
                context=context
            )
            
            logger.info(f"Raw LLM response: {response}")
            
            # Parse the structured output
            try:
                parsed_output = self.fixing_parser.parse(response)
                return {
                    "title": self._clean_title(parsed_output.title),
                    "description": parsed_output.description,
                    "acceptance_criteria": parsed_output.acceptance_criteria,
                    "priority": parsed_output.priority.lower(),
                    "labels": parsed_output.labels + [ticket_type],
                    "estimated_effort": parsed_output.estimated_effort,
                    "confidence": 0.9
                }
            except Exception as parse_error:
                logger.error(f"Failed to parse structured output: {parse_error}")
                # Try to extract manually
                return self._manual_extraction(response, ticket_type)
                
        except Exception as e:
            logger.error(f"LangChain ticket generation failed: {e}")
            return self._fallback_generation(action_item, context, ticket_type)
    
    def _clean_title(self, title: str) -> str:
        """Clean and optimize the title."""
        # Remove common prefixes and suffixes
        title = re.sub(r'^(Implement|Create|Add|Fix|Update|Build|Setup|Configure)\s+', '', title, flags=re.IGNORECASE)
        
        # Remove personal pronouns
        pronouns = ['we need to', 'we should', 'I will', 'you should', 'the team needs to']
        for pronoun in pronouns:
            title = re.sub(pronoun, '', title, flags=re.IGNORECASE).strip()
        
        # Ensure it starts with an action verb
        action_verbs = ['Implement', 'Create', 'Fix', 'Update', 'Add', 'Remove', 'Configure', 'Build', 'Deploy']
        if not any(title.startswith(verb) for verb in action_verbs):
            # Try to infer the action
            if 'bug' in title.lower() or 'error' in title.lower():
                title = f"Fix {title}"
            elif 'new' in title.lower() or 'add' in title.lower():
                title = f"Implement {title}"
            else:
                title = f"Update {title}"
        
        # Limit to 40 characters with smart truncation
        if len(title) > 40:
            words = title.split()
            truncated = []
            length = 0
            for word in words:
                if length + len(word) + 1 <= 40:
                    truncated.append(word)
                    length += len(word) + 1
                else:
                    break
            title = ' '.join(truncated)
        
        return title.strip()
    
    def _manual_extraction(self, response: str, ticket_type: str) -> Dict[str, Any]:
        """Manually extract ticket information from unstructured response."""
        lines = response.split('\n')
        
        title = "Update Task"
        description = "Task description not available"
        acceptance_criteria = ["Implementation complete", "Tests passing"]
        priority = "medium"
        labels = [ticket_type]
        
        for line in lines:
            line = line.strip()
            if line.startswith('"title"') or line.startswith('title:'):
                title = re.sub(r'["\':,]', '', line.split(':', 1)[1]).strip()
            elif line.startswith('"description"') or line.startswith('description:'):
                description = re.sub(r'["\':,]', '', line.split(':', 1)[1]).strip()
            elif line.startswith('"priority"') or line.startswith('priority:'):
                priority = re.sub(r'["\':,]', '', line.split(':', 1)[1]).strip().lower()
        
        return {
            "title": self._clean_title(title),
            "description": description,
            "acceptance_criteria": acceptance_criteria,
            "priority": priority,
            "labels": labels,
            "estimated_effort": None,
            "confidence": 0.6
        }
    
    def _fallback_generation(self, action_item: str, context: str, ticket_type: str) -> Dict[str, Any]:
        """Fallback generation when LangChain fails."""
        # Extract title from action item
        title = action_item.replace("We need to", "").replace("we should", "").strip()
        title = self._clean_title(title)
        
        # Create basic description
        description = f"As a User, I want to {action_item.lower()} so that {context[:100]}..."
        
        return {
            "title": title,
            "description": description,
            "acceptance_criteria": ["Implementation complete", "Tests passing", "Documentation updated"],
            "priority": "medium",
            "labels": [ticket_type],
            "estimated_effort": None,
            "confidence": 0.5
        }


# Integration function for existing codebase
def create_langchain_generator(model_pipeline=None) -> LangChainTicketGenerator:
    """Factory function to create LangChain ticket generator."""
    return LangChainTicketGenerator(model_pipeline)
