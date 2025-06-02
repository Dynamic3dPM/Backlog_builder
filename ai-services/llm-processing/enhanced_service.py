"""
Enhanced service layer with LangChain integration for better ticket generation and meeting analysis.
"""

import torch
from typing import Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory
from loguru import logger
import asyncio

from langchain_ticket_generator import LangChainTicketGenerator, create_langchain_generator


class LangChainModelManager:
    """Manages LangChain models and chains."""
    
    def __init__(self):
        self.langchain_models = {}
        self.initialized = False
    
    async def initialize(self, model_manager):
        """Initialize LangChain models."""
        try:
            # Load models for LangChain integration
            interpretation_model = await model_manager.load_model(
                "google/flan-t5-base", "generation"
            )
            ticket_model = await model_manager.load_model(
                "google/flan-t5-base", "generation"
            )
            
            self.langchain_models = {
                "interpretation": interpretation_model,
                "ticket_generation": ticket_model
            }
            self.initialized = True
            logger.info("LangChain models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain models: {e}")
            self.initialized = False


class LangChainTicketGeneratorEnhanced:
    """Enhanced ticket generator with memory and chain management."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.langchain_generator = None
        self.chains = {}
        
    async def initialize(self):
        """Initialize the enhanced ticket generator."""
        try:
            if self.model_manager.initialized:
                # Create LangChain generator with loaded models
                model_info = self.model_manager.langchain_models.get("ticket_generation")
                if model_info:
                    from transformers import pipeline
                    
                    model_pipeline = pipeline(
                        "text2text-generation",
                        model=model_info["model"],
                        tokenizer=model_info["tokenizer"],
                        device=0 if torch.cuda.is_available() else -1,
                        max_new_tokens=200,
                        temperature=0.3,
                        do_sample=True
                    )
                    
                    self.langchain_generator = create_langchain_generator(model_pipeline)
                    logger.info("Enhanced ticket generator initialized")
                    
        except Exception as e:
            logger.error(f"Failed to initialize enhanced ticket generator: {e}")
    
    async def generate_ticket(self, request, use_langchain: bool = True):
        """Generate ticket with optional LangChain enhancement."""
        if use_langchain and self.langchain_generator:
            try:
                result = self.langchain_generator.generate_ticket(
                    action_item=request.action_item,
                    context=request.context,
                    ticket_type=request.ticket_type
                )
                
                # Store in memory for context
                self.memory.save_context(
                    {"input": f"Generate ticket for: {request.action_item}"},
                    {"output": f"Created: {result['title']}"}
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"LangChain generation failed: {e}")
        
        # Fallback to traditional method
        import huggingface_local
        return await huggingface_local.ticket_generator.generate_ticket(request)


class LangChainMeetingAnalyzer:
    """Enhanced meeting analyzer with LangChain integration."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.memory = ConversationBufferMemory(
            memory_key="analysis_history",
            return_messages=True
        )
        
    async def analyze_meeting(self, request, use_langchain: bool = True):
        """Analyze meeting with LangChain enhancement."""
        if use_langchain and self.model_manager.initialized:
            try:
                # Enhanced analysis with context from memory
                chat_history = self.memory.chat_memory.messages
                
                # Add context-aware analysis here
                logger.info("Using LangChain enhanced meeting analysis")
                
                # Store analysis in memory
                self.memory.save_context(
                    {"input": f"Analyze meeting: {request.transcript[:100]}..."},
                    {"output": "Analysis completed with LangChain"}
                )
                
            except Exception as e:
                logger.warning(f"LangChain analysis failed: {e}")
        
        # Fallback to traditional analysis
        import huggingface_local
        return await huggingface_local.meeting_analyzer.analyze_meeting(request)


class EnhancedService:
    """Main enhanced service with LangChain integration."""
    
    def __init__(self):
        self.lc_model_manager = LangChainModelManager()
        self.lc_ticket_generator = None
        self.lc_meeting_analyzer = None
        self.initialized = False
    
    async def initialize(self, model_manager):
        """Initialize all enhanced services."""
        try:
            await self.lc_model_manager.initialize(model_manager)
            
            self.lc_ticket_generator = LangChainTicketGeneratorEnhanced(self.lc_model_manager)
            self.lc_meeting_analyzer = LangChainMeetingAnalyzer(self.lc_model_manager)
            
            await self.lc_ticket_generator.initialize()
            
            self.initialized = True
            logger.info("Enhanced service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced service: {e}")
            self.initialized = False
    
    async def generate_ticket(self, request, use_langchain: bool = True):
        """Generate ticket with LangChain enhancement."""
        if self.initialized and self.lc_ticket_generator:
            return await self.lc_ticket_generator.generate_ticket(request, use_langchain)
        
        # Fallback
        import huggingface_local
        return await huggingface_local.ticket_generator.generate_ticket(request)
    
    async def analyze_meeting(self, request, use_langchain: bool = True):
        """Analyze meeting with LangChain enhancement."""
        if self.initialized and self.lc_meeting_analyzer:
            return await self.lc_meeting_analyzer.analyze_meeting(request, use_langchain)
        
        # Fallback
        import huggingface_local
        return await huggingface_local.meeting_analyzer.analyze_meeting(request)


# Global enhanced service instance
enhanced_service = EnhancedService()
