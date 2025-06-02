#!/usr/bin/env python3
import json
"""
Backlog Builder Local Hugging Face LLM Processing Service
Advanced text analysis using local transformer models
Optimized for RTX 4060 GPU (8GB VRAM)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
import gc
import re

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    pipeline, Pipeline, BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk

# Try to import optional dependencies
try:
    import redis
except ImportError:
    redis = None
    logger.warning("Redis not available. Caching will be disabled.")

try:
    import spacy
    # Try to load spacy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Some NER features will be limited.")
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Some NER features will be disabled.")
    nlp = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import LangChain ticket generator
try:
    from langchain_ticket_generator import LangChainTicketGenerator, create_langchain_generator
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain ticket generator available")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning(f"LangChain not available: {e}")

# Optional spaCy import
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. Some NER features will be limited.")

# Configuration class
class Config:
    """Configuration settings optimized for RTX 4060"""
    
    # Model configurations optimized for RTX 4060 (8GB VRAM)
    MODELS = {
        "summarization": {
            "primary": "facebook/bart-large-cnn",
            "backup": "sshleifer/distilbart-cnn-12-6",  # Smaller model for memory constraints
            "local_path": "/app/models/summarization"
        },
        "classification": {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "topic": "facebook/bart-large-mnli",
            "priority": "microsoft/DialoGPT-medium"
        },
        "extraction": {
            "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "keywords": "all-MiniLM-L6-v2"
        },
        "generation": { # General purpose, might be phased out for tickets
            "primary": "google/flan-t5-base",
            "backup": "t5-small"
        },
        "interpretation": {
            "primary": "google/flan-t5-small",
            "backup": "t5-small" # Fallback if flan-t5-small fails
        },
        "ticket_writing": {
            "primary": "t5-small",
            "backup": "google/flan-t5-small" # Fallback if t5-small fails
        },
        "ticket_generation": {
            "primary": "google/flan-t5-base",
            "backup": "t5-small"
        }
    }
    
    # Text processing settings
    MAX_CHUNK_SIZE = 512
    OVERLAP_SIZE = 50
    CACHE_TTL = 3600
    
    # GPU Configuration optimized for RTX 4060
    GPU_ENABLED = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_ENABLED else "cpu"
    
    # RTX 4060 specific optimizations
    if GPU_ENABLED:
        # Check GPU memory and set appropriate settings
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} with {gpu_memory:.1f}GB memory")
        
        # Enable mixed precision for RTX 4060
        USE_MIXED_PRECISION = True
        
        # Memory management settings
        MAX_MEMORY_USAGE = 0.85  # Use up to 85% of GPU memory
        ENABLE_MEMORY_EFFICIENT_ATTENTION = True
        
        # Batch size optimization for 8GB VRAM
        if gpu_memory >= 7.5:  # RTX 4060 8GB
            MAX_BATCH_SIZE = 8
            MODEL_MAX_LENGTH = 1024
        else:
            MAX_BATCH_SIZE = 4
            MODEL_MAX_LENGTH = 512
            
        # Enable gradient checkpointing to save memory
        USE_GRADIENT_CHECKPOINTING = True
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(MAX_MEMORY_USAGE)
        
        # Enable optimized attention
        torch.backends.cuda.enable_flash_sdp(True)
        
    else:
        USE_MIXED_PRECISION = False
        MAX_BATCH_SIZE = 2
        MODEL_MAX_LENGTH = 512
        ENABLE_MEMORY_EFFICIENT_ATTENTION = False
        USE_GRADIENT_CHECKPOINTING = False
    
    # Redis settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API settings
    PORT = int(os.getenv("LLM_PORT", 8002))

# Pydantic models
class MeetingAnalysisRequest(BaseModel):
    transcript: str
    meeting_type: Optional[str] = "general"
    language: str = "en"
    extract_action_items: bool = True
    extract_decisions: bool = True
    generate_summary: bool = True
    detect_sentiment: bool = False
    identify_speakers: bool = False

class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "medium"
    dependencies: List[str] = []
    confidence: float

class Decision(BaseModel):
    decision: str
    context: str
    stakeholders: List[str] = []
    impact: str = "medium"
    confidence: float

class Summary(BaseModel):
    executive_summary: str
    key_points: List[str]
    topics_discussed: List[str]
    confidence: float

class AnalysisResponse(BaseModel):
    processing_time: float
    summary: Optional[Summary] = None
    action_items: List[ActionItem] = []
    decisions: List[Decision] = []
    sentiment: Optional[Dict] = None
    metadata: Dict = {}

class TicketGenerationRequest(BaseModel):
    action_item: str
    context: str
    ticket_type: str = "task"  # task, bug, feature, epic
    project_info: Optional[Dict] = None

class GeneratedTicket(BaseModel):
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: str
    labels: List[str]
    estimated_effort: Optional[str] = None
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="Backlog Builder Local LLM Service",
    description="Advanced meeting analysis using local transformer models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models_cache = {}
pipelines_cache = {}
redis_client = None
nlp = None
embedding_model = None
start_time = time.time()

class ModelManager:
    """Manages loading and caching of transformer models with RTX 4060 optimizations"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_usage = {}
        self.last_cleanup = time.time()
        self.memory_monitor = GPUMemoryMonitor() if Config.GPU_ENABLED else None
        
    async def load_model(self, model_name: str, model_type: str = "auto"):
        """Load and cache a model with GPU optimizations"""
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self.loaded_models:
            self.model_usage[cache_key] = time.time()
            return self.loaded_models[cache_key]
        
        # Check GPU memory before loading
        if Config.GPU_ENABLED and self.memory_monitor:
            available_memory = self.memory_monitor.get_available_memory()
            if available_memory < 1.0:  # Less than 1GB available
                logger.warning(f"Low GPU memory ({available_memory:.1f}GB). Cleaning up models...")
                await self.cleanup_unused_models(max_age_hours=0.5)
        
        logger.info(f"Loading model: {model_name} ({model_type})")
        
        try:
            if model_type == "summarization":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                    device_map="auto" if Config.GPU_ENABLED else None,
                    low_cpu_mem_usage=True
                )
                
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                    if Config.USE_GRADIENT_CHECKPOINTING:
                        model.gradient_checkpointing_enable()
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
                
            elif model_type == "classification":
                self.loaded_models[cache_key] = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if Config.GPU_ENABLED else -1,
                    torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "device_map": "auto" if Config.GPU_ENABLED else None
                    }
                )
                
            elif model_type == "ner":
                self.loaded_models[cache_key] = pipeline(
                    "ner",
                    model=model_name,
                    device=0 if Config.GPU_ENABLED else -1,
                    aggregation_strategy="simple",
                    torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "device_map": "auto" if Config.GPU_ENABLED else None
                    }
                )
                
            elif model_type == "generation":
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                    device_map="auto" if Config.GPU_ENABLED else None,
                    low_cpu_mem_usage=True
                )
                
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                    if Config.USE_GRADIENT_CHECKPOINTING:
                        model.gradient_checkpointing_enable()
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
                
            else:
                # Auto mode
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                    device_map="auto" if Config.GPU_ENABLED else None,
                    low_cpu_mem_usage=True
                )
                
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
            
            self.model_usage[cache_key] = time.time()
            logger.info(f"Model {model_name} loaded successfully")
            
            # Log GPU memory usage after loading
            if Config.GPU_ENABLED and self.memory_monitor:
                memory_info = self.memory_monitor.get_memory_info()
                logger.info(f"GPU Memory after loading: {memory_info}")
            
            return self.loaded_models[cache_key]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            
            # Try backup model if available
            if model_type in Config.MODELS and "backup" in Config.MODELS[model_type]:
                backup_model = Config.MODELS[model_type]["backup"]
                logger.info(f"Trying backup model: {backup_model}")
                return await self.load_model(backup_model, model_type)
            
            raise HTTPException(status_code=500, detail={"error": f"Failed to load model: {e}"})
    
    async def cleanup_unused_models(self, max_age_hours: float = 2):
        """Remove unused models from memory with aggressive cleanup for GPU"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        models_to_remove = []
        for cache_key, last_used in self.model_usage.items():
            if current_time - last_used > max_age_seconds:
                models_to_remove.append(cache_key)
        
        for cache_key in models_to_remove:
            if cache_key in self.loaded_models:
                logger.info(f"Removing unused model: {cache_key}")
                
                # Properly cleanup model from GPU memory
                model_data = self.loaded_models[cache_key]
                if isinstance(model_data, dict) and "model" in model_data:
                    model = model_data["model"]
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                
                del self.loaded_models[cache_key]
                del self.model_usage[cache_key]
        
        # Force garbage collection and clear GPU cache
        gc.collect()
        if Config.GPU_ENABLED:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.last_cleanup = current_time
        logger.info(f"Model cleanup completed. Removed {len(models_to_remove)} models")


class GPUMemoryMonitor:
    """Monitor GPU memory usage for RTX 4060"""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.device = torch.cuda.current_device()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information in GB"""
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "total_gb": total,
            "available_gb": total - cached,
            "utilization_percent": (cached / total) * 100
        }
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        return self.get_memory_info()["available_gb"]
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if enough memory is available"""
        return self.get_available_memory() >= required_gb
    
    def clear_cache(self):
        """Clear GPU cache"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class TextProcessor:
    """Advanced text processing and analysis"""
    
    def __init__(self):
        self.tfidf = None
        self.keyword_extractor = None
    
    def chunk_text(self, text: str, max_length: int = Config.MAX_CHUNK_SIZE, overlap: int = Config.OVERLAP_SIZE) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = " ".join(words[i:i + max_length])
            chunks.append(chunk)
            
            if i + max_length >= len(words):
                break
        
        return chunks
    
    def extract_key_phrases(self, text: str, num_phrases: int = 5) -> List[str]:
        """Extract key phrases using TF-IDF"""
        if not self.tfidf:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words='english',
                max_features=1000
            )
        
        try:
            # Fit on single document
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            phrase_scores = list(zip(feature_names, tfidf_scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [phrase for phrase, score in phrase_scores[:num_phrases]]
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []
    
    def detect_action_items_patterns(self, text: str) -> List[Dict]:
        """Use regex patterns to detect action items"""
        # Improved patterns for natural meeting language
        patterns = [
            # Pattern: "Person needs to/should/will do something"
            r"(\w+)\s+(?:needs to|should|will|must)\s+([^.!?;]+)",
            # Pattern: "Someone is responsible for something"
            r"(\w+)\s+(?:is responsible for|will handle|will take care of|will coordinate)\s+([^.!?;]+)",
            # Pattern: "Action item: something" or "TODO: something"
            r"(?:action item|todo|task|assign|responsible):?\s*([^.!?;]+)",
            # Pattern: "We need to do something" or "Team should do something"
            r"(?:we|team|everyone)\s+(?:need to|should|must|will)\s+([^.!?;]+)",
            # Pattern: "Something needs to be done by someone"
            r"([^.!?;]+?)\s+(?:by|from)\s+(\w+)",
            # Pattern: "Create/Update/Fix/Implement something"
            r"(?:create|update|fix|implement|finalize|coordinate|document)\s+([^.!?;]+)",
        ]
        
        action_items = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Extract the full sentence as the task
                    task_text = sentence.strip()
                    
                    # Try to identify assignee from the match groups
                    assignee = None
                    groups = match.groups()
                    
                    # Check if first group looks like a person name (capitalized)
                    if groups and len(groups) > 0:
                        potential_assignee = groups[0].strip()
                        if potential_assignee and potential_assignee[0].isupper() and len(potential_assignee.split()) <= 2:
                            assignee = potential_assignee
                    
                    # Skip if task is too short or doesn't seem actionable
                    if len(task_text) < 15:
                        continue
                        
                    action_items.append({
                        "text": task_text,
                        "assignee": assignee,
                        "groups": groups,
                        "confidence": 0.7  # Higher confidence for improved patterns
                    })
                    break  # Only match once per sentence
        
        # Remove duplicates based on similar text
        unique_items = []
        for item in action_items:
            is_duplicate = False
            for existing in unique_items:
                # Simple similarity check
                if len(set(item["text"].lower().split()) & set(existing["text"].lower().split())) / max(len(item["text"].split()), len(existing["text"].split())) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_items.append(item)
        
        return unique_items
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        if not nlp:
            return {}
        
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities

# Initialize text processor
text_processor = TextProcessor()

class MeetingAnalyzer:
    """Main meeting analysis engine"""
    
    async def analyze_meeting(self, request: MeetingAnalysisRequest) -> AnalysisResponse:
        """Perform comprehensive meeting analysis"""
        start_time = time.time()
        
        try:
            results = AnalysisResponse(
                processing_time=0,
                metadata={
                    "transcript_length": len(request.transcript),
                    "language": request.language,
                    "meeting_type": request.meeting_type
                }
            )
            
            # Run analysis tasks
            tasks = []
            
            if request.generate_summary:
                tasks.append(self._generate_summary(request.transcript))
            
            if request.extract_action_items:
                tasks.append(self._extract_action_items(request.transcript))
            
            if request.extract_decisions:
                tasks.append(self._extract_decisions(request.transcript))
            
            if request.detect_sentiment:
                tasks.append(self._analyze_sentiment(request.transcript))
            
            # Execute tasks concurrently
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            task_index = 0
            if request.generate_summary:
                if not isinstance(task_results[task_index], Exception):
                    results.summary = task_results[task_index]
                task_index += 1
            
            if request.extract_action_items:
                if not isinstance(task_results[task_index], Exception):
                    results.action_items = task_results[task_index]
                task_index += 1
            
            if request.extract_decisions:
                if not isinstance(task_results[task_index], Exception):
                    results.decisions = task_results[task_index]
                task_index += 1
            
            if request.detect_sentiment:
                if not isinstance(task_results[task_index], Exception):
                    results.sentiment = task_results[task_index]
            
            results.processing_time = time.time() - start_time
            return results
            
        except Exception as e:
            logger.error(f"Meeting analysis failed: {e}")
            raise HTTPException(status_code=500, detail={"error": f"Analysis failed: {e}"})
    
    async def _generate_summary(self, transcript: str) -> Summary:
        """Generate meeting summary using BART with RTX 4060 optimizations"""
        try:
            model_info = await model_manager.load_model(
                Config.MODELS["summarization"]["primary"],
                "summarization"
            )
            
            # Create summarization pipeline with GPU optimizations
            summarizer = pipeline(
                "summarization",
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                device=0 if Config.GPU_ENABLED else -1,
                torch_dtype=torch.float16 if Config.USE_MIXED_PRECISION else torch.float32,
                batch_size=Config.MAX_BATCH_SIZE if Config.GPU_ENABLED else 1
            )
            
            # Handle long text by chunking with optimized chunk size
            chunks = text_processor.chunk_text(transcript, max_length=Config.MODEL_MAX_LENGTH)
            chunk_summaries = []
            
            # Process chunks in batches for GPU efficiency
            batch_size = Config.MAX_BATCH_SIZE if Config.GPU_ENABLED else 1
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                valid_chunks = [chunk for chunk in batch_chunks if len(chunk.strip()) >= 50]
                
                if not valid_chunks:
                    continue
                
                # Process batch
                with torch.cuda.amp.autocast(enabled=Config.USE_MIXED_PRECISION):
                    summaries = summarizer(
                        valid_chunks,
                        max_length=150,
                        min_length=30,
                        do_sample=False,
                        truncation=True
                    )
                
                # Extract summary texts
                if isinstance(summaries, list):
                    for summary in summaries:
                        if isinstance(summary, list):
                            chunk_summaries.extend([s['summary_text'] for s in summary])
                        else:
                            chunk_summaries.append(summary['summary_text'])
                else:
                    chunk_summaries.append(summaries['summary_text'])
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # Generate final summary if combined is too long
            if len(combined_summary) > 1024:
                final_summary = summarizer(
                    combined_summary,
                    max_length=300,
                    min_length=100,
                    do_sample=False
                )
                executive_summary = final_summary[0]['summary_text']
            else:
                executive_summary = combined_summary
            
            # Extract key points and topics
            key_phrases = text_processor.extract_key_phrases(transcript, num_phrases=8)
            
            return Summary(
                executive_summary=executive_summary,
                key_points=chunk_summaries[:5],  # Top 5 key points
                topics_discussed=key_phrases,
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return Summary(
                executive_summary="Summary generation failed",
                key_points=[],
                topics_discussed=[],
                confidence=0.0
            )
    
    async def _extract_action_items(self, transcript: str) -> List[ActionItem]:
        """Extract action items using patterns and NLP"""
        try:
            # Use improved pattern-based extraction
            pattern_items = text_processor.detect_action_items_patterns(transcript)
            
            # Use NER for additional person detection if available
            entities = text_processor.extract_named_entities(transcript)
            people = entities.get('PERSON', [])
            
            # Load classification model for priority detection
            try:
                classifier = await model_manager.load_model(
                    Config.MODELS["classification"]["sentiment"],
                    "classification"
                )
            except:
                classifier = None
            
            result_items = []
            
            for item in pattern_items:
                # Extract task and assignee
                task_text = item["text"]
                assignee = item.get("assignee")
                
                # If no assignee found in pattern, try NER detection
                if not assignee and people:
                    for person in people:
                        if person.lower() in task_text.lower():
                            assignee = person
                            break
                
                # Priority detection (improved heuristics)
                priority = "medium"
                task_lower = task_text.lower()
                
                if any(word in task_lower for word in ["urgent", "asap", "critical", "immediately", "emergency", "high priority"]):
                    priority = "high"
                elif any(word in task_lower for word in ["later", "when possible", "low priority", "nice to have", "eventually"]):
                    priority = "low"
                elif any(word in task_lower for word in ["friday", "deadline", "due", "by end of", "before"]):
                    priority = "high"
                
                # Extract potential deadline information
                deadline = None
                deadline_patterns = [
                    r"by\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday)",
                    r"due\s+([^.!?;]+)",
                    r"deadline\s+([^.!?;]+)",
                    r"by\s+end\s+of\s+([^.!?;]+)",
                    r"before\s+([^.!?;]+)"
                ]
                
                for pattern in deadline_patterns:
                    match = re.search(pattern, task_lower)
                    if match:
                        deadline = match.group(1).strip()
                        break
                
                result_items.append(ActionItem(
                    task=task_text,
                    assignee=assignee,
                    deadline=deadline,
                    priority=priority,
                    confidence=item["confidence"]
                ))
            
            return result_items
            
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            return []
    
    async def _extract_decisions(self, transcript: str) -> List[Decision]:
        """Extract decisions from meeting transcript"""
        try:
            decisions = []
            
            # Decision patterns
            decision_patterns = [
                r"(?:decided|decision|agreed|resolved|concluded):?\s*(.+?)(?:\.|$|;)",
                r"(?:we will|team will|going to)\s+(.+?)(?:\.|$|;)",
                r"(?:final decision|conclusion):?\s*(.+?)(?:\.|$|;)",
            ]
            
            entities = text_processor.extract_named_entities(transcript)
            people = entities.get('PERSON', [])
            
            for pattern in decision_patterns:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    decision_text = match.group(1).strip()
                    
                    # Find context (surrounding sentences)
                    sentences = transcript.split('.')
                    context = ""
                    for i, sentence in enumerate(sentences):
                        if decision_text in sentence:
                            # Get context from surrounding sentences
                            start = max(0, i-1)
                            end = min(len(sentences), i+2)
                            context = ". ".join(sentences[start:end])
                            break
                    
                    decisions.append(Decision(
                        decision=decision_text,
                        context=context,
                        stakeholders=people[:3],  # Assume first 3 people are stakeholders
                        confidence=0.7
                    ))
            
            return decisions
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {e}")
            return []
    
    async def _analyze_sentiment(self, transcript: str) -> Dict:
        """Analyze overall sentiment of the meeting"""
        try:
            # Use TextBlob for quick sentiment analysis
            blob = TextBlob(transcript)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment_label = "positive"
            elif polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "overall_sentiment": sentiment_label,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "confidence": abs(polarity)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.0}

# Initialize analyzer
meeting_analyzer = MeetingAnalyzer()

class TicketGenerator:
    """Generate structured tickets from action items"""
    
    def __init__(self):
        """Initialize ticket generator with LangChain support."""
        self.langchain_generator = None
        self.use_langchain = LANGCHAIN_AVAILABLE and os.getenv("USE_LANGCHAIN", "true").lower() == "true"
        logger.info(f"TicketGenerator initialized - LangChain enabled: {self.use_langchain}")
    
    async def generate_ticket(self, request: TicketGenerationRequest) -> GeneratedTicket:
        """Generate a structured ticket from action item"""
        try:
            # Try LangChain approach first if available
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                try:
                    return await self._generate_with_langchain(request)
                except Exception as e:
                    logger.warning(f"LangChain generation failed, falling back to traditional approach: {e}")
            
            # Fallback to traditional approach
            return await self._generate_traditional(request)
            
        except Exception as e:
            logger.error(f"Ticket generation failed: {e}")
            # Return a basic fallback ticket
            return GeneratedTicket(
                title=self._extract_simple_title(request.action_item),
                description=f"As a User, I want to {request.action_item.lower()} so that {request.context[:100]}...",
                acceptance_criteria=["Implementation complete", "Tests passing"],
                priority="medium",
                labels=[request.ticket_type],
                estimated_effort=None,
                confidence=0.3
            )
    
    async def _generate_with_langchain(self, request: TicketGenerationRequest) -> GeneratedTicket:
        """Generate ticket using LangChain approach."""
        logger.info("Generating ticket with LangChain...")
        
        # Initialize LangChain generator if not already done
        if not self.langchain_generator:
            # Load a model for LangChain
            model_info = await model_manager.load_model(
                Config.MODELS["ticket_generation"]["primary"],
                "generation"
            )
            
            # Create pipeline for LangChain
            model_pipeline = pipeline(
                "text2text-generation",
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                device=0 if Config.GPU_ENABLED else -1,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
            
            self.langchain_generator = create_langchain_generator(model_pipeline)
        
        # Generate ticket using LangChain
        result = self.langchain_generator.generate_ticket(
            action_item=request.action_item,
            context=request.context,
            ticket_type=request.ticket_type
        )
        
        # Convert to GeneratedTicket format
        return GeneratedTicket(
            title=result["title"],
            description=result["description"],
            acceptance_criteria=result["acceptance_criteria"],
            priority=result["priority"],
            labels=result["labels"],
            estimated_effort=result["estimated_effort"],
            confidence=result["confidence"]
        )
    
    async def _generate_traditional(self, request: TicketGenerationRequest) -> GeneratedTicket:
        """Generate ticket using traditional template-based approach."""
        logger.info("Generating ticket with traditional approach...")
        
        # Load prompt templates - try multiple paths for different environments
        template_paths = [
            "prompt-templates.json",  # Local development
            "/app/llm-processing/prompt-templates.json",  # Docker container
            "./prompt-templates.json",  # Current directory
            os.path.join(os.path.dirname(__file__), "prompt-templates.json")  # Same directory as script
        ]
        
        templates = None
        for path in template_paths:
            try:
                with open(path, "r") as f:
                    templates = json.load(f)
                    break
            except FileNotFoundError:
                continue
        
        if templates is None:
            raise FileNotFoundError("prompt-templates.json not found in any expected location")
        
        # Get templates from the correct structure
        interpretation_template = templates["templates"]["task_interpretation"]["default"]
        user_story_template = templates["templates"]["ticket_generation"]["user_story"]["agile"]
        
        # STAGE 1: Interpretation LLM Call
        logger.info("Loading interpretation model...")
        interpretation_model_info = await model_manager.load_model(
            Config.MODELS["interpretation"]["primary"],
            "generation"  # Use generation type for T5 models
        )
        
        interpretation_generator = pipeline(
            "text2text-generation",
            model=interpretation_model_info["model"],
            tokenizer=interpretation_model_info["tokenizer"],
            device=0 if Config.GPU_ENABLED else -1
        )
        logger.info(f"Using interpretation model: {Config.MODELS['interpretation']['primary']}")

        # Build interpretation prompt from template
        interpretation_prompt = interpretation_template["prompt"]
        interpretation_prompt = interpretation_prompt.replace("{{action_item}}", request.action_item)
        interpretation_prompt = interpretation_prompt.replace("{{context}}", request.context)
        
        logger.debug(f"Interpretation Prompt: {interpretation_prompt[:500]}...")

        interpretation_result = interpretation_generator(
            interpretation_prompt,
            max_length=interpretation_template.get("max_tokens", 150),
            temperature=interpretation_template.get("temperature", 0.2),
            do_sample=True
        )
        raw_interpreted_text = interpretation_result[0]["generated_text"].strip()
        logger.info(f"Raw Interpreted Text from LLM: {raw_interpreted_text}")

        interpreted_data = None
        try:
            # The LLM might sometimes wrap the JSON in backticks or other text
            # Attempt to extract JSON block if present
            if raw_interpreted_text.startswith("```json") and raw_interpreted_text.endswith("```"):
                json_str = raw_interpreted_text[len("```json"):-(len("```"))].strip()
            elif raw_interpreted_text.startswith("```") and raw_interpreted_text.endswith("```"):
                json_str = raw_interpreted_text[len("```"):-(len("```"))].strip()
            else:
                json_str = raw_interpreted_text
            
            interpreted_data = json.loads(json_str)
            logger.info(f"Parsed Interpreted JSON: {interpreted_data}")

            # Validate required keys
            required_keys = ["problem", "desired_outcome", "user_role"]
            if not all(key in interpreted_data for key in required_keys):
                logger.error(f"Interpretation JSON is missing one or more required keys: {required_keys}. Data: {interpreted_data}")
                # Fallback to using the raw action item if parsing fails or keys are missing
                interpreted_data = {
                    "problem": request.action_item, 
                    "desired_outcome": "Not specified", 
                    "user_role": "User"
                }
                logger.warning(f"Falling back to default interpretation due to parsing/validation error.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse interpretation JSON: {e}. Raw text: {raw_interpreted_text}")
            # Fallback strategy: use the original action item if JSON parsing fails
            interpreted_data = {
                "problem": request.action_item, 
                "desired_outcome": "Not specified", 
                "user_role": "User"
            }
            logger.warning(f"Falling back to default interpretation due to JSONDecodeError.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during interpretation processing: {e}")
            interpreted_data = {
                "problem": request.action_item, 
                "desired_outcome": "Not specified", 
                "user_role": "User"
            }
            logger.warning(f"Falling back to default interpretation due to unexpected error.")

        # STAGE 2: Ticket Creation LLM Call (using the structured interpreted data)
        logger.info("Loading ticket writing model...")
        ticket_model_info = await model_manager.load_model(
            Config.MODELS["ticket_writing"]["primary"],
            "generation"  # Use generation type for T5 models
        )
        ticket_generator_pipeline = pipeline(
            "text2text-generation",
            model=ticket_model_info["model"],
            tokenizer=ticket_model_info["tokenizer"],
            device=0 if Config.GPU_ENABLED else -1
        )
        logger.info(f"Using ticket writing model: {Config.MODELS['ticket_writing']['primary']}")

        # Build ticket prompt from template
        ticket_prompt = user_story_template["prompt"]
        ticket_prompt = ticket_prompt.replace("{{interpreted_problem}}", str(interpreted_data.get("problem", request.action_item)))
        ticket_prompt = ticket_prompt.replace("{{interpreted_outcome}}", str(interpreted_data.get("desired_outcome", "Achieve task goal")))
        ticket_prompt = ticket_prompt.replace("{{user_role}}", str(interpreted_data.get("user_role", "User")))
        # Note: The template doesn't have {{context}} placeholder, so we skip this replacement

        logger.debug(f"Populated Ticket Prompt for Stage 2: {ticket_prompt[:500]}...") # Log start of prompt 

        ticket_generation_result = ticket_generator_pipeline(
            ticket_prompt,
            max_length=user_story_template.get("max_tokens", 500),
            temperature=user_story_template.get("temperature", 0.3),
            do_sample=True
        )
        generated_text = ticket_generation_result[0]["generated_text"]
        
        # Parse generated ticket
        title = self._extract_title(generated_text)
        description = self._extract_description(generated_text)
        acceptance_criteria = self._extract_acceptance_criteria(generated_text)
        priority = self._extract_priority(generated_text)
        labels = self._extract_labels(request.action_item)
        
        return GeneratedTicket(
            title=title,
            description=description,
            acceptance_criteria=acceptance_criteria,
            priority=priority,
            labels=labels,
            confidence=0.8
        )
    
    def _extract_simple_title(self, action_item: str) -> str:
        """Extract a simple title from action item for fallback scenarios."""
        # Remove common prefixes
        title = action_item.replace("We need to", "").replace("we should", "").replace("The team needs to", "").strip()
        
        # Ensure it starts with an action verb
        if not any(title.lower().startswith(verb.lower()) for verb in ['implement', 'create', 'fix', 'update', 'add', 'remove', 'configure', 'setup', 'build', 'deploy']):
            if 'bug' in title.lower() or 'error' in title.lower():
                title = f"Fix {title}"
            elif 'new' in title.lower():
                title = f"Implement {title}"
            else:
                title = f"Update {title}"
        
        # Limit to 40 characters
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
    
    def _extract_title(self, text: str) -> str:
        """Extract a concise, action-oriented ticket title from generated text."""
        import re
        
        # 1. Try to find '**Title:**' or 'title:' using regex (case-insensitive)
        match = re.search(r'\*\*Title:\*\*\s*(.+)', text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Remove bracketed instruction text
            title = re.sub(r'\[.*?\]', '', title).strip()
            # Remove common instruction phrases
            title = re.sub(r'(Write a|Create a|Generate a|Make a)\s+', '', title, flags=re.IGNORECASE).strip()
            if title and not title.lower().startswith(('short', 'actionable', 'concise')):
                return self._shorten_title(title)
                
        for line in text.split('\n'):
            if 'title:' in line.lower():
                title = line.split(':', 1)[1].strip()
                # Remove bracketed instruction text
                title = re.sub(r'\[.*?\]', '', title).strip()
                if title and not title.lower().startswith(('short', 'actionable', 'concise')):
                    return self._shorten_title(title)
        
        # 2. Extract action items and convert to concise titles
        action_words = ['implement', 'create', 'update', 'fix', 'add', 'remove', 'configure', 'setup', 'build', 'deploy']
        
        # Look for action-oriented phrases
        sentences = re.split(r'[.!?]\s+', text.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if any(action in sentence.lower() for action in action_words):
                return self._shorten_title(sentence)
        
        # 3. Fallback: use first meaningful phrase
        fallback = text.strip().split('\n')[0] if text.strip() else "Task"
        
        # Remove common prefixes
        prefixes_to_remove = [
            'we need to', 'we should', 'it is necessary to', 'the team needs to',
            'action item:', 'todo:', 'task:', 'requirement:'
        ]
        
        fallback_lower = fallback.lower()
        for prefix in prefixes_to_remove:
            if fallback_lower.startswith(prefix):
                fallback = fallback[len(prefix):].strip()
                break
        
        return self._shorten_title(fallback)
    
    def _shorten_title(self, title: str) -> str:
        """Shorten title to max 40 characters while preserving meaning"""
        title = title.strip()
        
        # Remove personal pronouns and make more direct
        pronoun_replacements = {
            'we need to': '',
            'you need to': '',
            'i need to': '',
            'we need': '',
            'you need': '',
            'i need': '',
            'we should': '',
            'you should': '',
            'i should': '',
            'we must': '',
            'you must': '',
            'i must': '',
            'we can': '',
            'you can': '',
            'i can': '',
            'we will': '',
            'you will': '',
            'i will': '',
            'we have to': '',
            'you have to': '',
            'i have to': '',
            'we want to': '',
            'you want to': '',
            'i want to': '',
            'let us': '',
            'let\'s': '',
            'we are': '',
            'you are': '',
            'i am': '',
            'we\'re': '',
            'you\'re': '',
            'i\'m': ''
        }
        
        title_lower = title.lower()
        for phrase, replacement in pronoun_replacements.items():
            if title_lower.startswith(phrase):
                title = title[len(phrase):].strip()
                break
        
        # Additional cleanup for remaining pronouns at start
        words = title.split()
        if words and words[0].lower() in ['we', 'you', 'i', 'us', 'me']:
            words = words[1:]
            title = ' '.join(words)
        
        # Remove unnecessary words
        unnecessary_words = ['the', 'a', 'an', 'that', 'which', 'this', 'these', 'those', 'please', 'from', 'by', 'with', 'for']
        words = title.split()
        
        # Convert to imperative form if needed
        if words:
            first_word = words[0].lower()
            
            # Convert common patterns to imperative
            imperative_conversions = {
                'creating': 'create',
                'implementing': 'implement',
                'updating': 'update',
                'fixing': 'fix',
                'adding': 'add',
                'removing': 'remove',
                'configuring': 'configure',
                'setting': 'setup',
                'building': 'build',
                'deploying': 'deploy',
                'testing': 'test',
                'reviewing': 'review',
                'getting': 'get',
                'making': 'make',
                'ensuring': 'ensure',
                'checking': 'check'
            }
            
            if first_word in imperative_conversions:
                words[0] = imperative_conversions[first_word].capitalize()
            elif first_word in ['implement', 'create', 'update', 'fix', 'add', 'remove', 'configure', 'setup', 'build', 'deploy', 'test', 'review', 'get', 'make', 'ensure', 'check']:
                words[0] = words[0].capitalize()
        
        # Keep action words at the beginning, filter unnecessary words from the rest
        if words and words[0].lower() in ['implement', 'create', 'update', 'fix', 'add', 'remove', 'configure', 'setup', 'build', 'deploy', 'test', 'review', 'get', 'make', 'ensure', 'check']:
            # Keep first word, filter unnecessary words from the rest
            filtered_words = [words[0]] + [w for w in words[1:] if w.lower() not in unnecessary_words]
        else:
            filtered_words = [w for w in words if w.lower() not in unnecessary_words]
        
        shortened = ' '.join(filtered_words)
        
        # Final cleanup - capitalize first word if it's an action
        if shortened:
            first_word = shortened.split()[0].lower()
            if first_word in ['feedback', 'designs', 'tasks', 'testing', 'documentation']:
                # Convert nouns to verbs where possible
                noun_to_verb = {
                    'feedback': 'Get feedback',
                    'designs': 'Review designs', 
                    'tasks': 'Complete tasks',
                    'testing': 'Perform testing',
                    'documentation': 'Create documentation'
                }
                if first_word in noun_to_verb:
                    shortened = noun_to_verb[first_word] + shortened[len(first_word):]
        
        # If still too long, truncate at word boundary
        if len(shortened) > 40:
            cutoff = shortened[:37].rsplit(' ', 1)[0]
            return cutoff + '...'
        
        return shortened if shortened else "Task"
    
    def _extract_description(self, text: str) -> str:
        """Extract description from generated text with better user story parsing"""
        import re
        
        # Look for description section
        desc_match = re.search(r'\*\*Description:\*\*\s*(.*?)(?:\*\*|$)', text, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
            return description[:500] + "..." if len(description) > 500 else description
        
        # Fallback: look for user story pattern
        user_story_match = re.search(r'As a .+?, I want .+? so that .+?\.', text, re.IGNORECASE)
        if user_story_match:
            return user_story_match.group(0)
        
        # Final fallback
        return text[:500] + "..." if len(text) > 500 else text
    
    def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria from generated text"""
        criteria = []
        lines = text.split('\n')
        in_criteria_section = False
        
        for line in lines:
            if 'acceptance criteria' in line.lower():
                in_criteria_section = True
                continue
            
            if in_criteria_section and line.strip().startswith('-'):
                criteria.append(line.strip()[1:].strip())
        
        return criteria if criteria else ["Implementation complete", "Tests passing"]
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority from generated text"""
        priority_keywords = {
            "high": ["urgent", "critical", "asap", "immediately", "emergency", "high priority"],
            "low": ["later", "when possible", "low priority", "nice to have", "eventually"]
        }
        
        text_lower = text.lower()
        for priority, keywords in priority_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority
        
        return "medium"
    
    def _extract_labels(self, action_item: str) -> List[str]:
        """Generate relevant labels"""
        labels = []
        
        # Simple keyword-based labeling
        if any(word in action_item.lower() for word in ["bug", "fix", "error", "issue"]):
            labels.append("bug")
        if any(word in action_item.lower() for word in ["feature", "enhance", "improve"]):
            labels.append("enhancement")
        if any(word in action_item.lower() for word in ["test", "testing"]):
            labels.append("testing")
        if any(word in action_item.lower() for word in ["doc", "documentation"]):
            labels.append("documentation")
        
        return labels if labels else ["task"]

# Initialize ticket generator
ticket_generator = TicketGenerator()

# Initialize services
async def initialize_services():
    """Initialize Redis, models, and other services"""
    global redis_client, nlp, embedding_model
    
    try:
        # Initialize Redis if available
        if redis:
            redis_client = redis.from_url(Config.REDIS_URL)
            redis_client.ping()
            logger.info("Redis connection established")
        else:
            logger.info("Redis not available, proceeding without caching")
            redis_client = None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load spaCy model if available
    if not nlp and spacy:
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found. NER features will be limited.")
            nlp = None
    elif nlp:
        logger.info("spaCy model already loaded")
    else:
        logger.warning("spaCy not available. NER features will be limited.")
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer(Config.MODELS["extraction"]["keywords"])
        logger.info("Sentence transformer model loaded")
    except Exception as e:
        logger.warning(f"Embedding model failed to load: {e}")

# Initialize model manager
model_manager = ModelManager()

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint with RTX 4060 GPU information"""
    health_info = {
        "status": "healthy",
        "models_loaded": len(model_manager.loaded_models),
        "gpu_available": Config.GPU_ENABLED,
        "uptime": time.time() - start_time,
        "device": Config.DEVICE
    }
    
    # Add GPU-specific information for RTX 4060
    if Config.GPU_ENABLED and model_manager.memory_monitor:
        try:
            memory_info = model_manager.memory_monitor.get_memory_info()
            health_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": memory_info,
                "mixed_precision_enabled": Config.USE_MIXED_PRECISION,
                "max_batch_size": Config.MAX_BATCH_SIZE,
                "model_max_length": Config.MODEL_MAX_LENGTH
            })
        except Exception as e:
            health_info["gpu_error"] = str(e)
    
    return health_info

@app.get("/gpu/memory")
async def gpu_memory_status():
    """Get detailed GPU memory information for RTX 4060"""
    if not Config.GPU_ENABLED:
        raise HTTPException(status_code=404, detail="GPU not available")
    
    if not model_manager.memory_monitor:
        raise HTTPException(status_code=500, detail="GPU memory monitor not initialized")
    
    try:
        memory_info = model_manager.memory_monitor.get_memory_info()
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_info": memory_info,
            "recommendations": {
                "memory_usage_ok": memory_info["utilization_percent"] < 90,
                "cleanup_needed": memory_info["available_gb"] < 1.0,
                "can_load_large_model": memory_info["available_gb"] > 2.0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPU memory info: {str(e)}")

@app.post("/gpu/cleanup")
async def cleanup_gpu_memory():
    """Manually trigger GPU memory cleanup"""
    if not Config.GPU_ENABLED:
        raise HTTPException(status_code=404, detail="GPU not available")
    
    try:
        # Get memory before cleanup
        before_memory = model_manager.memory_monitor.get_memory_info() if model_manager.memory_monitor else None
        
        # Cleanup models
        await model_manager.cleanup_unused_models(max_age_hours=0.1)  # Aggressive cleanup
        
        # Clear GPU cache
        if model_manager.memory_monitor:
            model_manager.memory_monitor.clear_cache()
        
        # Get memory after cleanup
        after_memory = model_manager.memory_monitor.get_memory_info() if model_manager.memory_monitor else None
        
        return {
            "status": "cleanup_completed",
            "memory_before": before_memory,
            "memory_after": after_memory,
            "memory_freed_gb": (before_memory["cached_gb"] - after_memory["cached_gb"]) if (before_memory and after_memory) else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPU cleanup failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_meeting(request: MeetingAnalysisRequest):
    """Analyze meeting transcript"""
    return await meeting_analyzer.analyze_meeting(request)

@app.post("/generate-ticket", response_model=GeneratedTicket)
async def generate_ticket(request: TicketGenerationRequest):
    """Generate ticket from action item"""
    return await ticket_generator.generate_ticket(request)

@app.get("/models/status")
async def model_status():
    """Get status of loaded models"""
    return {
        "loaded_models": list(model_manager.loaded_models.keys()),
        "model_usage": model_manager.model_usage,
        "cache_size": len(model_manager.loaded_models)
    }

@app.post("/models/cleanup")
async def cleanup_models():
    """Manually trigger model cleanup"""
    await model_manager.cleanup_unused_models(max_age_hours=0.5)
    return {"status": "cleanup_completed"}

# Enhanced LangChain endpoints
@app.post("/analyze-enhanced", response_model=AnalysisResponse)
async def analyze_meeting_enhanced(
    request: MeetingAnalysisRequest,
    use_langchain: bool = True
):
    """Analyze meeting transcript with LangChain enhancement"""
    try:
        from enhanced_service import enhanced_service
        return await enhanced_service.analyze_meeting(request, use_langchain=use_langchain)
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        # Fallback to traditional analysis
        return await analyze_meeting(request)

@app.post("/analyze-conversation", response_model=Dict[str, Any])
async def analyze_conversation_intelligent(
    request: MeetingAnalysisRequest
):
    """Intelligently analyze conversation with deep context understanding"""
    try:
        logger.info("Starting intelligent conversation analysis...")
        
        # Load the conversation analysis model
        model_info = await model_manager.load_model("google/flan-t5-base", "generation")
        
        # Create conversation analyzer
        from conversation_analyzer import create_conversation_analyzer
        analyzer = create_conversation_analyzer(model_info)
        
        # Perform deep conversation analysis
        conversation_analysis = await analyzer.analyze_conversation(request.transcript)
        
        # Generate intelligent tickets based on conversation context
        intelligent_tickets = await analyzer.generate_intelligent_tickets(conversation_analysis)
        
        # Convert to response format
        tickets_data = []
        for ticket in intelligent_tickets:
            tickets_data.append({
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
            })
        
        return {
            "success": True,
            "conversation_analysis": {
                "project_context": conversation_analysis.project_context,
                "team_dynamics": conversation_analysis.team_dynamics,
                "key_decisions": conversation_analysis.key_decisions,
                "blockers_and_risks": conversation_analysis.blockers_and_risks,
                "timeline_constraints": conversation_analysis.timeline_constraints,
                "technical_requirements": conversation_analysis.technical_requirements,
                "business_requirements": conversation_analysis.business_requirements
            },
            "intelligent_tickets": tickets_data,
            "insights_analyzed": len(conversation_analysis.conversation_flow),
            "actionable_insights": len([i for i in conversation_analysis.conversation_flow if i.actionable]),
            "processing_metadata": {
                "analysis_method": "intelligent_conversation_analysis",
                "model_used": "google/flan-t5-base",
                "langchain_enhanced": True
            }
        }
        
    except Exception as e:
        logger.error(f"Intelligent conversation analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_suggestion": "Use /analyze-enhanced endpoint for traditional analysis"
        }

@app.post("/analyze-conversation-smart", response_model=Dict[str, Any])
async def analyze_conversation_smart(
    request: MeetingAnalysisRequest
):
    """Smart conversation analysis with pattern-based intelligence"""
    try:
        logger.info("Starting smart conversation analysis...")
        
        # Create simple conversation analyzer (no model required)
        from simple_conversation_analyzer import create_simple_conversation_analyzer
        analyzer = create_simple_conversation_analyzer()
        
        # Perform smart conversation analysis
        result = analyzer.analyze_conversation(request.transcript)
        
        logger.info(f"Smart analysis completed: {result['insights_analyzed']} insights, {result['actionable_insights']} actionable")
        
        return result
        
    except Exception as e:
        logger.error(f"Smart conversation analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_suggestion": "Use /analyze-enhanced endpoint for traditional analysis"
        }

@app.post("/generate-ticket-enhanced", response_model=GeneratedTicket)
async def generate_ticket_enhanced(
    request: TicketGenerationRequest,
    use_langchain: bool = True
):
    """Generate ticket from action item with LangChain enhancement"""
    try:
        from enhanced_service import enhanced_service
        return await enhanced_service.generate_ticket(request, use_langchain=use_langchain)
    except Exception as e:
        logger.error(f"Enhanced ticket generation failed: {e}")
        # Fallback to traditional generation
        return await generate_ticket(request)

@app.get("/langchain/status")
async def langchain_status():
    """Get LangChain integration status"""
    try:
        from enhanced_service import enhanced_service
        
        if not enhanced_service.initialized:
            return {
                "langchain_models_loaded": 0,
                "chains_available": 0,
                "memory_enabled": False,
                "integration_status": "not_initialized"
            }
        
        return {
            "langchain_models_loaded": len(enhanced_service.lc_model_manager.langchain_models),
            "chains_available": len(getattr(enhanced_service.lc_ticket_generator, 'chains', {})),
            "memory_enabled": enhanced_service.lc_ticket_generator.memory is not None,
            "integration_status": "active"
        }
    except Exception as e:
        logger.error(f"LangChain status check failed: {e}")
        return {
            "langchain_models_loaded": 0,
            "chains_available": 0,
            "memory_enabled": False,
            "integration_status": "error",
            "error": str(e)
        }

@app.post("/langchain/clear-memory")
async def clear_langchain_memory():
    """Clear LangChain conversation memory"""
    try:
        from enhanced_service import enhanced_service
        
        if enhanced_service.initialized:
            if enhanced_service.lc_ticket_generator and enhanced_service.lc_ticket_generator.memory:
                enhanced_service.lc_ticket_generator.memory.clear()
            if enhanced_service.lc_meeting_analyzer and enhanced_service.lc_meeting_analyzer.memory:
                enhanced_service.lc_meeting_analyzer.memory.clear()
            
            return {"status": "memory_cleared", "success": True}
        else:
            return {"status": "service_not_initialized", "success": False}
            
    except Exception as e:
        logger.error(f"Memory clear failed: {e}")
        return {"status": "error", "success": False, "error": str(e)}

@app.get("/health-enhanced")
async def health_check_enhanced():
    """Enhanced health check with LangChain status"""
    try:
        # Get base health check
        base_health = await health_check()
        
        # Add LangChain health information
        try:
            from enhanced_service import enhanced_service
            
            langchain_health = {
                "langchain_available": True,
                "langchain_models": len(enhanced_service.lc_model_manager.langchain_models) if enhanced_service.initialized else 0,
                "chains_initialized": enhanced_service.lc_ticket_generator is not None if enhanced_service.initialized else False,
                "memory_active": (enhanced_service.lc_ticket_generator.memory is not None) if enhanced_service.initialized and enhanced_service.lc_ticket_generator else False,
                "enhanced_service_status": "active" if enhanced_service.initialized else "inactive"
            }
        except Exception as e:
            langchain_health = {
                "langchain_available": False,
                "langchain_models": 0,
                "chains_initialized": False,
                "memory_active": False,
                "enhanced_service_status": "error",
                "error": str(e)
            }
        
        base_health.update(langchain_health)
        return base_health
        
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        return {
            "status": "error",
            "langchain_available": False,
            "error": str(e)
        }

# Background tasks
async def periodic_cleanup():
    """Periodic cleanup of unused models"""
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        try:
            await model_manager.cleanup_unused_models()
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Backlog Builder Local LLM Service")
    await initialize_services()
    
    # Start background tasks
    asyncio.create_task(periodic_cleanup())
    
    logger.info("Service initialization complete")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "huggingface-local:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=False,
        workers=1,
        log_level="info"
    )