#!/usr/bin/env python3
"""
Backlog Builder Local Hugging Face LLM Processing Service
Advanced text analysis using local transformer models
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
        nlp = None
        logger.warning("spaCy English model not found. Some NER features will be limited.")
except ImportError:
    spacy = None
    nlp = None
    logger.warning("spaCy not available. Some NLP features will be limited.")

# Download required NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"NLTK data download failed: {e}")

# Configuration
class Config:
    # Model configurations
    MODELS = {
        "summarization": {
            "primary": "facebook/bart-large-cnn",
            "backup": "sshleifer/distilbart-cnn-12-6",
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
        "generation": {
            "primary": "google/flan-t5-base",
            "backup": "t5-small"
        }
    }
    
    # Processing settings
    MAX_CHUNK_SIZE = 512
    OVERLAP_SIZE = 50
    CACHE_TTL = 3600
    GPU_ENABLED = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_ENABLED else "cpu"
    
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
    """Manages loading and caching of transformer models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_usage = {}
        self.last_cleanup = time.time()
        
    async def load_model(self, model_name: str, model_type: str = "auto"):
        """Load and cache a model"""
        cache_key = f"{model_type}:{model_name}"
        
        if cache_key in self.loaded_models:
            self.model_usage[cache_key] = time.time()
            return self.loaded_models[cache_key]
        
        logger.info(f"Loading model: {model_name} ({model_type})")
        
        try:
            if model_type == "summarization":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
                
            elif model_type == "classification":
                self.loaded_models[cache_key] = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if Config.GPU_ENABLED else -1
                )
                
            elif model_type == "ner":
                self.loaded_models[cache_key] = pipeline(
                    "ner",
                    model=model_name,
                    device=0 if Config.GPU_ENABLED else -1,
                    aggregation_strategy="simple"
                )
                
            elif model_type == "generation":
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
                
            else:
                # Auto mode
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                if Config.GPU_ENABLED:
                    model = model.to(Config.DEVICE)
                
                self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
            
            self.model_usage[cache_key] = time.time()
            logger.info(f"Model {model_name} loaded successfully")
            
            return self.loaded_models[cache_key]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    async def cleanup_unused_models(self, max_age_hours: float = 2):
        """Remove unused models from memory"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for cache_key, last_used in self.model_usage.items():
            if current_time - last_used > max_age_seconds:
                to_remove.append(cache_key)
        
        for cache_key in to_remove:
            logger.info(f"Cleaning up unused model: {cache_key}")
            if cache_key in self.loaded_models:
                del self.loaded_models[cache_key]
            del self.model_usage[cache_key]
            
        if to_remove:
            gc.collect()
            if Config.GPU_ENABLED:
                torch.cuda.empty_cache()

# Initialize model manager
model_manager = ModelManager()

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
        patterns = [
            r"(?:action item|todo|task|assign|responsible):?\s*(.+?)(?:\.|$|;)",
            r"(.+?)\s+(?:will|should|needs to|must)\s+(.+?)(?:\.|$|;)",
            r"(?:@|assign|assigned to)\s*(\w+)\s*[:-]?\s*(.+?)(?:\.|$|;)",
            r"(?:deadline|due|by)\s*([^.;]+?)(?:\.|$|;)",
        ]
        
        action_items = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action_items.append({
                    "text": match.group(0),
                    "groups": match.groups(),
                    "confidence": 0.6  # Pattern-based confidence
                })
        
        return action_items
    
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
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    
    async def _generate_summary(self, transcript: str) -> Summary:
        """Generate meeting summary using BART"""
        try:
            model_info = await model_manager.load_model(
                Config.MODELS["summarization"]["primary"],
                "summarization"
            )
            
            # Create summarization pipeline
            summarizer = pipeline(
                "summarization",
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                device=0 if Config.GPU_ENABLED else -1
            )
            
            # Handle long text by chunking
            chunks = text_processor.chunk_text(transcript, max_length=1024)
            chunk_summaries = []
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                summary = summarizer(
                    chunk,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )
                chunk_summaries.append(summary[0]['summary_text'])
            
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
            action_items = []
            
            # Use pattern-based extraction
            pattern_items = text_processor.detect_action_items_patterns(transcript)
            
            # Use NER for person detection
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
            
            for item in pattern_items:
                # Extract task and assignee
                task_text = item["text"]
                assignee = None
                
                # Simple assignee detection
                for person in people:
                    if person.lower() in task_text.lower():
                        assignee = person
                        break
                
                # Priority detection (simple heuristic)
                priority = "medium"
                if any(word in task_text.lower() for word in ["urgent", "asap", "critical", "immediately"]):
                    priority = "high"
                elif any(word in task_text.lower() for word in ["later", "when possible", "low priority"]):
                    priority = "low"
                
                action_items.append(ActionItem(
                    task=task_text,
                    assignee=assignee,
                    priority=priority,
                    confidence=item["confidence"]
                ))
            
            return action_items
            
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
    
    async def generate_ticket(self, request: TicketGenerationRequest) -> GeneratedTicket:
        """Generate a structured ticket from action item"""
        try:
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
            
            # Select appropriate template
            if request.ticket_type == "user_story":
                template = templates["templates"]["ticket_generation"]["user_story"]["agile"]
            elif request.ticket_type == "bug":
                template = templates["templates"]["ticket_generation"]["bug_report"]["standard"]
            else:
                template = templates["templates"]["ticket_generation"]["feature_request"]["detailed"]
            
            # Load generation model
            model_info = await model_manager.load_model(
                Config.MODELS["generation"]["primary"],
                "generation"
            )
            
            # Prepare prompt
            prompt = template["prompt"].replace("{{action_item}}", request.action_item)
            prompt = prompt.replace("{{context}}", request.context)
            
            # Generate ticket content
            generator = pipeline(
                "text2text-generation",
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                device=0 if Config.GPU_ENABLED else -1
            )
            
            result = generator(
                prompt,
                max_length=template.get("max_tokens", 500),
                temperature=template.get("temperature", 0.3),
                do_sample=True
            )
            
            generated_text = result[0]["generated_text"]
            
            # Parse generated ticket (simplified)
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
            
        except Exception as e:
            logger.error(f"Ticket generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Ticket generation failed: {e}")
    
    def _extract_title(self, text: str) -> str:
        """Extract title from generated text"""
        lines = text.split('\n')
        for line in lines:
            if 'title:' in line.lower():
                return line.split(':', 1)[1].strip()
        
        # Fallback: use first line
        return lines[0][:100] if lines else "Generated Ticket"
    
    def _extract_description(self, text: str) -> str:
        """Extract description from generated text"""
        # Simple extraction - in production, this would be more sophisticated
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
            "high": ["urgent", "critical", "asap", "immediately"],
            "low": ["later", "when possible", "low priority"],
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

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.loaded_models),
        "gpu_available": Config.GPU_ENABLED,
        "uptime": time.time() - start_time,
        "device": Config.DEVICE
    }

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