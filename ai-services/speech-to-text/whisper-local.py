#!/usr/bin/env python3
"""
Backlog Builder Local Whisper Speech-to-Text Service
FastAPI service for offline speech recognition using OpenAI Whisper
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import uuid

import whisper
import torch
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger
import redis
from transformers import pipeline
import soundfile as sf
from pyannote.audio import Pipeline as DiarizationPipeline

# Configuration
class Config:
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
    DEFAULT_MODEL = "base"
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    CACHE_TTL = 3600  # 1 hour
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    GPU_ENABLED = torch.cuda.is_available()
    
    # Model selection based on audio length (seconds)
    MODEL_SELECTION_THRESHOLDS = {
        300: "tiny",      # < 5 minutes
        1800: "base",     # < 30 minutes  
        3600: "small",    # < 1 hour
        7200: "medium",   # < 2 hours
        float('inf'): "large"  # >= 2 hours
    }

# Pydantic models
class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    task: str = "transcribe"  # transcribe or translate
    enable_diarization: bool = False
    model_size: Optional[str] = None
    return_timestamps: bool = True

class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    processing_time: float
    segments: List[Dict]
    speakers: Optional[List[Dict]] = None
    model_used: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    gpu_available: bool
    uptime: float

# Initialize FastAPI app
app = FastAPI(
    title="Backlog Builder Whisper STT Service",
    description="Local speech-to-text processing using OpenAI Whisper",
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
redis_client = None
diarization_pipeline = None
start_time = time.time()
active_connections: List[WebSocket] = []

# Initialize components
async def initialize_services():
    """Initialize Redis, models, and other services"""
    global redis_client, diarization_pipeline
    
    try:
        # Initialize Redis
        redis_client = redis.from_url(Config.REDIS_URL)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load default model
    await load_whisper_model(Config.DEFAULT_MODEL)
    
    # Initialize speaker diarization (optional)
    try:
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        if Config.GPU_ENABLED:
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        logger.info("Speaker diarization pipeline loaded")
    except Exception as e:
        logger.warning(f"Speaker diarization not available: {e}")

async def load_whisper_model(model_name: str) -> whisper.Whisper:
    """Load and cache Whisper model"""
    if model_name in models_cache:
        return models_cache[model_name]
    
    logger.info(f"Loading Whisper model: {model_name}")
    
    device = "cuda" if Config.GPU_ENABLED else "cpu"
    model = whisper.load_model(model_name, device=device)
    models_cache[model_name] = model
    
    logger.info(f"Model {model_name} loaded on {device}")
    return model

def get_optimal_model(audio_duration: float, requested_model: Optional[str] = None) -> str:
    """Select optimal model based on audio duration"""
    if requested_model and requested_model in Config.WHISPER_MODELS:
        return requested_model
    
    for threshold, model in Config.MODEL_SELECTION_THRESHOLDS.items():
        if audio_duration < threshold:
            return model
    
    return Config.DEFAULT_MODEL

def preprocess_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Preprocess audio file for optimal transcription"""
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Apply noise reduction (basic)
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

async def perform_diarization(audio_path: str) -> Optional[List[Dict]]:
    """Perform speaker diarization if enabled"""
    if not diarization_pipeline:
        return None
    
    try:
        diarization = diarization_pipeline(audio_path)
        speakers = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start
            })
        
        return speakers
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None

async def transcribe_audio(
    audio_path: str,
    request: TranscriptionRequest,
    websocket: Optional[WebSocket] = None
) -> TranscriptionResponse:
    """Main transcription function"""
    start_time = time.time()
    
    try:
        # Get audio info
        audio_info = sf.info(audio_path)
        duration = audio_info.duration
        
        # Select optimal model
        model_name = get_optimal_model(duration, request.model_size)
        
        # Load model
        model = await load_whisper_model(model_name)
        
        # Send progress update
        if websocket:
            await websocket.send_json({"status": "processing", "progress": 0.2, "stage": "model_loaded"})
        
        # Preprocess audio
        audio_array = preprocess_audio(audio_path)
        
        if websocket:
            await websocket.send_json({"status": "processing", "progress": 0.4, "stage": "audio_preprocessed"})
        
        # Perform transcription
        result = model.transcribe(
            audio_array,
            language=request.language,
            task=request.task,
            verbose=False,
            word_timestamps=request.return_timestamps
        )
        
        if websocket:
            await websocket.send_json({"status": "processing", "progress": 0.8, "stage": "transcription_complete"})
        
        # Perform speaker diarization if requested
        speakers = None
        if request.enable_diarization:
            speakers = await perform_diarization(audio_path)
        
        # Calculate confidence score
        if 'segments' in result:
            confidences = [seg.get('avg_logprob', 0) for seg in result['segments']]
            avg_confidence = np.exp(np.mean(confidences)) if confidences else 0.5
        else:
            avg_confidence = 0.5
        
        processing_time = time.time() - start_time
        
        response = TranscriptionResponse(
            transcript=result['text'].strip(),
            language=result.get('language', 'unknown'),
            confidence=float(avg_confidence),
            processing_time=processing_time,
            segments=result.get('segments', []),
            speakers=speakers,
            model_used=model_name
        )
        
        if websocket:
            await websocket.send_json({"status": "complete", "progress": 1.0, "result": response.dict()})
        
        return response
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(error_msg)
        if websocket:
            await websocket.send_json({"status": "error", "error": error_msg})
        raise HTTPException(status_code=500, detail=error_msg)

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models_cache.keys()),
        gpu_available=Config.GPU_ENABLED,
        uptime=time.time() - start_time
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    task: str = "transcribe",
    enable_diarization: bool = False,
    model_size: Optional[str] = None,
    return_timestamps: bool = True
):
    """Transcribe uploaded audio file"""
    
    # Validate file
    if file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported format. Supported: {Config.SUPPORTED_FORMATS}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        request = TranscriptionRequest(
            language=language,
            task=task,
            enable_diarization=enable_diarization,
            model_size=model_size,
            return_timestamps=return_timestamps
        )
        
        result = await transcribe_audio(temp_file_path, request)
        
        # Schedule cleanup
        background_tasks.add_task(os.unlink, temp_file_path)
        
        return result
        
    except Exception as e:
        # Cleanup on error
        os.unlink(temp_file_path)
        raise

@app.websocket("/transcribe-ws")
async def transcribe_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription progress"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Wait for file data
            data = await websocket.receive_json()
            
            if data.get("type") == "file_upload":
                # Handle file upload via WebSocket
                # Implementation for streaming file upload and processing
                await websocket.send_json({"status": "ready", "message": "Ready to receive audio"})
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

@app.get("/models")
async def list_models():
    """List available and loaded models"""
    return {
        "available_models": Config.WHISPER_MODELS,
        "loaded_models": list(models_cache.keys()),
        "default_model": Config.DEFAULT_MODEL
    }

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Preload a specific model"""
    if model_name not in Config.WHISPER_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    await load_whisper_model(model_name)
    return {"status": "loaded", "model": model_name}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Backlog Builder Whisper STT Service")
    await initialize_services()
    logger.info("Service initialization complete")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "whisper-local:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1,
        log_level="info"
    )