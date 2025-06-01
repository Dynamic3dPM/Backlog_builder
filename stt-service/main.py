from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for job status
jobs: Dict[str, Dict] = {}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model cache
model_cache = {}

def get_model(model_size: str = "base"):
    if model_size not in model_cache:
        logger.info(f"Loading model: {model_size}")
        model_cache[model_size] = WhisperModel(
            model_size,
            device="cpu",  # Change to "cuda" if you have a GPU
            compute_type="int8"  # Use "float16" for better quality if you have a GPU
        )
    return model_cache[model_size]

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: Optional[str] = None,
    model: str = "base",
    speaker_diarization: bool = False,
    enable_progress: bool = False
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting transcription...",
        "result": None,
        "error": None
    }
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process in background
    import threading
    thread = threading.Thread(
        target=process_audio,
        args=(job_id, file_path, language, model, speaker_diarization, enable_progress)
    )
    thread.start()
    
    return {"job_id": job_id, "status": "started"}

def process_audio(job_id: str, file_path: str, language: str, model_size: str, diarize: bool, enable_progress: bool):
    try:
        jobs[job_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Loading model..."
        })
        
        model = get_model(model_size)
        
        jobs[job_id].update({
            "progress": 20,
            "message": "Transcribing audio..."
        })
        
        # Transcribe the audio
        segments, info = model.transcribe(
            file_path,
            language=language if language != "auto" else None,
            beam_size=5,
            word_timestamps=diarize
        )
        
        # Convert to list to consume the generator
        segments = list(segments)
        
        # Prepare the result
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": sum(segment.end - segment.start for segment in segments),
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in (segment.words or [])
                    ]
                }
                for segment in segments
            ]
        }
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Transcription complete",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "error": str(e)
        })
    finally:
        # Clean up the uploaded file
        try:
            os.remove(file_path)
        except:
            pass

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(model_cache.keys()),
        "gpu_available": False,  # Change to True if using GPU
        "uptime": 0  # You can implement this if needed
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)