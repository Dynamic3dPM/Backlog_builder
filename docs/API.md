# Backlog Builder API Documentation

## Base URL
```
http://localhost:3000
```

## Authentication
Currently, the API endpoints are marked as Private but authentication middleware may not be fully implemented. Check the actual implementation for current auth requirements.

## Endpoints

### Health Check
#### GET /health
Check if the backend service is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-05-31T12:16:26.961Z",
  "uptime": 28.961293623
}
```

### AI Services

#### GET /api/ai/health
Health check specifically for AI services.

#### GET /api/ai/status
Get the status and availability of AI services.

#### POST /api/ai/transcribe
Transcribe audio file to text with speaker diarization.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `audio` file (mp3, wav, m4a, flac, ogg, wma, aac)
- Max file size: 500MB

**Response:**
```json
{
  "success": true,
  "transcription": "...",
  "speakers": [...],
  "duration": 120.5
}
```

#### POST /api/ai/analyze
Analyze meeting transcription to extract insights.

**Request:**
```json
{
  "transcription": "Meeting transcription text...",
  "options": {
    "extractActionItems": true,
    "generateSummary": true,
    "identifyDecisions": true
  }
}
```

#### POST /api/ai/generate-tickets
Generate tickets from meeting analysis.

**Request:**
```json
{
  "analysis": {...},
  "project": "project-name",
  "template": "default"
}
```

#### POST /api/ai/process-audio
Complete audio processing pipeline (transcribe + analyze + generate tickets).

**Request:**
- Content-Type: `multipart/form-data`
- Body: `audio` file + processing options

#### GET /api/ai/progress/:jobId
Get processing progress for long-running AI jobs.

**Response:**
```json
{
  "jobId": "123",
  "status": "processing",
  "progress": 45,
  "stage": "transcription",
  "eta": 120
}
```

### Component-Specific AI Endpoints

#### POST /api/ai/action-items
Generate action items from transcription.

#### POST /api/ai/summary
Generate meeting summary from transcription.

#### POST /api/ai/decisions
Extract decisions from meeting transcription.

### Templates and Configuration

#### GET /api/ai/templates
Get available prompt templates.

#### POST /api/ai/templates
Create or update prompt template.

#### GET /api/ai/providers
Get available AI providers and their capabilities.

#### POST /api/ai/providers/test
Test connectivity to AI providers.

### Batch Processing

#### POST /api/ai/batch/process
Process multiple audio files in batch (max 10 files).

#### GET /api/ai/batch/:batchId
Get batch processing status.

### Analytics

#### GET /api/ai/analytics/usage
Get AI service usage analytics.

#### GET /api/ai/analytics/costs
Get AI service cost analytics.

#### GET /api/ai/analytics/performance
Get AI service performance metrics.

### WebSocket Endpoints

#### GET /api/ai/ws/transcribe/:jobId
Real-time transcription updates via WebSocket.

#### GET list of all projects in GitHub


#### GET /api/ai/ws/progress/:jobId
Real-time progress updates via WebSocket.

## Error Responses

All endpoints return errors in this format:
```json
{
  "success": false,
  "error": "Error description",
  "message": "Detailed error message (development only)"
}
```

## File Upload Constraints

- **Supported formats:** mp3, wav, m4a, flac, ogg, wma, aac
- **Max file size:** 500MB (configurable via MAX_AUDIO_SIZE_MB env var)
- **Max files per batch:** 10

## Environment Variables

- `MAX_AUDIO_SIZE_MB`: Maximum upload file size (default: 500)
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads/audio)