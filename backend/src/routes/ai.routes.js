const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { AIController, addRequestTiming } = require('../controllers/ai.controller');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = process.env.UPLOAD_DIR || './uploads/audio';
        
        // Ensure upload directory exists
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        const ext = path.extname(file.originalname);
        const name = path.basename(file.originalname, ext);
        cb(null, `${name}-${timestamp}${ext}`);
    }
});

const upload = multer({
    storage,
    limits: {
        fileSize: parseInt(process.env.MAX_AUDIO_SIZE_MB || '500') * 1024 * 1024 // Default 500MB
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = [
            'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac', 
            'audio/ogg', 'audio/wma', 'audio/aac', 'audio/mp3'
        ];
        
        const allowedExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'];
        const ext = path.extname(file.originalname).toLowerCase();
        
        if (allowedTypes.includes(file.mimetype) || allowedExtensions.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported file format: ${ext}. Allowed formats: ${allowedExtensions.join(', ')}`));
        }
    }
});

// Middleware for request validation
const validateProcessingOptions = (req, res, next) => {
    const {
        language,
        priority,
        model,
        includeActionItems,
        includeDecisions,
        includeSummary,
        includeTickets
    } = req.body;

    // Validate language
    if (language && !['auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'].includes(language)) {
        return res.status(400).json({
            success: false,
            error: 'Invalid language code'
        });
    }

    // Validate model (if provided)
    if (model && !['tiny', 'base'].includes(model)) {
        return res.status(400).json({
            success: false,
            error: 'Invalid model. Must be one of: tiny, base'
        });
    }

    // Validate priority
    if (priority && !['speed', 'balanced', 'quality'].includes(priority)) {
        return res.status(400).json({
            success: false,
            error: 'Invalid priority. Must be: speed, balanced, or quality'
        });
    }

    // Convert string booleans to actual booleans
    if (includeActionItems !== undefined) {
        req.body.includeActionItems = includeActionItems === 'true';
    }
    if (includeDecisions !== undefined) {
        req.body.includeDecisions = includeDecisions === 'true';
    }
    if (includeSummary !== undefined) {
        req.body.includeSummary = includeSummary === 'true';
    }
    if (includeTickets !== undefined) {
        req.body.includeTickets = includeTickets === 'true';
    }

    next();
};

// AI Processing Routes

/**
 * @route POST /api/ai/process-audio
 * @desc Process audio file through complete AI pipeline (STT + Analysis + Tickets)
 * @access Private
 */
router.post('/process-audio', 
    upload.single('audio'), 
    validateProcessingOptions,
    AIController.processAudioComplete.bind(AIController)
);

/**
 * @route POST /api/ai/transcribe
 * @desc Transcribe audio file to text with speaker diarization
 * @access Private
 */
router.post('/transcribe', 
    upload.single('audio'), 
    validateProcessingOptions,
    AIController.transcribeAudio.bind(AIController)
);

/**
 * @route POST /api/ai/analyze
 * @desc Analyze meeting transcription to extract insights
 * @access Private
 */
router.post('/analyze', AIController.analyzeMeeting.bind(AIController));

/**
 * @route POST /api/ai/generate-tickets
 * @desc Generate tickets from meeting analysis
 * @access Private
 */
router.post('/generate-tickets', AIController.generateTickets.bind(AIController));

/**
 * @route GET /api/ai/progress/:jobId
 * @desc Get processing progress for long-running AI jobs
 * @access Private
 */
router.get('/progress/:jobId', AIController.getProgress.bind(AIController));

/**
 * @route GET /api/ai/status
 * @desc Get AI services status and availability
 * @access Private
 */
router.get('/status', AIController.getServiceStatus.bind(AIController));

/**
 * @route GET /api/ai/health
 * @desc Health check for AI services
 * @access Public
 */
router.get('/health', AIController.healthCheck.bind(AIController));

// Component-specific routes

/**
 * @route POST /api/ai/action-items
 * @desc Generate action items from transcription
 * @access Private
 */
router.post('/action-items', AIController.generateActionItems.bind(AIController));

/**
 * @route POST /api/ai/summary
 * @desc Generate meeting summary from transcription
 * @access Private
 */
router.post('/summary', AIController.generateSummary.bind(AIController));

/**
 * @route POST /api/ai/decisions
 * @desc Extract decisions from meeting transcription
 * @access Private
 */
router.post('/decisions', AIController.extractDecisions.bind(AIController));

// Template and configuration routes

/**
 * @route GET /api/ai/templates
 * @desc Get available prompt templates
 * @access Private
 */
router.get('/templates', AIController.getPromptTemplates.bind(AIController));

/**
 * @route POST /api/ai/templates
 * @desc Create or update prompt template
 * @access Private
 */
router.post('/templates', AIController.updatePromptTemplate.bind(AIController));

/**
 * @route GET /api/ai/providers
 * @desc Get available AI providers and their capabilities
 * @access Private
 */
router.get('/providers', AIController.getProviders.bind(AIController));

/**
 * @route POST /api/ai/providers/test
 * @desc Test connectivity to AI providers
 * @access Private
 */
router.post('/providers/test', AIController.testProviders.bind(AIController));

// Batch processing routes

/**
 * @route POST /api/ai/batch/process
 * @desc Process multiple audio files in batch
 * @access Private
 */
router.post('/batch/process', 
    upload.array('audio', 10), // Maximum 10 files
    validateProcessingOptions,
    AIController.processBatch.bind(AIController)
);

/**
 * @route GET /api/ai/batch/:batchId
 * @desc Get batch processing status
 * @access Private
 */
router.get('/batch/:batchId', AIController.getBatchStatus.bind(AIController));

// Analytics and reporting routes

/**
 * @route GET /api/ai/analytics/usage
 * @desc Get AI service usage analytics
 * @access Private
 */
router.get('/analytics/usage', AIController.getUsageAnalytics.bind(AIController));

/**
 * @route GET /api/ai/analytics/costs
 * @desc Get AI service cost analytics
 * @access Private
 */
router.get('/analytics/costs', AIController.getCostAnalytics.bind(AIController));

/**
 * @route GET /api/ai/analytics/performance
 * @desc Get AI service performance metrics
 * @access Private
 */
router.get('/analytics/performance', AIController.getPerformanceAnalytics.bind(AIController));

// WebSocket routes for real-time updates

/**
 * @route GET /api/ai/ws/transcribe/:jobId
 * @desc WebSocket endpoint for real-time transcription updates
 * @access Private
 */
router.get('/ws/transcribe/:jobId', AIController.transcribeWebSocket.bind(AIController));

/**
 * @route GET /api/ai/ws/progress/:jobId
 * @desc WebSocket endpoint for real-time progress updates
 * @access Private
 */
router.get('/ws/progress/:jobId', AIController.progressWebSocket.bind(AIController));

// Error handling middleware
router.use((error, req, res, next) => {
    console.error('AI routes error:', error);

    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                error: 'File size too large',
                maxSize: `${process.env.MAX_AUDIO_SIZE_MB || '500'}MB`
            });
        }
        
        if (error.code === 'LIMIT_FILE_COUNT') {
            return res.status(400).json({
                success: false,
                error: 'Too many files uploaded',
                maxFiles: 10
            });
        }
    }

    if (error.message.includes('Unsupported file format')) {
        return res.status(400).json({
            success: false,
            error: error.message
        });
    }

    res.status(500).json({
        success: false,
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
    });
});

module.exports = router;