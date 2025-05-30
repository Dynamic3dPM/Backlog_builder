#!/usr/bin/env node
/**
 * Backlog Builder Cloud Speech-to-Text Service
 * Multi-provider cloud STT integration with intelligent fallback
 * 
 * TEMPORARILY COMMENTED OUT FOR LOCAL-ONLY DEPLOYMENT
 * Uncomment this entire file when cloud functionality is needed
 */

/* CLOUD SERVICE TEMPORARILY DISABLED
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// Cloud provider SDKs
const speech = require('@google-cloud/speech');
const { CognitiveServicesCredentials } = require('@azure/ms-rest-azure-js');
const { SpeechConfig, AudioConfig, SpeechRecognizer } = require('microsoft-cognitiveservices-speech-sdk');
const AWS = require('aws-sdk');
const axios = require('axios');
const WebSocket = require('ws');

// Configuration
const CONFIG = {
    PORT: process.env.STT_PORT || 8003,
    MAX_FILE_SIZE: 500 * 1024 * 1024, // 500MB
    SUPPORTED_FORMATS: ['.wav', '.mp3', '.flac', '.m4a', '.ogg'],
    
    // Provider configurations
    PROVIDERS: {
        GOOGLE: {
            enabled: !!process.env.GOOGLE_CLOUD_KEY_FILE,
            keyFile: process.env.GOOGLE_CLOUD_KEY_FILE,
            projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
            costPerMinute: 0.006, // USD
            maxFileSizeMB: 1000,
            supportedLanguages: ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN']
        },
        AZURE: {
            enabled: !!(process.env.AZURE_SPEECH_KEY && process.env.AZURE_SPEECH_REGION),
            subscriptionKey: process.env.AZURE_SPEECH_KEY,
            region: process.env.AZURE_SPEECH_REGION,
            costPerMinute: 0.0085, // USD
            maxFileSizeMB: 200,
            supportedLanguages: ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN']
        },
        AWS: {
            enabled: !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY),
            accessKeyId: process.env.AWS_ACCESS_KEY_ID,
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
            region: process.env.AWS_REGION || 'us-east-1',
            costPerMinute: 0.004, // USD
            maxFileSizeMB: 2000,
            supportedLanguages: ['en-US', 'es-US', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN']
        }
    },
    
    // Retry configuration
    RETRY: {
        maxAttempts: 3,
        baseDelay: 1000, // ms
        maxDelay: 10000, // ms
        backoffFactor: 2
    }
};

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({
    dest: 'uploads/',
    limits: { fileSize: CONFIG.MAX_FILE_SIZE },
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        if (CONFIG.SUPPORTED_FORMATS.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported file format. Supported: ${CONFIG.SUPPORTED_FORMATS.join(', ')}`));
        }
    }
});

// Initialize cloud providers
let cloudProviders = {};

class CloudSTTManager {
    constructor() {
        this.initializeProviders();
        this.requestQueue = new Map();
        this.costTracking = {
            google: { requests: 0, totalCost: 0, audioMinutes: 0 },
            azure: { requests: 0, totalCost: 0, audioMinutes: 0 },
            aws: { requests: 0, totalCost: 0, audioMinutes: 0 }
        };
    }

    async initializeProviders() {
        // Initialize Google Speech-to-Text
        if (CONFIG.PROVIDERS.GOOGLE.enabled) {
            try {
                cloudProviders.google = new speech.SpeechClient({
                    keyFilename: CONFIG.PROVIDERS.GOOGLE.keyFile,
                    projectId: CONFIG.PROVIDERS.GOOGLE.projectId
                });
                console.log('✅ Google Speech-to-Text initialized');
            } catch (error) {
                console.warn('⚠️ Google Speech-to-Text initialization failed:', error.message);
            }
        }

        // Initialize Azure Cognitive Services
        if (CONFIG.PROVIDERS.AZURE.enabled) {
            try {
                cloudProviders.azure = {
                    subscriptionKey: CONFIG.PROVIDERS.AZURE.subscriptionKey,
                    region: CONFIG.PROVIDERS.AZURE.region
                };
                console.log('✅ Azure Speech Services initialized');
            } catch (error) {
                console.warn('⚠️ Azure Speech Services initialization failed:', error.message);
            }
        }

        // Initialize AWS Transcribe
        if (CONFIG.PROVIDERS.AWS.enabled) {
            try {
                AWS.config.update({
                    accessKeyId: CONFIG.PROVIDERS.AWS.accessKeyId,
                    secretAccessKey: CONFIG.PROVIDERS.AWS.secretAccessKey,
                    region: CONFIG.PROVIDERS.AWS.region
                });
                cloudProviders.aws = new AWS.TranscribeService();
                console.log('✅ AWS Transcribe initialized');
            } catch (error) {
                console.warn('⚠️ AWS Transcribe initialization failed:', error.message);
            }
        }
    }

    selectOptimalProvider(audioLengthMinutes, language = 'en-US', priority = 'cost') {
        const availableProviders = [];
        
        // Check which providers support the language and file size
        Object.entries(CONFIG.PROVIDERS).forEach(([name, config]) => {
            if (!config.enabled || !cloudProviders[name.toLowerCase()]) return;
            
            if (config.supportedLanguages.includes(language)) {
                availableProviders.push({
                    name: name.toLowerCase(),
                    cost: config.costPerMinute * audioLengthMinutes,
                    reliability: this.getProviderReliability(name.toLowerCase()),
                    speed: this.getProviderSpeed(name.toLowerCase())
                });
            }
        });

        if (availableProviders.length === 0) {
            throw new Error(`No providers available for language: ${language}`);
        }

        // Sort by priority
        switch (priority) {
            case 'cost':
                return availableProviders.sort((a, b) => a.cost - b.cost)[0];
            case 'speed':
                return availableProviders.sort((a, b) => b.speed - a.speed)[0];
            case 'reliability':
                return availableProviders.sort((a, b) => b.reliability - a.reliability)[0];
            default:
                // Balanced scoring
                return availableProviders.sort((a, b) => {
                    const scoreA = (a.reliability * 0.4) + (a.speed * 0.3) + ((1 / a.cost) * 0.3);
                    const scoreB = (b.reliability * 0.4) + (b.speed * 0.3) + ((1 / b.cost) * 0.3);
                    return scoreB - scoreA;
                })[0];
        }
    }

    getProviderReliability(provider) {
        // Simple reliability scoring based on recent success rate
        const stats = this.costTracking[provider];
        if (stats.requests === 0) return 0.9; // Default for new providers
        
        // This would be calculated from actual success/failure tracking
        return 0.95; // Placeholder
    }

    getProviderSpeed(provider) {
        // Speed scoring (higher is better)
        const speedMap = {
            'google': 0.8,
            'azure': 0.9,
            'aws': 0.7
        };
        return speedMap[provider] || 0.5;
    }

    async transcribeWithGoogle(audioBuffer, options = {}) {
        if (!cloudProviders.google) {
            throw new Error('Google Speech-to-Text not available');
        }

        const request = {
            audio: { content: audioBuffer },
            config: {
                encoding: options.encoding || 'LINEAR16',
                sampleRateHertz: options.sampleRate || 16000,
                languageCode: options.language || 'en-US',
                enableWordTimeOffsets: options.timestamps || true,
                enableAutomaticPunctuation: true,
                model: options.useEnhanced ? 'latest_long' : 'latest_short',
                useEnhanced: options.useEnhanced || false
            }
        };

        const [response] = await cloudProviders.google.recognize(request);
        
        return {
            transcript: response.results
                .map(result => result.alternatives[0].transcript)
                .join(' '),
            confidence: response.results.length > 0 
                ? response.results[0].alternatives[0].confidence 
                : 0,
            segments: response.results.map(result => ({
                text: result.alternatives[0].transcript,
                confidence: result.alternatives[0].confidence,
                words: result.alternatives[0].words || []
            })),
            provider: 'google'
        };
    }

    async transcribeWithAzure(audioBuffer, options = {}) {
        if (!cloudProviders.azure) {
            throw new Error('Azure Speech Services not available');
        }

        // Create temporary file for Azure SDK
        const tempFile = `temp_${uuidv4()}.wav`;
        await fs.writeFile(tempFile, audioBuffer);

        try {
            const speechConfig = SpeechConfig.fromSubscription(
                cloudProviders.azure.subscriptionKey,
                cloudProviders.azure.region
            );
            speechConfig.speechRecognitionLanguage = options.language || 'en-US';
            
            const audioConfig = AudioConfig.fromWavFileInput(await fs.readFile(tempFile));
            const recognizer = new SpeechRecognizer(speechConfig, audioConfig);

            return new Promise((resolve, reject) => {
                recognizer.recognizeOnceAsync(
                    result => {
                        if (result.reason === 1) { // ResultReason.RecognizedSpeech
                            resolve({
                                transcript: result.text,
                                confidence: result.json ? JSON.parse(result.json).Confidence : 0.9,
                                segments: [{ text: result.text, confidence: 0.9 }],
                                provider: 'azure'
                            });
                        } else {
                            reject(new Error(`Azure recognition failed: ${result.errorDetails}`));
                        }
                        recognizer.close();
                    },
                    error => {
                        recognizer.close();
                        reject(error);
                    }
                );
            });
        } finally {
            // Cleanup
            try {
                await fs.unlink(tempFile);
            } catch (e) {
                console.warn('Failed to cleanup temp file:', e.message);
            }
        }
    }

    async transcribeWithAWS(audioBuffer, options = {}) {
        if (!cloudProviders.aws) {
            throw new Error('AWS Transcribe not available');
        }

        // AWS Transcribe requires files to be in S3, so we'll use a simplified approach
        // In production, you'd upload to S3 and use async transcription jobs
        
        const jobName = `Backlog Builder-${uuidv4()}`;
        const params = {
            TranscriptionJobName: jobName,
            LanguageCode: options.language || 'en-US',
            MediaFormat: 'wav',
            Media: {
                MediaFileUri: `s3://your-bucket/audio/${jobName}.wav`
            },
            OutputBucketName: 'your-transcription-bucket'
        };

        // This is a placeholder - in real implementation you'd:
        // 1. Upload file to S3
        // 2. Start transcription job
        // 3. Poll for completion
        // 4. Download and parse results
        
        throw new Error('AWS Transcribe requires S3 setup - not implemented in demo');
    }

    async retryWithBackoff(fn, maxAttempts = CONFIG.RETRY.maxAttempts) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (attempt === maxAttempts) {
                    break;
                }
                
                const delay = Math.min(
                    CONFIG.RETRY.baseDelay * Math.pow(CONFIG.RETRY.backoffFactor, attempt - 1),
                    CONFIG.RETRY.maxDelay
                );
                
                console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms:`, error.message);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        
        throw lastError;
    }

    async transcribe(audioBuffer, options = {}) {
        const audioLengthMinutes = options.duration || 1; // Estimate if not provided
        
        // Select optimal provider
        const provider = this.selectOptimalProvider(
            audioLengthMinutes,
            options.language,
            options.priority
        );
        
        console.log(`Using provider: ${provider.name} for transcription`);
        
        // Define fallback order
        const fallbackOrder = ['google', 'azure', 'aws'].filter(p => 
            p !== provider.name && cloudProviders[p]
        );
        
        // Try primary provider with retry
        try {
            const result = await this.retryWithBackoff(async () => {
                switch (provider.name) {
                    case 'google':
                        return await this.transcribeWithGoogle(audioBuffer, options);
                    case 'azure':
                        return await this.transcribeWithAzure(audioBuffer, options);
                    case 'aws':
                        return await this.transcribeWithAWS(audioBuffer, options);
                    default:
                        throw new Error(`Unknown provider: ${provider.name}`);
                }
            });
            
            // Track successful request
            this.updateCostTracking(provider.name, audioLengthMinutes, true);
            return result;
            
        } catch (primaryError) {
            console.warn(`Primary provider ${provider.name} failed:`, primaryError.message);
            
            // Try fallback providers
            for (const fallbackProvider of fallbackOrder) {
                try {
                    console.log(`Trying fallback provider: ${fallbackProvider}`);
                    
                    const result = await this.retryWithBackoff(async () => {
                        switch (fallbackProvider) {
                            case 'google':
                                return await this.transcribeWithGoogle(audioBuffer, options);
                            case 'azure':
                                return await this.transcribeWithAzure(audioBuffer, options);
                            case 'aws':
                                return await this.transcribeWithAWS(audioBuffer, options);
                        }
                    });
                    
                    this.updateCostTracking(fallbackProvider, audioLengthMinutes, true);
                    return result;
                    
                } catch (fallbackError) {
                    console.warn(`Fallback provider ${fallbackProvider} failed:`, fallbackError.message);
                    continue;
                }
            }
            
            // All providers failed
            throw new Error('All speech-to-text providers failed');
        }
    }

    updateCostTracking(provider, audioMinutes, success) {
        if (!this.costTracking[provider]) return;
        
        this.costTracking[provider].requests++;
        if (success) {
            this.costTracking[provider].audioMinutes += audioMinutes;
            this.costTracking[provider].totalCost += 
                CONFIG.PROVIDERS[provider.toUpperCase()].costPerMinute * audioMinutes;
        }
    }

    getCostSummary() {
        return this.costTracking;
    }
}

// Initialize STT manager
const sttManager = new CloudSTTManager();

// API Routes
app.get('/health', (req, res) => {
    const availableProviders = Object.keys(cloudProviders);
    res.json({
        status: 'healthy',
        providers: availableProviders,
        uptime: process.uptime()
    });
});

app.post('/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file provided' });
        }

        const audioBuffer = await fs.readFile(req.file.path);
        
        const options = {
            language: req.body.language || 'en-US',
            timestamps: req.body.timestamps !== 'false',
            priority: req.body.priority || 'balanced',
            useEnhanced: req.body.enhanced === 'true'
        };

        const result = await sttManager.transcribe(audioBuffer, options);
        
        // Cleanup uploaded file
        await fs.unlink(req.file.path);
        
        res.json({
            success: true,
            ...result,
            processing_time: Date.now() - req.file.uploadTime,
            file_size: req.file.size
        });

    } catch (error) {
        console.error('Transcription error:', error);
        
        // Cleanup on error
        if (req.file) {
            try {
                await fs.unlink(req.file.path);
            } catch (e) {
                console.warn('Failed to cleanup file:', e.message);
            }
        }
        
        res.status(500).json({
            error: 'Transcription failed',
            message: error.message
        });
    }
});

app.get('/providers', (req, res) => {
    const providersInfo = Object.entries(CONFIG.PROVIDERS)
        .filter(([name, config]) => config.enabled && cloudProviders[name.toLowerCase()])
        .map(([name, config]) => ({
            name: name.toLowerCase(),
            costPerMinute: config.costPerMinute,
            maxFileSizeMB: config.maxFileSizeMB,
            supportedLanguages: config.supportedLanguages
        }));
    
    res.json({ providers: providersInfo });
});

app.get('/costs', (req, res) => {
    res.json(sttManager.getCostSummary());
});

// WebSocket for real-time transcription
const wss = new WebSocket.Server({ port: 8004 });

wss.on('connection', (ws) => {
    console.log('WebSocket connection established');
    
    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            
            if (data.type === 'audio_chunk') {
                // Handle streaming audio transcription
                // This would implement real-time processing
                ws.send(JSON.stringify({
                    type: 'partial_result',
                    text: 'Streaming transcription not implemented in demo',
                    confidence: 0.5
                }));
            }
        } catch (error) {
            ws.send(JSON.stringify({
                type: 'error',
                message: error.message
            }));
        }
    });

    ws.on('close', () => {
        console.log('WebSocket connection closed');
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Express error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server
app.listen(CONFIG.PORT, () => {
    console.log(`🎙️ Backlog Builder Cloud STT Service running on port ${CONFIG.PORT}`);
    console.log(`📊 Available providers: ${Object.keys(cloudProviders).join(', ')}`);
});

module.exports = app;

*/ // END CLOUD SERVICE TEMPORARILY DISABLED

// Placeholder service for local-only deployment
const express = require('express');
const app = express();

app.use(express.json());

// Return error for all cloud STT requests when disabled
app.all('*', (req, res) => {
    res.status(503).json({
        error: 'Cloud STT service temporarily disabled',
        message: 'System is running in local-only mode. To enable cloud services, uncomment the cloud-stt.js file.',
        provider: 'disabled'
    });
});

const PORT = process.env.STT_PORT || 8003;
app.listen(PORT, () => {
    console.log(`🚫 Cloud STT Service DISABLED - placeholder running on port ${PORT}`);
    console.log(`💡 To enable: Uncomment the main service code in cloud-stt.js`);
});

module.exports = app;