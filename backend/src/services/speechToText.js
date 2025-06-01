const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const EventEmitter = require('events');
const logger = require('../utils/logger');

class SpeechToTextService extends EventEmitter {
    constructor() {
        super();
        this.localSTTUrl = process.env.LOCAL_STT_URL || 'http://localhost:8001';
        this.timeout = parseInt(process.env.STT_TIMEOUT) || 300000; // 5 minutes
        this.maxRetries = parseInt(process.env.STT_MAX_RETRIES) || 3;
        this.retryDelay = parseInt(process.env.STT_RETRY_DELAY) || 1000; // 1 second
        this.preferLocal = process.env.PREFER_LOCAL_STT !== 'false'; // Default to true
        
        // Supported audio formats
        this.supportedFormats = [
            'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac',
            'audio/ogg', 'audio/wma', 'audio/aac', 'audio/mp3'
        ];
        
        this.supportedExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'];
    }

    /**
     * Process audio file through Speech-to-Text
     * @param {string} audioFilePath - Path to the audio file
     * @param {Object} options - Processing options
     * @returns {Object} Transcription result with speaker diarization
     */
    async processAudio(audioFilePath, options = {}) {
        const {
            language = 'auto',
            model = 'base', // Using 'base' model as it's one of the available models
            speakerDiarization = true,
            enableProgress = true,
            priority = 'balanced',
            onProgress = null
        } = options;

        // Validate input file
        try {
            this.validateAudioFile(audioFilePath);
        } catch (error) {
            logger.error(`Audio file validation failed: ${error.message}`, { file: audioFilePath });
            throw new Error(`Invalid audio file: ${error.message}`);
        }

        // Setup progress tracking
        let progress = 0;
        const updateProgress = (newProgress, message = '') => {
            progress = Math.min(100, Math.max(0, newProgress));
            const progressData = { progress, message };
            this.emit('progress', progressData);
            if (typeof onProgress === 'function') {
                onProgress(progressData);
            }
            return progressData;
        };

        try {
            updateProgress(5, 'Starting audio processing...');
            
            // Process with local Whisper service
            const result = await this.retryWithBackoff(
                () => this.processWithLocal(audioFilePath, {
                    language,
                    model,
                    speakerDiarization,
                    enableProgress: true // Always enable progress for internal tracking
                }),
                this.maxRetries,
                this.retryDelay,
                (attempt, error) => {
                    logger.warn(`Attempt ${attempt} failed: ${error.message}`);
                    updateProgress(
                        5 + (attempt * 10), // Progress increases with each retry
                        `Attempt ${attempt} of ${this.maxRetries} failed. Retrying...`
                    );
                }
            );

            updateProgress(100, 'Audio processing completed');
            
            return {
                success: true,
                provider: 'local',
                ...result,
                metadata: {
                    ...(result.metadata || {}),
                    modelUsed: model,
                    languageDetected: result.language || language,
                    processingTime: result.processing_time || 0
                }
            };
            
        } catch (error) {
            logger.error('STT processing failed', { 
                error: error.message, 
                stack: error.stack,
                file: audioFilePath,
                options
            });
            
            updateProgress(0, `Processing failed: ${error.message}`);
            
            throw new Error(`Speech-to-text processing failed: ${error.message}`);
        } finally {
            // Clean up the temporary file if it exists
            try {
                if (fs.existsSync(audioFilePath)) {
                    await fs.promises.unlink(audioFilePath);
                }
            } catch (cleanupError) {
                logger.warn('Failed to clean up audio file', { 
                    file: audioFilePath, 
                    error: cleanupError.message 
                });
            }
        }
    }

    /**
     * Process with local Whisper service
     */
    async processWithLocal(audioFilePath, options) {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(audioFilePath));
        formData.append('language', options.language);
        formData.append('model', options.model);
        formData.append('speaker_diarization', options.speakerDiarization.toString());
        formData.append('enable_progress', options.enableProgress.toString());

        // Setup timeout controller for proper cleanup
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await axios.post(
                `${this.localSTTUrl}/transcribe`, 
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        'Content-Type': 'multipart/form-data',
                        'Accept': 'application/json'
                    },
                    timeout: this.timeout,
                    maxContentLength: Infinity,
                    maxBodyLength: Infinity,
                    signal: controller.signal,
                    onUploadProgress: (progressEvent) => {
                        if (progressEvent.total > 0) {
                            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                            this.emit('upload-progress', { progress: percent });
                        }
                    },
                    onDownloadProgress: (progressEvent) => {
                        if (progressEvent.total > 0) {
                            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                            this.emit('download-progress', { progress: percent });
                        }
                    }
                }
            );

            clearTimeout(timeoutId);
            
            if (!response.data) {
                throw new Error('Empty response from STT service');
            }

            return this.formatLocalResponse(response.data);
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.code === 'ECONNABORTED') {
                throw new Error(`Request timed out after ${this.timeout}ms`);
            } else if (error.response) {
                // The request was made and the server responded with a status code
                // that falls out of the range of 2xx
                const status = error.response.status;
                let message = `STT service error: ${status}`;
                
                if (error.response.data?.error) {
                    message += ` - ${error.response.data.error}`;
                } else if (error.response.data?.message) {
                    message += ` - ${error.response.data.message}`;
                }
                
                throw new Error(message);
            } else if (error.request) {
                // The request was made but no response was received
                throw new Error('No response from STT service');
            } else {
                // Something happened in setting up the request that triggered an Error
                throw new Error(`Request failed: ${error.message}`);
            }
        }
    }

    /**
     * Validate audio file before processing
     */
    validateAudioFile(filePath) {
        if (!fs.existsSync(filePath)) {
            throw new Error('Audio file not found');
        }

        const stats = fs.statSync(filePath);
        const fileSizeMB = stats.size / (1024 * 1024);
        const maxSizeMB = parseInt(process.env.MAX_AUDIO_SIZE_MB) || 500;

        if (fileSizeMB > maxSizeMB) {
            throw new Error(`File size (${fileSizeMB.toFixed(2)}MB) exceeds maximum allowed size (${maxSizeMB}MB)`);
        }

        const ext = path.extname(filePath).toLowerCase();
        const allowedExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'];
        
        if (!allowedExtensions.includes(ext)) {
            throw new Error(`Unsupported file format: ${ext}. Allowed formats: ${allowedExtensions.join(', ')}`);
        }

        return true;
    }

    /**
     * Retry a function with exponential backoff
     */
    async retryWithBackoff(operation, maxRetries, initialDelay, onRetry) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error;
                
                if (attempt === maxRetries) {
                    break; // Don't wait on the last attempt
                }
                
                if (onRetry) {
                    onRetry(attempt, error);
                }
                
                // Exponential backoff with jitter
                const delay = initialDelay * Math.pow(2, attempt - 1);
                const jitter = Math.floor(Math.random() * 1000);
                await new Promise(resolve => setTimeout(resolve, delay + jitter));
            }
        }
        
        throw lastError;
    }

    /**
     * Format the local STT response
     */
    formatLocalResponse(data) {
        console.log('=== RAW STT RESPONSE ===');
        console.log('Data type:', typeof data);
        console.log('Data keys:', data ? Object.keys(data) : 'null');
        console.log('Data content:', JSON.stringify(data, null, 2).substring(0, 500) + '...');
        
        if (!data) {
            throw new Error('Empty response from STT service');
        }
        
        // Handle different response formats with comprehensive output
        const response = {
            transcription: data.transcript || data.transcription || data.text || '',
            segments: data.segments || [],
            speakers: data.speakers || [],
            language: data.language || data.language_detected || 'en',
            duration: data.duration || 0,
            confidence: data.confidence || 0,
            model: data.model_used || data.model || 'unknown',
            processingTime: data.processing_time || 0,
            metadata: {
                wordCount: data.word_count || 0,
                speakerCount: data.speaker_count || (data.speakers ? data.speakers.length : 0),
                hasTimestamps: Boolean(data.segments?.length),
                hasSpeakerDiarization: Boolean(data.speakers?.length),
                ...(data.metadata || {})
            }
        };

        // Also include the simpler format for backward compatibility
        return {
            ...response,
            success: true,
            text: response.transcription,
            processing_time: response.processingTime,
            language_detected: response.language
        };
    }

    /**
     * Get transcription progress for long-running jobs
     */
    async getProgress(jobId, provider = 'local') {
        try {
            // Only local provider available (cloud disabled)
            const baseUrl = this.localSTTUrl;
            const response = await axios.get(`${baseUrl}/progress/${jobId}`);
            return response.data;
        } catch (error) {
            console.error('Failed to get progress:', error);
            return { progress: 0, status: 'error', error: error.message };
        }
    }

    /**
     * Check if local service is available
     */
    async isLocalServiceAvailable() {
        try {
            const response = await axios.get(`${this.localSTTUrl}/health`, { timeout: 5000 });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    /**
     * Get service status and capabilities
     */
    async getServiceStatus() {
        const localAvailable = await this.isLocalServiceAvailable();

        return {
            local: {
                available: localAvailable,
                url: this.localSTTUrl,
                preferred: this.preferLocal
            },
            timeout: this.timeout,
            maxRetries: this.maxRetries,
            supportedFormats: this.supportedFormats,
            supportedExtensions: this.supportedExtensions
        };
    }
}

module.exports = new SpeechToTextService();