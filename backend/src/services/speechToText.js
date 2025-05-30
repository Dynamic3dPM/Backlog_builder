const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

class SpeechToTextService {
    constructor() {
        this.localSTTUrl = process.env.LOCAL_STT_URL || 'http://localhost:8001';
        // CLOUD SERVICE TEMPORARILY DISABLED
        // this.cloudSTTUrl = process.env.CLOUD_STT_URL || 'http://localhost:8002';
        this.preferLocal = true; // Force local processing only
        this.timeout = parseInt(process.env.STT_TIMEOUT) || 300000; // 5 minutes
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
            model = 'auto',
            speakerDiarization = true,
            enableProgress = true,
            priority = 'balanced'
        } = options;

        try {
            // Try local service only (cloud disabled)
            try {
                const result = await this.processWithLocal(audioFilePath, {
                    language,
                    model,
                    speakerDiarization,
                    enableProgress
                });
                
                if (result.success) {
                    return {
                        success: true,
                        provider: 'local',
                        ...result
                    };
                }
            } catch (localError) {
                console.error('Local STT processing failed:', localError.message);
                throw new Error(`STT processing failed: ${localError.message}. Cloud services are temporarily disabled.`);
            }

            /* CLOUD FALLBACK TEMPORARILY DISABLED
            // Fallback to cloud service
            const cloudResult = await this.processWithCloud(audioFilePath, {
                language,
                priority,
                speakerDiarization
            });

            return {
                success: true,
                provider: 'cloud',
                ...cloudResult
            };
            */ // END CLOUD FALLBACK TEMPORARILY DISABLED

        } catch (error) {
            console.error('STT processing failed:', error);
            throw new Error(`Speech-to-text processing failed: ${error.message}`);
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

        const response = await axios.post(`${this.localSTTUrl}/transcribe`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Content-Type': 'multipart/form-data'
            },
            timeout: this.timeout,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        return this.formatLocalResponse(response.data);
    }

    /* CLOUD SERVICE TEMPORARILY DISABLED
    // Process with cloud STT service
    async processWithCloud(audioFilePath, options) {
        const formData = new FormData();
        formData.append('audio', fs.createReadStream(audioFilePath));
        formData.append('language', options.language);
        formData.append('priority', options.priority);
        formData.append('speakerDiarization', options.speakerDiarization.toString());

        const response = await axios.post(`${this.cloudSTTUrl}/transcribe`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Content-Type': 'multipart/form-data'
            },
            timeout: this.timeout,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        return this.formatCloudResponse(response.data);
    }
    */ // END CLOUD SERVICE TEMPORARILY DISABLED

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

    /* CLOUD SERVICE TEMPORARILY DISABLED
    // Check if cloud service is available
    async isCloudServiceAvailable() {
        try {
            const response = await axios.get(`${this.cloudSTTUrl}/health`, { timeout: 5000 });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }
    */ // END CLOUD SERVICE TEMPORARILY DISABLED

    /**
     * Get service status and capabilities
     */
    async getServiceStatus() {
        const localAvailable = await this.isLocalServiceAvailable();
        // Cloud services temporarily disabled

        return {
            local: {
                available: localAvailable,
                url: this.localSTTUrl,
                preferred: this.preferLocal
            },
            /* CLOUD SERVICE TEMPORARILY DISABLED
            cloud: {
                available: cloudAvailable,
                url: this.cloudSTTUrl
            },
            */ // END CLOUD SERVICE TEMPORARILY DISABLED
            timeout: this.timeout
        };
    }

    /**
     * Format local service response
     */
    formatLocalResponse(data) {
        return {
            transcription: data.transcription,
            segments: data.segments || [],
            speakers: data.speakers || [],
            language: data.language,
            duration: data.duration,
            confidence: data.confidence,
            model: data.model_used,
            processingTime: data.processing_time,
            metadata: {
                wordCount: data.word_count,
                speakerCount: data.speaker_count,
                hasTimestamps: Boolean(data.segments?.length),
                hasSpeakerDiarization: Boolean(data.speakers?.length)
            }
        };
    }

    /* CLOUD SERVICE TEMPORARILY DISABLED
    // Format cloud service response
    formatCloudResponse(data) {
        return {
            transcription: data.transcription,
            segments: data.segments || [],
            speakers: data.speakers || [],
            language: data.language,
            duration: data.duration,
            confidence: data.confidence,
            provider: data.provider,
            cost: data.cost,
            metadata: {
                wordCount: data.metadata?.wordCount || 0,
                speakerCount: data.metadata?.speakerCount || 0,
                hasTimestamps: Boolean(data.segments?.length),
                hasSpeakerDiarization: Boolean(data.speakers?.length),
                estimatedCost: data.cost
            }
        };
    }
    */ // END CLOUD SERVICE TEMPORARILY DISABLED

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
}

module.exports = new SpeechToTextService();