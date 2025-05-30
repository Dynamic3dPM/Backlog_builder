const axios = require('axios');
const fs = require('fs').promises;

class AIAnalysisService {
    constructor() {
        this.localLLMUrl = process.env.LOCAL_LLM_URL || 'http://localhost:8003';
        // CLOUD SERVICE TEMPORARILY DISABLED
        // this.cloudLLMUrl = process.env.CLOUD_LLM_URL || 'http://localhost:8004';
        this.preferLocal = true; // Force local processing only
        this.timeout = parseInt(process.env.LLM_TIMEOUT) || 120000; // 2 minutes
        this.promptTemplatesPath = process.env.PROMPT_TEMPLATES_PATH || 
            '/home/timc/Documents/github/Backlog_builder/ai-services/llm-processing/prompt-templates.json';
    }

    /**
     * Analyze meeting transcription and generate insights
     * @param {string} transcription - The meeting transcription text
     * @param {Object} options - Analysis options
     * @returns {Object} Analysis results with action items, decisions, and summary
     */
    async analyzeMeeting(transcription, options = {}) {
        const {
            meetingType = 'general',
            complexity = 'medium',
            includeActionItems = true,
            includeDecisions = true,
            includeSummary = true,
            includeTickets = true,
            template = null,
            priority = 'balanced'
        } = options;

        try {
            // Validate transcription
            if (!transcription || transcription.trim().length < 50) {
                throw new Error('Transcription is too short for meaningful analysis');
            }

            // Prepare analysis request
            const analysisRequest = {
                transcription,
                options: {
                    meetingType,
                    complexity,
                    includeActionItems,
                    includeDecisions,
                    includeSummary,
                    includeTickets,
                    template,
                    priority
                }
            };

            // Try local service only (cloud disabled)
            try {
                const result = await this.analyzeWithLocal(analysisRequest);
                if (result.success) {
                    return {
                        success: true,
                        provider: 'local',
                        ...result
                    };
                }
            } catch (localError) {
                console.error('Local LLM analysis failed:', localError.message);
                throw new Error(`AI analysis failed: ${localError.message}. Cloud services are temporarily disabled.`);
            }

            /* CLOUD FALLBACK TEMPORARILY DISABLED
            // Fallback to cloud service
            const cloudResult = await this.analyzeWithCloud(analysisRequest);
            return {
                success: true,
                provider: 'cloud',
                ...cloudResult
            };
            */

        } catch (error) {
            console.error('AI analysis failed:', error);
            throw new Error(`Meeting analysis failed: ${error.message}`);
        }
    }

    /**
     * Analyze with local LLM service
     */
    async analyzeWithLocal(request) {
        const response = await axios.post(`${this.localLLMUrl}/analyze`, request, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: this.timeout
        });

        return this.formatLocalResponse(response.data);
    }

    /* CLOUD SERVICE TEMPORARILY DISABLED
    // Analyze with cloud LLM service
    async analyzeWithCloud(request) {
        const response = await axios.post(`${this.cloudLLMUrl}/analyze`, request, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: this.timeout
        });

        return this.formatCloudResponse(response.data);
    }
    */ // END CLOUD SERVICE TEMPORARILY DISABLED

    /**
     * Generate specific analysis components
     */
    async generateComponent(transcription, componentType, options = {}) {
        const request = {
            transcription,
            componentType,
            options
        };

        try {
            // Try local service only (cloud disabled)
            try {
                const response = await axios.post(`${this.localLLMUrl}/generate/${componentType}`, request, {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: this.timeout
                });
                return { success: true, provider: 'local', ...response.data };
            } catch (localError) {
                console.error(`Local ${componentType} generation failed:`, localError.message);
                throw new Error(`${componentType} generation failed: ${localError.message}. Cloud services are temporarily disabled.`);
            }

            /* CLOUD FALLBACK TEMPORARILY DISABLED
            // Fallback to cloud
            const response = await axios.post(`${this.cloudLLMUrl}/generate/${componentType}`, request, {
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            
            return { success: true, provider: 'cloud', ...response.data };
            */ // END CLOUD FALLBACK TEMPORARILY DISABLED

        } catch (error) {
            console.error(`${componentType} generation failed:`, error);
            throw new Error(`Failed to generate ${componentType}: ${error.message}`);
        }
    }

    /**
     * Generate action items from transcription
     */
    async generateActionItems(transcription, options = {}) {
        return this.generateComponent(transcription, 'action-items', options);
    }

    /**
     * Generate meeting summary
     */
    async generateSummary(transcription, options = {}) {
        return this.generateComponent(transcription, 'summary', options);
    }

    /**
     * Extract decisions from meeting
     */
    async extractDecisions(transcription, options = {}) {
        return this.generateComponent(transcription, 'decisions', options);
    }

    /**
     * Generate tickets from analysis
     */
    async generateTickets(analysisResult, options = {}) {
        const request = {
            analysis: analysisResult,
            options
        };

        try {
            // Try local service only (cloud disabled)
            try {
                const response = await axios.post(`${this.localLLMUrl}/generate/tickets`, request, {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: this.timeout
                });
                return { success: true, provider: 'local', ...response.data };
            } catch (localError) {
                console.error('Local ticket generation failed:', localError.message);
                throw new Error(`Ticket generation failed: ${localError.message}. Cloud services are temporarily disabled.`);
            }

            /* CLOUD FALLBACK TEMPORARILY DISABLED
            // Fallback to cloud
            const response = await axios.post(`${this.cloudLLMUrl}/generate/tickets`, request, {
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            
            return { success: true, provider: 'cloud', ...response.data };
            */ // END CLOUD FALLBACK TEMPORARILY DISABLED

        } catch (error) {
            console.error('Ticket generation failed:', error);
            throw new Error(`Failed to generate tickets: ${error.message}`);
        }
    }

    /**
     * Check service availability
     */
    async isLocalServiceAvailable() {
        try {
            const response = await axios.get(`${this.localLLMUrl}/health`, { timeout: 5000 });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    /* CLOUD SERVICE TEMPORARILY DISABLED
    async isCloudServiceAvailable() {
        try {
            const response = await axios.get(`${this.cloudLLMUrl}/health`, { timeout: 5000 });
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
                url: this.localLLMUrl,
                preferred: this.preferLocal
            },
            /* CLOUD SERVICE TEMPORARILY DISABLED
            cloud: {
                available: cloudAvailable,
                url: this.cloudLLMUrl
            },
            */ // END CLOUD SERVICE TEMPORARILY DISABLED
            timeout: this.timeout
        };
    }

    /**
     * Get available prompt templates
     */
    async getPromptTemplates() {
        try {
            const templatesData = await fs.readFile(this.promptTemplatesPath, 'utf8');
            return JSON.parse(templatesData);
        } catch (error) {
            console.error('Failed to load prompt templates:', error);
            return { templates: {} };
        }
    }

    /**
     * Validate analysis request
     */
    validateAnalysisRequest(transcription, options) {
        if (!transcription || typeof transcription !== 'string') {
            throw new Error('Valid transcription text is required');
        }

        if (transcription.trim().length < 50) {
            throw new Error('Transcription is too short for meaningful analysis');
        }

        const wordCount = transcription.split(/\s+/).length;
        const maxWords = parseInt(process.env.MAX_ANALYSIS_WORDS) || 50000;
        
        if (wordCount > maxWords) {
            throw new Error(`Transcription is too long (${wordCount} words). Maximum allowed: ${maxWords} words`);
        }

        return true;
    }

    /**
     * Format local service response
     */
    formatLocalResponse(data) {
        return {
            summary: data.summary,
            actionItems: data.action_items || [],
            decisions: data.decisions || [],
            keyPoints: data.key_points || [],
            participants: data.participants || [],
            topics: data.topics || [],
            sentiment: data.sentiment,
            confidence: data.confidence,
            model: data.model_used,
            processingTime: data.processing_time,
            metadata: {
                wordCount: data.word_count,
                participantCount: data.participant_count,
                topicCount: data.topic_count,
                actionItemCount: data.action_items?.length || 0,
                decisionCount: data.decisions?.length || 0
            }
        };
    }

    /* CLOUD SERVICE TEMPORARILY DISABLED
    // Format cloud service response
    formatCloudResponse(data) {
        return {
            summary: data.summary,
            actionItems: data.actionItems || [],
            decisions: data.decisions || [],
            keyPoints: data.keyPoints || [],
            participants: data.participants || [],
            topics: data.topics || [],
            sentiment: data.sentiment,
            confidence: data.confidence,
            provider: data.provider,
            cost: data.cost,
            metadata: {
                wordCount: data.metadata?.wordCount || 0,
                participantCount: data.metadata?.participantCount || 0,
                topicCount: data.metadata?.topicCount || 0,
                actionItemCount: data.actionItems?.length || 0,
                decisionCount: data.decisions?.length || 0,
                estimatedCost: data.cost
            }
        };
    }
    */ // END CLOUD SERVICE TEMPORARILY DISABLED

    /**
     * Process analysis in chunks for large transcriptions
     */
    async processLargeTranscription(transcription, options = {}) {
        const maxChunkSize = parseInt(process.env.MAX_CHUNK_SIZE) || 4000;
        const words = transcription.split(/\s+/);
        
        if (words.length <= maxChunkSize) {
            return this.analyzeMeeting(transcription, options);
        }

        // Split into chunks with overlap
        const chunks = [];
        const overlapSize = Math.floor(maxChunkSize * 0.1); // 10% overlap
        
        for (let i = 0; i < words.length; i += maxChunkSize - overlapSize) {
            const chunk = words.slice(i, i + maxChunkSize).join(' ');
            chunks.push(chunk);
        }

        // Process chunks and merge results
        const results = await Promise.all(
            chunks.map((chunk, index) => 
                this.analyzeMeeting(chunk, { ...options, chunkIndex: index })
            )
        );

        return this.mergeChunkResults(results);
    }

    /**
     * Merge results from multiple chunks
     */
    mergeChunkResults(results) {
        const merged = {
            summary: '',
            actionItems: [],
            decisions: [],
            keyPoints: [],
            participants: new Set(),
            topics: new Set(),
            sentiment: { overall: 'neutral', details: [] },
            confidence: 0,
            metadata: {
                wordCount: 0,
                participantCount: 0,
                topicCount: 0,
                actionItemCount: 0,
                decisionCount: 0,
                chunkCount: results.length
            }
        };

        results.forEach((result, index) => {
            // Merge summaries
            merged.summary += result.summary + '\n\n';
            
            // Merge arrays
            merged.actionItems.push(...(result.actionItems || []));
            merged.decisions.push(...(result.decisions || []));
            merged.keyPoints.push(...(result.keyPoints || []));
            
            // Merge sets
            (result.participants || []).forEach(p => merged.participants.add(p));
            (result.topics || []).forEach(t => merged.topics.add(t));
            
            // Merge sentiment
            if (result.sentiment) {
                merged.sentiment.details.push({
                    chunk: index,
                    sentiment: result.sentiment
                });
            }
            
            // Accumulate metadata
            merged.metadata.wordCount += result.metadata?.wordCount || 0;
            merged.confidence += result.confidence || 0;
        });

        // Finalize merged data
        merged.participants = Array.from(merged.participants);
        merged.topics = Array.from(merged.topics);
        merged.confidence = merged.confidence / results.length;
        merged.metadata.participantCount = merged.participants.length;
        merged.metadata.topicCount = merged.topics.length;
        merged.metadata.actionItemCount = merged.actionItems.length;
        merged.metadata.decisionCount = merged.decisions.length;

        // Calculate overall sentiment
        const sentiments = merged.sentiment.details.map(d => d.sentiment.overall || d.sentiment);
        merged.sentiment.overall = this.calculateOverallSentiment(sentiments);

        return merged;
    }

    /**
     * Calculate overall sentiment from multiple sentiment values
     */
    calculateOverallSentiment(sentiments) {
        const counts = { positive: 0, negative: 0, neutral: 0 };
        
        sentiments.forEach(sentiment => {
            if (typeof sentiment === 'string') {
                counts[sentiment] = (counts[sentiment] || 0) + 1;
            }
        });

        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }
}

module.exports = new AIAnalysisService();