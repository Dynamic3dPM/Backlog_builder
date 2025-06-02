const axios = require('axios');
const fs = require('fs').promises;

class AIAnalysisService {
    constructor() {
        this.localLLMUrl = process.env.LOCAL_LLM_URL || 'http://huggingface-llm:8002';
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
        console.log('=== Starting analyzeMeeting ===');
        console.log('Input transcription type:', typeof transcription);
        console.log('Input transcription value:', transcription);
        
        // Debug the actual object structure
        if (transcription && typeof transcription === 'object') {
            console.log('Transcription object keys:', Object.keys(transcription));
            console.log('Transcription object content:', JSON.stringify(transcription, null, 2));
        }
        
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
            console.log('Validating transcription...');
            
            // Handle null/undefined
            if (transcription === null || transcription === undefined) {
                throw new Error('Transcription is null or undefined');
            }
            
            // Convert to string safely
            let transcriptStr;
            if (typeof transcription === 'string') {
                transcriptStr = transcription;
            } else if (transcription && typeof transcription.text === 'string') {
                // Handle case where transcription is an object with a text property
                transcriptStr = transcription.text;
            } else if (typeof transcription.toString === 'function') {
                // Try to convert to string if possible
                transcriptStr = transcription.toString();
            } else {
                // Last resort - JSON stringify
                transcriptStr = JSON.stringify(transcription);
            }
            
            console.log('Transcription after conversion:', {
                type: typeof transcriptStr,
                length: transcriptStr?.length,
                first50: transcriptStr?.substring?.(0, 50) || 'N/A',
                isString: typeof transcriptStr === 'string'
            });
            
            // Ensure we have a valid string
            if (!transcriptStr || typeof transcriptStr !== 'string') {
                throw new Error(`Invalid transcription type: ${typeof transcriptStr}`);
            }
            
            const trimmed = transcriptStr.trim();
            if (!trimmed) {
                throw new Error('Transcription is empty after trimming');
            }
            
            if (trimmed.length < 50) {
                throw new Error(`Transcription is too short (${trimmed.length} chars) for meaningful analysis. Minimum 50 characters required.`);
            }

            // Prepare analysis request with the processed transcript string
            const analysisRequest = {
                transcription: trimmed, // Use the processed and trimmed string
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
        // Convert backend request format to LLM service format
        const llmRequest = {
            transcript: request.transcription,
            meeting_type: request.options?.meetingType || 'general',
            language: request.options?.language || 'en',
            extract_action_items: request.options?.includeActionItems !== false,
            extract_decisions: request.options?.includeDecisions !== false,
            generate_summary: request.options?.includeSummary !== false,
            detect_sentiment: request.options?.detectSentiment || false,
            identify_speakers: request.options?.identifySpeakers || false
        };

        const response = await axios.post(`${this.localLLMUrl}/analyze`, llmRequest, {
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
        try {
            // Prepare request for the /analyze endpoint
            const analysisRequest = {
                transcript: transcription,
                meeting_type: options.meetingType || 'general',
                language: options.language || 'en',
                extract_action_items: componentType === 'action-items',
                extract_decisions: componentType === 'decisions',
                generate_summary: componentType === 'summary',
                detect_sentiment: false,
                identify_speakers: false
            };

            // Try local service only (cloud disabled)
            try {
                const response = await axios.post(`${this.localLLMUrl}/analyze`, analysisRequest, {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: this.timeout
                });

                // Extract the specific component from the response
                const analysisResult = response.data;
                let componentResult;

                switch (componentType) {
                    case 'action-items':
                        componentResult = analysisResult.action_items || [];
                        break;
                    case 'summary':
                        componentResult = analysisResult.summary || null;
                        break;
                    case 'decisions':
                        componentResult = analysisResult.decisions || [];
                        break;
                    default:
                        throw new Error(`Unknown component type: ${componentType}`);
                }

                return { 
                    success: true, 
                    provider: 'local', 
                    [componentType.replace('-', '_')]: componentResult,
                    processing_time: analysisResult.processing_time
                };
            } catch (localError) {
                console.error(`Local ${componentType} generation failed:`, localError.message);
                throw new Error(`${componentType} generation failed: ${localError.message}. Cloud services are temporarily disabled.`);
            }

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
            const response = await fetch(`${this.localLLMUrl}/health`, { timeout: 5000 });
            return response.ok;
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
        console.log('Validating transcription:', {
            type: typeof transcription,
            isString: typeof transcription === 'string',
            length: transcription?.length,
            value: typeof transcription === 'string' ? transcription.substring(0, 100) + (transcription.length > 100 ? '...' : '') : transcription
        });

        if (!transcription) {
            throw new Error('Transcription text is required');
        }

        // Ensure transcription is a string
        const transcriptText = String(transcription || '');

        if (transcriptText.trim().length < 50) {
            throw new Error(`Transcription is too short (${transcriptText.trim().length} chars) for meaningful analysis. Minimum 50 characters required.`);
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
        // Handle both snake_case and camelCase response formats
        const actionItems = data.action_items || data.actionItems || [];
        const keyPoints = data.key_points || data.keyPoints || [];
        const modelUsed = data.model_used || data.modelUsed || 'unknown';
        const processingTime = data.processing_time || data.processingTime || 0;
        const wordCount = data.word_count || (data.metadata?.wordCount || 0);
        const participantCount = data.participant_count || (data.metadata?.participantCount || 0);
        const topicCount = data.topic_count || (data.metadata?.topicCount || 0);
        
        return {
            success: true,
            summary: data.summary || {},
            actionItems: actionItems,
            decisions: data.decisions || [],
            keyPoints: keyPoints,
            participants: data.participants || [],
            topics: data.topics || [],
            sentiment: data.sentiment || null,
            confidence: data.confidence || 0,
            model: modelUsed,
            processingTime: processingTime,
            metadata: {
                wordCount: wordCount,
                participantCount: participantCount,
                topicCount: topicCount,
                actionItemCount: actionItems.length,
                decisionCount: (data.decisions || []).length
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

    /**
     * Smart conversation analysis with intelligent ticket generation
     * @param {string} transcript - The conversation transcript
     * @returns {Object} Smart analysis results with conversation insights and intelligent tickets
     */
    async analyzeConversationSmart(transcript) {
        try {
            console.log('Calling smart conversation analysis endpoint');
            
            const response = await fetch(`${this.localLLMUrl}/analyze-conversation-smart`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transcript: transcript
                }),
                timeout: this.timeout
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Smart analysis failed: ${response.status} ${errorText}`);
            }

            const result = await response.json();
            
            console.log('Smart conversation analysis completed:', {
                success: result.success,
                insights_found: result.conversation_analysis?.insights_found,
                tickets_generated: result.intelligent_tickets?.length
            });

            return result;

        } catch (error) {
            console.error('Smart conversation analysis error:', error);
            throw new Error(`Smart conversation analysis failed: ${error.message}`);
        }
    }
}

module.exports = new AIAnalysisService();