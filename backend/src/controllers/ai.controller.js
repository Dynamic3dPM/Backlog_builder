/**
 * Backlog Builder AI Controller
 * Orchestrates AI services for meeting processing and ticket generation
 */

const speechToTextService = require('../services/speechToText');
const aiAnalysisService = require('../services/aiAnalysis');
const ticketGeneratorService = require('../services/ticketGenerator');
const logger = require('../utils/logger');
const { validateFileUpload } = require('../middleware/fileValidation');
const fs = require('fs'); // Add fs import for cleanup and path handling

class AIController {
    /**
     * Process uploaded audio file through complete AI pipeline (alias for processAudioFile)
     */
    async processAudioComplete(req, res) {
        return this.processAudioFile(req, res);
    }

    /**
     * Process uploaded audio file through complete AI pipeline
     */
    async processAudioFile(req, res) {
        const startTime = Date.now();
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const { file } = req;
        
        try {
            const { 
                language = 'auto',
                meetingType = 'general',
                projectId,
                generateTickets = true,
                priority = 'balanced',
                enableDiarization = true,
                model = 'small'
            } = req.body;

            if (!file) {
                return res.status(400).json({
                    error: 'No audio file provided',
                    code: 'MISSING_FILE',
                    jobId
                });
            }

            logger.info(`Starting AI pipeline for file: ${file.originalname}`, { jobId });

            // Setup WebSocket connection if available
            const wsClient = req.app.get('wsClient');
            const sendProgress = (progress, message = '') => {
                const progressData = {
                    jobId,
                    type: 'progress',
                    progress: Math.min(100, Math.max(0, progress)),
                    message,
                    timestamp: new Date().toISOString()
                };
                
                if (wsClient) {
                    wsClient.emit('progress', progressData);
                }
                
                return progressData;
            };

            sendProgress(5, 'Starting audio processing...');

            // Step 1: Speech-to-Text with progress tracking
            let transcriptionResult;
            try {
                console.log('=== STARTING SPEECH-TO-TEXT PROCESSING ===');
                console.log('Audio file path:', file.path);
                console.log('Language:', language);
                console.log('Model:', model);
                
                transcriptionResult = await speechToTextService.processAudio(
                    file.path,
                    {
                        language,
                        model,
                        speakerDiarization: enableDiarization,
                        enableProgress: true,
                        priority,
                        onProgress: (progressData) => {
                            // Map progress from 0-80% for STT (leaving 20% for AI analysis)
                            const mappedProgress = 5 + (progressData.progress * 0.75);
                            sendProgress(mappedProgress, `Transcribing: ${progressData.message || ''}`);
                        }
                    }
                );
                
                console.log('=== SPEECH-TO-TEXT RESULT ===');
                console.log('Result type:', typeof transcriptionResult);
                console.log('Result keys:', Object.keys(transcriptionResult));
                console.log('Result content:', JSON.stringify(transcriptionResult, null, 2));
                
                if (!transcriptionResult || !transcriptionResult.success) {
                    throw new Error(transcriptionResult?.error || 'Unknown error during transcription');
                }
                
                sendProgress(80, 'Transcription complete. Analyzing content...');
                
            } catch (error) {
                logger.error('Speech-to-text processing failed', { 
                    error: error.message, 
                    stack: error.stack,
                    jobId,
                    file: file.originalname 
                });
                
                sendProgress(0, 'Transcription failed');
                
                return res.status(500).json({
                    success: false,
                    error: 'Speech-to-text processing failed',
                    message: error.message,
                    code: 'STT_FAILED',
                    jobId
                });
            }

            // Step 2: AI Analysis
            let analysisResult;
            try {
                // Debug transcription content and type
                logger.debug('=== RAW TRANSCRIPTION RESULT ===', {
                    resultKeys: Object.keys(transcriptionResult),
                    textType: typeof transcriptionResult.text,
                    textValue: typeof transcriptionResult.text === 'string' ? 
                        transcriptionResult.text.substring(0, 100) + (transcriptionResult.text.length > 100 ? '...' : '') : 
                        transcriptionResult.text,
                    hasSegments: Array.isArray(transcriptionResult.segments),
                    segmentCount: Array.isArray(transcriptionResult.segments) ? transcriptionResult.segments.length : 0,
                    firstSegment: Array.isArray(transcriptionResult.segments) && transcriptionResult.segments[0] ? 
                        JSON.stringify(transcriptionResult.segments[0]) : 'No segments',
                    resultString: JSON.stringify(transcriptionResult).substring(0, 200) + '...',
                    // Additional debug info
                    hasTranscript: 'transcript' in transcriptionResult,
                    transcriptionType: typeof transcriptionResult.transcription,
                    hasTranscription: 'transcription' in transcriptionResult,
                    transcriptionValue: typeof transcriptionResult.transcription === 'string' ?
                        transcriptionResult.transcription.substring(0, 100) + '...' : 'N/A'
                });

                // Coerce transcription to string with fallbacks
                let transcriptText = '';
                try {
                    // Try to get text from the most common fields
                    if (transcriptionResult?.text) {
                        // If text exists, use it regardless of type
                        transcriptText = String(transcriptionResult.text);
                    } else if (transcriptionResult?.transcription) {
                        // Try the transcription field if text isn't available
                        transcriptText = String(transcriptionResult.transcription);
                    } else if (transcriptionResult?.transcript) {
                        // Another common field name for transcript
                        transcriptText = String(transcriptionResult.transcript);
                    } else if (Array.isArray(transcriptionResult?.segments)) {
                        // Fallback to joining segments if text isn't available
                        transcriptText = transcriptionResult.segments
                            .filter(seg => seg?.text)
                            .map(seg => String(seg.text))
                            .join(' ');
                    }
                } catch (error) {
                    logger.error('Error processing transcription result:', {
                        error: error.message,
                        transcriptionResult: JSON.stringify(transcriptionResult).substring(0, 200) + '...',
                        jobId
                    });
                    throw new Error(`Failed to process transcription: ${error.message}`);
                }

                // Ensure we have valid text
                if (!transcriptText || typeof transcriptText !== 'string') {
                    const errorMsg = 'Failed to extract valid text from transcription result';
                    logger.error(errorMsg, { transcriptionResult: JSON.stringify(transcriptionResult) });
                    throw new Error(errorMsg);
                }

                logger.debug('Transcript text before analysis:', {
                    type: typeof transcriptText,
                    length: transcriptText.length,
                    first100: transcriptText.substring(0, 100) + (transcriptText.length > 100 ? '...' : '')
                });

                // Ensure text is properly formatted for analysis
                transcriptText = transcriptText.trim();
                if (transcriptText.length < 50) {
                    throw new Error(`Transcription is too short (${transcriptText.length} chars). Minimum 50 characters required.`);
                }

                analysisResult = await aiAnalysisService.analyzeMeeting(
                    transcriptText,
                    {
                        meetingType,
                        language: transcriptionResult.language || language,
                        extractActionItems: true,
                        extractDecisions: true,
                        generateSummary: true,
                        detectSentiment: true
                    }
                );
                
                if (!analysisResult.success) {
                    throw new Error(analysisResult.error || 'Unknown error during analysis');
                }
                
                sendProgress(95, 'Analysis complete. Finalizing results...');
                
            } catch (error) {
                logger.error('AI analysis failed', { 
                    error: error.message, 
                    stack: error.stack,
                    jobId,
                    file: file.originalname 
                });
                
                sendProgress(0, 'Analysis failed');
                
                return res.status(500).json({
                    success: false,
                    error: 'AI analysis failed',
                    message: error.message,
                    code: 'ANALYSIS_FAILED',
                    jobId
                });
            }

            // Step 3: Generate Tickets (if requested)
            let ticketResults = null;
            if (generateTickets && analysisResult.actionItems?.length > 0) {
                try {
                    ticketResults = await ticketGeneratorService.generateTicketsFromActionItems(
                        analysisResult.actionItems,
                        {
                            projectId,
                            context: analysisResult.summary?.executive_summary,
                            priority
                        }
                    );
                    
                    if (ticketResults && !ticketResults.success) {
                        logger.warn('Ticket generation completed with warnings', {
                            jobId,
                            warnings: ticketResults.warnings
                        });
                    }
                    
                } catch (error) {
                    logger.error('Ticket generation failed', {
                        error: error.message,
                        stack: error.stack,
                        jobId
                    });
                    // Continue with the response even if ticket generation fails
                }
            }

            // Compile final response
            const processingTime = Date.now() - startTime;
            const response = {
                success: true,
                jobId,
                processing_time: processingTime,
                data: {
                    transcription: {
                        text: transcriptionResult.text,
                        language: transcriptionResult.language,
                        segments: transcriptionResult.segments || [],
                        speakers: transcriptionResult.speakers || [],
                        processing_time: transcriptionResult.processing_time
                    },
                    analysis: analysisResult,
                    tickets: ticketResults?.tickets || null
                },
                metadata: {
                    file_name: file.originalname,
                    file_size: file.size,
                    meeting_type: meetingType,
                    project_id: projectId,
                    model_used: transcriptionResult.metadata?.model,
                    language_detected: transcriptionResult.language
                },
                warnings: ticketResults?.warnings || []
            };

            // Send completion event
            sendProgress(100, 'Processing complete');
            
            if (wsClient) {
                wsClient.emit('complete', {
                    jobId,
                    type: 'complete',
                    timestamp: new Date().toISOString(),
                    result: response
                });
            }

            logger.info(`AI pipeline completed successfully for ${file.originalname}`, {
                jobId,
                processingTime: `${processingTime}ms`,
                fileSize: file.size,
                language: response.data.transcription.language
            });
            
            res.json(response);

        } catch (error) {
            logger.error('AI pipeline failed', {
                error: error.message,
                stack: error.stack,
                jobId,
                file: file?.originalname
            });
            
            // Send error event
            const errorData = {
                jobId,
                type: 'error',
                error: 'AI pipeline failed',
                message: error.message,
                code: 'PIPELINE_ERROR',
                timestamp: new Date().toISOString()
            };
            
            if (req.app.get('wsClient')) {
                req.app.get('wsClient').emit('error', errorData);
            }
            
            // Cleanup on error
            try {
                if (file?.path && fs.existsSync(file.path)) {
                    await fs.promises.unlink(file.path);
                }
            } catch (cleanupError) {
                logger.warn('Failed to clean up file', {
                    file: file?.path,
                    error: cleanupError.message,
                    jobId
                });
            }

            res.status(500).json({
                success: false,
                ...errorData
            });
        }
    }

    /**
     * Process text transcript directly (bypass STT)
     */
    async processTranscript(req, res) {
        const startTime = Date.now();
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        try {
            const {
                transcript,
                meetingType = 'general',
                language = 'en',
                projectId,
                generateTickets = true,
                priority = 'medium'
            } = req.body;

            if (!transcript || transcript.trim().length === 0) {
                return res.status(400).json({
                    success: false,
                    error: 'Transcript text is required',
                    code: 'MISSING_TRANSCRIPT',
                    jobId
                });
            }

            logger.info('Starting AI analysis for provided transcript', { jobId });

            // Setup WebSocket connection if available
            const wsClient = req.app.get('wsClient');
            const sendProgress = (progress, message = '') => {
                const progressData = {
                    jobId,
                    type: 'progress',
                    progress: Math.min(100, Math.max(0, progress)),
                    message,
                    timestamp: new Date().toISOString()
                };
                
                if (wsClient) {
                    wsClient.emit('progress', progressData);
                }
                
                return progressData;
            };
            
            sendProgress(10, 'Starting analysis...');

            // AI Analysis
            const analysisResult = await aiAnalysisService.analyzeMeeting({
                transcript,
                meetingType,
                language,
                extractActionItems: true,
                extractDecisions: true,
                generateSummary: true,
                detectSentiment: true
            });

            if (!analysisResult.success) {
                throw new Error(analysisResult.error || 'AI analysis failed');
            }
            
            sendProgress(70, 'Analysis complete. Generating tickets...');

            // Generate Tickets (if requested)
            let ticketResults = null;
            if (generateTickets && analysisResult.actionItems?.length > 0) {
                try {
                    ticketResults = await ticketGeneratorService.generateTicketsFromActionItems(
                        analysisResult.actionItems,
                        {
                            projectId,
                            context: analysisResult.summary?.executive_summary,
                            priority
                        }
                    );
                } catch (error) {
                    logger.error('Ticket generation failed', {
                        error: error.message,
                        stack: error.stack,
                        jobId
                    });
                    // Continue with the response even if ticket generation fails
                }
            }
            
            sendProgress(95, 'Finalizing results...');

            const processingTime = Date.now() - startTime;
            const response = {
                success: true,
                jobId,
                processing_time: processingTime,
                data: {
                    actionItems: analysisResult.actionItems || [],
                    decisions: analysisResult.decisions || [],
                    summary: analysisResult.summary || {},
                    keyPoints: analysisResult.keyPoints || [],
                    participants: analysisResult.participants || [],
                    topics: analysisResult.topics || [],
                    sentiment: analysisResult.sentiment || null,
                    confidence: analysisResult.confidence || 0
                },
                metadata: {
                    meeting_type: meetingType,
                    project_id: projectId,
                    language
                },
                warnings: ticketResults?.warnings || []
            };

            // Send completion event
            sendProgress(100, 'Processing complete');
            if (wsClient) {
                wsClient.emit('complete', {
                    jobId,
                    type: 'complete',
                    timestamp: new Date().toISOString(),
                    result: response
                });
            }

            logger.info(`Transcript processing completed successfully`, {
                jobId,
                processingTime: `${processingTime}ms`,
                meetingType,
                language
            });

            return res.json(response);
        } catch (error) {
            logger.error('Transcript analysis error:', error);
            res.status(500).json({
                error: 'Transcript analysis failed',
                message: error.message,
                code: 'ANALYSIS_ERROR'
            });
        }
    }

    /**
     * Get AI service health status
     */
    async getServiceHealth(req, res) {
        try {
            const healthChecks = await Promise.allSettled([
                speechToTextService.healthCheck(),
                aiAnalysisService.healthCheck()
            ]);

            const sttHealth = healthChecks[0];
            const analysisHealth = healthChecks[1];

            const overallHealth = sttHealth.status === 'fulfilled' && 
                                analysisHealth.status === 'fulfilled' &&
                                sttHealth.value.status === 'healthy' &&
                                analysisHealth.value.status === 'healthy';

            res.json({
                status: overallHealth ? 'healthy' : 'degraded',
                services: {
                    speech_to_text: sttHealth.status === 'fulfilled' ? sttHealth.value : { status: 'error', error: sttHealth.reason?.message },
                    ai_analysis: analysisHealth.status === 'fulfilled' ? analysisHealth.value : { status: 'error', error: analysisHealth.reason?.message }
                },
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            logger.error('Health check error:', error);
            res.status(500).json({
                status: 'error',
                message: error.message
            });
        }
    }

    /**
     * Generate tickets from existing action items or analysis
     */
    async generateTickets(req, res) {
        try {
            const { 
                actionItems, 
                analysis, 
                projectId, 
                context, 
                project, 
                template 
            } = req.body;

            let items = actionItems;
            let analysisData = analysis;
            let projectIdentifier = projectId || project;
            
            // Handle nested analysis format (e.g., from /api/ai/analyze response)
            if (analysis && !actionItems) {
                // Handle both direct analysis format and nested data format
                if (analysis.data) {
                    items = analysis.data.actionItems;
                    analysisData = analysis.data;
                } else {
                    items = analysis.actionItems;
                    analysisData = analysis;
                }
            }

            // If using full analysis object, use the comprehensive ticket generator
            if (analysisData && (analysisData.actionItems || analysisData.decisions || analysisData.summary)) {
                const ticketResults = await ticketGeneratorService.generateTickets(analysisData, {
                    projectId: projectIdentifier,
                    templateId: template,
                    context: context || analysisData.summary
                });

                return res.json({
                    success: ticketResults.success,
                    tickets: ticketResults.tickets,
                    summary: ticketResults.summary,
                    metadata: ticketResults.metadata,
                    processing_time: Date.now() - req.startTime
                });
            }

            // Handle simple action items array format
            if (!Array.isArray(items) || items.length === 0) {
                return res.status(400).json({
                    error: 'Action items array is required',
                    code: 'MISSING_ACTION_ITEMS'
                });
            }

            const ticketResults = await ticketGeneratorService.generateTicketsFromActionItems(
                items,
                { projectId: projectIdentifier, context }
            );

            if (!ticketResults.success) {
                return res.status(500).json({
                    error: 'Ticket generation failed',
                    details: ticketResults.error,
                    code: 'TICKET_GENERATION_FAILED'
                });
            }

            res.json({
                success: true,
                tickets: ticketResults.tickets,
                processing_time: ticketResults.processing_time
            });

        } catch (error) {
            logger.error('Ticket generation error:', error);
            res.status(500).json({
                error: 'Ticket generation failed',
                message: error.message,
                code: 'TICKET_ERROR'
            });
        }
    }

    /**
     * Get AI service usage statistics
     */
    async getUsageStats(req, res) {
        try {
            const [sttStats, analysisStats] = await Promise.allSettled([
                speechToTextService.getUsageStats(),
                aiAnalysisService.getUsageStats()
            ]);

            res.json({
                speech_to_text: sttStats.status === 'fulfilled' ? sttStats.value : null,
                ai_analysis: analysisStats.status === 'fulfilled' ? analysisStats.value : null,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            logger.error('Usage stats error:', error);
            res.status(500).json({
                error: 'Failed to retrieve usage statistics',
                message: error.message
            });
        }
    }

    /**
     * WebSocket endpoint for real-time transcription updates
     */
    async transcribeWebSocket(req, res) {
        if (!req.ws) {
            return res.status(400).json({
                success: false,
                error: 'WebSocket connection required'
            });
        }

        const { jobId } = req.params;
        const ws = req.ws;

        try {
            // Set up WebSocket connection
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    
                    if (data.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong' }));
                        return;
                    }
                    
                    // Handle other message types as needed
                    
                } catch (error) {
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: 'Invalid message format'
                    }));
                }
            });

            // Start progress monitoring
            const progressInterval = setInterval(async () => {
                try {
                    const progress = await speechToTextService.getProgress(jobId);
                    
                    ws.send(JSON.stringify({
                        type: 'progress',
                        jobId,
                        progress: progress.progress,
                        status: progress.status,
                        message: progress.message
                    }));

                    // Stop monitoring when complete
                    if (progress.status === 'completed' || progress.status === 'failed') {
                        clearInterval(progressInterval);
                        
                        ws.send(JSON.stringify({
                            type: 'complete',
                            jobId,
                            result: progress.result
                        }));
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: error.message
                    }));
                }
            }, 2000); // Update every 2 seconds

            // Clean up on connection close
            ws.on('close', () => {
                clearInterval(progressInterval);
            });

        } catch (error) {
            logger.error('WebSocket transcription error:', error);
            ws.send(JSON.stringify({
                type: 'error',
                error: 'WebSocket connection failed'
            }));
        }
    }

    /**
     * WebSocket endpoint for real-time progress updates
     */
    async progressWebSocket(req, res) {
        if (!req.ws) {
            return res.status(400).json({
                success: false,
                error: 'WebSocket connection required'
            });
        }

        const { jobId } = req.params;
        const ws = req.ws;

        try {
            // Set up progress monitoring
            const progressInterval = setInterval(async () => {
                try {
                    // Get progress from both STT and LLM services
                    const [sttProgress, llmProgress] = await Promise.all([
                        speechToTextService.getProgress(jobId).catch(() => ({ progress: 0, status: 'unknown' })),
                        aiAnalysisService.getProgress?.(jobId).catch(() => ({ progress: 0, status: 'unknown' }))
                    ]);

                    const overallProgress = {
                        jobId,
                        stt: sttProgress,
                        llm: llmProgress,
                        overall: Math.max(sttProgress.progress || 0, llmProgress.progress || 0),
                        timestamp: new Date().toISOString()
                    };

                    ws.send(JSON.stringify({
                        type: 'progress_update',
                        data: overallProgress
                    }));

                    // Stop monitoring when all complete
                    if ((sttProgress.status === 'completed' || sttProgress.status === 'failed') &&
                        (llmProgress.status === 'completed' || llmProgress.status === 'failed')) {
                        clearInterval(progressInterval);
                        
                        ws.send(JSON.stringify({
                            type: 'complete',
                            data: overallProgress
                        }));
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: error.message
                    }));
                }
            }, 1000); // Update every second

            // Handle WebSocket messages
            ws.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    
                    if (data.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong' }));
                    }
                } catch (error) {
                    // Ignore invalid messages
                }
            });

            // Clean up on connection close
            ws.on('close', () => {
                clearInterval(progressInterval);
            });

        } catch (error) {
            logger.error('WebSocket progress error:', error);
            ws.send(JSON.stringify({
                type: 'error',
                error: 'WebSocket connection failed'
            }));
        }
    }

    /**
     * Get available prompt templates
     */
    async getPromptTemplates(req, res) {
        try {
            const templates = await aiAnalysisService.getPromptTemplates();
            
            res.json({
                success: true,
                templates
            });
        } catch (error) {
            logger.error('Error getting prompt templates:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get prompt templates'
            });
        }
    }

    /**
     * Update prompt template
     */
    async updatePromptTemplate(req, res) {
        try {
            const { templateId, template } = req.body;
            
            if (!templateId || !template) {
                return res.status(400).json({
                    success: false,
                    error: 'Template ID and template content required'
                });
            }

            // This would typically save to a database or file
            // For now, we'll just validate the template structure
            const requiredFields = ['name', 'description', 'prompt', 'category'];
            const missingFields = requiredFields.filter(field => !template[field]);
            
            if (missingFields.length > 0) {
                return res.status(400).json({
                    success: false,
                    error: `Missing required fields: ${missingFields.join(', ')}`
                });
            }

            res.json({
                success: true,
                message: 'Template updated successfully',
                templateId
            });
        } catch (error) {
            logger.error('Error updating prompt template:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to update prompt template'
            });
        }
    }

    /**
     * Get available AI providers and their capabilities
     */
    async getProviders(req, res) {
        try {
            const [sttStatus, llmStatus] = await Promise.all([
                speechToTextService.getServiceStatus(),
                aiAnalysisService.getServiceStatus()
            ]);

            const providers = {
                speechToText: {
                    local: sttStatus.local,
                    cloud: sttStatus.cloud,
                    capabilities: {
                        languages: ['auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                        formats: ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'],
                        maxFileSize: '500MB',
                        speakerDiarization: true,
                        realTimeProcessing: true
                    }
                },
                llm: {
                    local: llmStatus.local,
                    cloud: llmStatus.cloud,
                    capabilities: {
                        analysisTypes: ['summary', 'action-items', 'decisions', 'tickets'],
                        meetingTypes: ['general', 'development', 'planning', 'review', 'standup'],
                        complexityLevels: ['simple', 'medium', 'complex'],
                        maxWords: 50000
                    }
                }
            };

            res.json({
                success: true,
                providers
            });
        } catch (error) {
            logger.error('Error getting providers:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get provider information'
            });
        }
    }

    /**
     * Test connectivity to AI providers
     */
    async testProviders(req, res) {
        try {
            const { providers } = req.body;
            const results = {};

            // Test STT providers
            if (!providers || providers.includes('stt')) {
                results.stt = {
                    local: await speechToTextService.isLocalServiceAvailable(),
                    cloud: await speechToTextService.isCloudServiceAvailable()
                };
            }

            // Test LLM providers
            if (!providers || providers.includes('llm')) {
                results.llm = {
                    local: await aiAnalysisService.isLocalServiceAvailable(),
                    cloud: await aiAnalysisService.isCloudServiceAvailable()
                };
            }

            res.json({
                success: true,
                results,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            logger.error('Error testing providers:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to test providers'
            });
        }
    }

    /**
     * Process multiple audio files in batch
     */
    async processBatch(req, res) {
        try {
            const { files } = req;
            const options = req.body;

            if (!files || files.length === 0) {
                return res.status(400).json({
                    success: false,
                    error: 'No audio files provided'
                });
            }

            const batchId = 'batch-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            
            // Start batch processing asynchronously
            this.processBatchAsync(batchId, files, options);

            res.json({
                success: true,
                batchId,
                fileCount: files.length,
                status: 'processing',
                message: 'Batch processing started'
            });
        } catch (error) {
            logger.error('Error starting batch processing:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to start batch processing'
            });
        }
    }

    /**
     * Get batch processing status
     */
    async getBatchStatus(req, res) {
        try {
            const { batchId } = req.params;
            
            // This would typically query a database or cache
            // For now, return a mock status
            res.json({
                success: true,
                batchId,
                status: 'processing',
                progress: 45,
                completed: 2,
                total: 5,
                results: []
            });
        } catch (error) {
            logger.error('Error getting batch status:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get batch status'
            });
        }
    }

    /**
     * Get AI service usage analytics
     */
    async getUsageAnalytics(req, res) {
        try {
            const { timeRange = '24h' } = req.query;
            
            // Mock analytics data - would come from actual metrics store
            const analytics = {
                timeRange,
                requests: {
                    total: 156,
                    successful: 142,
                    failed: 14,
                    successRate: 91.0
                },
                processing: {
                    averageTime: 45.6,
                    totalProcessingTime: 7123.2,
                    audioHoursProcessed: 12.5
                },
                services: {
                    stt: {
                        requests: 78,
                        averageTime: 23.4
                    },
                    llm: {
                        requests: 78,
                        averageTime: 22.2
                    }
                },
                providers: {
                    local: 45,
                    cloud: 111
                }
            };

            res.json({
                success: true,
                analytics,
                generatedAt: new Date().toISOString()
            });
        } catch (error) {
            logger.error('Error getting usage analytics:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get usage analytics'
            });
        }
    }

    /**
     * Get AI service cost analytics
     */
    async getCostAnalytics(req, res) {
        try {
            const { timeRange = '24h' } = req.query;
            
            // Mock cost data - would come from actual cost tracking
            const costs = {
                timeRange,
                total: 15.67,
                breakdown: {
                    stt: 8.23,
                    llm: 7.44
                },
                providers: {
                    openai: 9.12,
                    anthropic: 3.45,
                    google: 2.10,
                    azure: 1.00
                },
                budget: {
                    daily: 100.00,
                    used: 15.67,
                    remaining: 84.33,
                    percentage: 15.67
                }
            };

            res.json({
                success: true,
                costs,
                generatedAt: new Date().toISOString()
            });
        } catch (error) {
            logger.error('Error getting cost analytics:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get cost analytics'
            });
        }
    }

    /**
     * Get AI service performance metrics
     */
    async getPerformanceAnalytics(req, res) {
        try {
            const { timeRange = '24h' } = req.query;
            
            // Mock performance data
            const performance = {
                timeRange,
                averageResponseTime: 45.6,
                p95ResponseTime: 123.4,
                p99ResponseTime: 234.5,
                throughput: 2.3,
                errorRate: 0.09,
                availability: 99.2,
                services: {
                    stt: {
                        averageResponseTime: 23.4,
                        availability: 99.5
                    },
                    llm: {
                        averageResponseTime: 22.2,
                        availability: 98.9
                    }
                }
            };

            res.json({
                success: true,
                performance,
                generatedAt: new Date().toISOString()
            });
        } catch (error) {
            logger.error('Error getting performance analytics:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get performance analytics'
            });
        }
    }

    /**
     * Transcribe audio file only (STT service only)
     */
    async transcribeAudio(req, res) {
        try {
            const { file } = req;
            const { language = 'en-US' } = req.body;

            if (!file) {
                return res.status(400).json({
                    error: 'No audio file provided',
                    code: 'MISSING_FILE'
                });
            }

            logger.info(`Starting transcription for file: ${file.originalname}`);

            const transcriptionResult = await speechToTextService.transcribeAudio(file.path, {
                language,
                useLocal: true,
                enableDiarization: true
            });

            if (!transcriptionResult.success) {
                return res.status(500).json({
                    error: 'Speech-to-text processing failed',
                    details: transcriptionResult.error,
                    code: 'STT_FAILED'
                });
            }

            // Cleanup uploaded file
            await speechToTextService.cleanupFile(file.path);

            res.json({
                success: true,
                data: {
                    transcription: {
                        text: transcriptionResult.transcript,
                        language: transcriptionResult.language,
                        confidence: transcriptionResult.confidence,
                        segments: transcriptionResult.segments,
                        speakers: transcriptionResult.speakers
                    }
                },
                metadata: {
                    file_name: file.originalname,
                    file_size: file.size,
                    processing_time: Date.now() - req.startTime
                }
            });

        } catch (error) {
            logger.error('Transcription error:', error);
            
            if (req.file?.path) {
                await speechToTextService.cleanupFile(req.file.path);
            }

            res.status(500).json({
                error: 'Transcription failed',
                message: error.message,
                code: 'TRANSCRIPTION_ERROR'
            });
        }
    }

    /**
     * Analyze meeting transcription only (LLM service only)
     */
    async analyzeMeeting(req, res) {
        try {
            const { transcription, meetingType = 'general', language = 'en' } = req.body;

            if (!transcription || transcription.trim().length === 0) {
                return res.status(400).json({
                    error: 'Transcription text is required',
                    code: 'MISSING_TRANSCRIPTION'
                });
            }

            logger.info('Starting AI analysis for transcription');

            const analysisResult = await aiAnalysisService.analyzeMeeting(transcription, {
                meetingType,
                language,
                extractActionItems: true,
                extractDecisions: true,
                generateSummary: true,
                detectSentiment: true
            });

            if (!analysisResult.success) {
                return res.status(500).json({
                    error: 'AI analysis failed',
                    details: analysisResult.error,
                    code: 'ANALYSIS_FAILED'
                });
            }

            res.json({
                success: true,
                data: {
                    actionItems: analysisResult.actionItems || [],
                    decisions: analysisResult.decisions || [],
                    summary: analysisResult.summary || {},
                    keyPoints: analysisResult.keyPoints || [],
                    participants: analysisResult.participants || [],
                    topics: analysisResult.topics || [],
                    sentiment: analysisResult.sentiment || null,
                    confidence: analysisResult.confidence || 0
                },
                metadata: {
                    transcription_length: transcription.length,
                    meeting_type: meetingType,
                    processing_time: Date.now() - req.startTime
                }
            });

        } catch (error) {
            logger.error('Analysis error:', error);
            res.status(500).json({
                error: 'Analysis failed',
                message: error.message,
                code: 'ANALYSIS_ERROR'
            });
        }
    }

    /**
     * Get processing progress for a specific job
     */
    async getProgress(req, res) {
        try {
            const { jobId } = req.params;

            const [sttProgress, llmProgress] = await Promise.allSettled([
                speechToTextService.getProgress(jobId),
                aiAnalysisService.getProgress?.(jobId) || Promise.resolve({ progress: 0, status: 'unknown' })
            ]);

            const progress = {
                jobId,
                overall: {
                    progress: Math.max(
                        sttProgress.status === 'fulfilled' ? sttProgress.value.progress || 0 : 0,
                        llmProgress.status === 'fulfilled' ? llmProgress.value.progress || 0 : 0
                    ),
                    status: sttProgress.status === 'fulfilled' && llmProgress.status === 'fulfilled' ? 'processing' : 'error'
                },
                services: {
                    stt: sttProgress.status === 'fulfilled' ? sttProgress.value : { error: sttProgress.reason?.message },
                    llm: llmProgress.status === 'fulfilled' ? llmProgress.value : { error: llmProgress.reason?.message }
                },
                timestamp: new Date().toISOString()
            };

            res.json({
                success: true,
                progress
            });

        } catch (error) {
            logger.error('Progress check error:', error);
            res.status(500).json({
                error: 'Failed to get progress',
                message: error.message
            });
        }
    }

    /**
     * Get AI services status and availability
     */
    async getServiceStatus(req, res) {
        try {
            const [sttStatus, llmStatus] = await Promise.allSettled([
                speechToTextService.getServiceStatus(),
                aiAnalysisService.getServiceStatus()
            ]);

            const status = {
                overall: sttStatus.status === 'fulfilled' && llmStatus.status === 'fulfilled' ? 'healthy' : 'degraded',
                services: {
                    speechToText: sttStatus.status === 'fulfilled' ? sttStatus.value : { error: sttStatus.reason?.message },
                    llm: llmStatus.status === 'fulfilled' ? llmStatus.value : { error: llmStatus.reason?.message }
                },
                timestamp: new Date().toISOString()
            };

            res.json({
                success: true,
                status
            });

        } catch (error) {
            logger.error('Service status error:', error);
            res.status(500).json({
                error: 'Failed to get service status',
                message: error.message
            });
        }
    }

    /**
     * Health check endpoint
     */
    async healthCheck(req, res) {
        try {
            const healthChecks = await Promise.allSettled([
                speechToTextService.healthCheck?.() || Promise.resolve({ status: 'unknown' }),
                aiAnalysisService.healthCheck?.() || Promise.resolve({ status: 'unknown' })
            ]);

            const overallHealth = healthChecks.every(check => 
                check.status === 'fulfilled' && check.value.status === 'healthy'
            );

            res.json({
                status: overallHealth ? 'healthy' : 'degraded',
                services: {
                    stt: healthChecks[0].status === 'fulfilled' ? healthChecks[0].value : { status: 'error' },
                    llm: healthChecks[1].status === 'fulfilled' ? healthChecks[1].value : { status: 'error' }
                },
                timestamp: new Date().toISOString(),
                uptime: process.uptime()
            });

        } catch (error) {
            logger.error('Health check error:', error);
            res.status(500).json({
                status: 'error',
                message: error.message
            });
        }
    }

    /**
     * Generate action items from transcription
     */
    async generateActionItems(req, res) {
        try {
            const { transcription, options = {} } = req.body;

            if (!transcription) {
                return res.status(400).json({
                    error: 'Transcription is required',
                    code: 'MISSING_TRANSCRIPTION'
                });
            }

            const result = await aiAnalysisService.generateActionItems(transcription, options);

            res.json({
                success: true,
                actionItems: result.actionItems || result.data?.actionItems,
                metadata: result.metadata
            });

        } catch (error) {
            logger.error('Action items generation error:', error);
            res.status(500).json({
                error: 'Failed to generate action items',
                message: error.message
            });
        }
    }

    /**
     * Generate meeting summary from transcription
     */
    async generateSummary(req, res) {
        try {
            const { transcription, options = {} } = req.body;

            if (!transcription) {
                return res.status(400).json({
                    error: 'Transcription is required',
                    code: 'MISSING_TRANSCRIPTION'
                });
            }

            const result = await aiAnalysisService.generateSummary(transcription, options);

            res.json({
                success: true,
                summary: result.summary || result.data?.summary,
                metadata: result.metadata
            });

        } catch (error) {
            logger.error('Summary generation error:', error);
            res.status(500).json({
                error: 'Failed to generate summary',
                message: error.message
            });
        }
    }

    /**
     * Extract decisions from meeting transcription
     */
    async extractDecisions(req, res) {
        try {
            const { transcription, options = {} } = req.body;

            if (!transcription) {
                return res.status(400).json({
                    error: 'Transcription is required',
                    code: 'MISSING_TRANSCRIPTION'
                });
            }

            const result = await aiAnalysisService.extractDecisions(transcription, options);

            res.json({
                success: true,
                decisions: result.decisions || result.data?.decisions,
                metadata: result.metadata
            });

        } catch (error) {
            logger.error('Decision extraction error:', error);
            res.status(500).json({
                error: 'Failed to extract decisions',
                message: error.message
            });
        }
    }

    /**
     * Smart conversation analysis with intelligent ticket generation
     */
    async analyzeConversationSmart(req, res) {
        const startTime = Date.now();
        
        try {
            const { transcript } = req.body;
            
            if (!transcript) {
                return res.status(400).json({
                    success: false,
                    error: 'Transcript is required for conversation analysis'
                });
            }
            
            logger.info('Starting smart conversation analysis');
            
            // Call the smart conversation analysis endpoint on the HuggingFace service
            const analysisResult = await aiAnalysisService.analyzeConversationSmart(transcript);
            
            if (!analysisResult.success) {
                throw new Error(analysisResult.error || 'Smart conversation analysis failed');
            }
            
            const processingTime = Date.now() - startTime;
            
            logger.info(`Smart conversation analysis completed in ${processingTime}ms`, {
                insights_found: analysisResult.conversation_analysis?.insights_found,
                tickets_generated: analysisResult.intelligent_tickets?.length,
                project_context: analysisResult.conversation_analysis?.project_context
            });
            
            res.json({
                success: true,
                conversation_analysis: analysisResult.conversation_analysis,
                intelligent_tickets: analysisResult.intelligent_tickets,
                insights_analyzed: analysisResult.insights_analyzed,
                actionable_insights: analysisResult.actionable_insights,
                processing_metadata: {
                    ...analysisResult.processing_metadata,
                    processing_time_ms: processingTime,
                    timestamp: new Date().toISOString()
                }
            });
            
        } catch (error) {
            const processingTime = Date.now() - startTime;
            logger.error('Smart conversation analysis failed:', error);
            
            res.status(500).json({
                success: false,
                error: 'Smart conversation analysis failed',
                message: error.message,
                processing_metadata: {
                    processing_time_ms: processingTime,
                    timestamp: new Date().toISOString(),
                    error_type: 'smart_analysis_error'
                }
            });
        }
    }

    // ...existing code...
}

// Add request timing middleware
const addRequestTiming = (req, res, next) => {
    req.startTime = Date.now();
    next();
};

module.exports = {
    AIController: new AIController(),
    addRequestTiming
};