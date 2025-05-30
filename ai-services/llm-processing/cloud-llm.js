#!/usr/bin/env node
/**
 * MeetFlow Cloud LLM Service
 * Multi-provider cloud LLM integration with intelligent routing and optimization
 * 
 * TEMPORARILY COMMENTED OUT FOR LOCAL-ONLY DEPLOYMENT
 * Uncomment this entire file when cloud functionality is needed
 */

/* CLOUD SERVICE TEMPORARILY DISABLED
const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// Cloud LLM providers
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const axios = require('axios');
const NodeCache = require('node-cache');

// Configuration
const CONFIG = {
    PORT: process.env.CLOUD_LLM_PORT || 8004,
    
    // Provider configurations
    PROVIDERS: {
        OPENAI: {
            enabled: !!process.env.OPENAI_API_KEY,
            apiKey: process.env.OPENAI_API_KEY,
            models: {
                'gpt-4': { costPer1kTokens: 0.03, maxTokens: 8192, contextWindow: 8192 },
                'gpt-4-turbo': { costPer1kTokens: 0.01, maxTokens: 4096, contextWindow: 128000 },
                'gpt-3.5-turbo': { costPer1kTokens: 0.002, maxTokens: 4096, contextWindow: 16384 }
            },
            defaultModel: 'gpt-3.5-turbo',
            reliability: 0.95,
            avgResponseTime: 2000
        },
        ANTHROPIC: {
            enabled: !!process.env.ANTHROPIC_API_KEY,
            apiKey: process.env.ANTHROPIC_API_KEY,
            models: {
                'claude-3-opus': { costPer1kTokens: 0.015, maxTokens: 4096, contextWindow: 200000 },
                'claude-3-sonnet': { costPer1kTokens: 0.003, maxTokens: 4096, contextWindow: 200000 },
                'claude-3-haiku': { costPer1kTokens: 0.00025, maxTokens: 4096, contextWindow: 200000 }
            },
            defaultModel: 'claude-3-sonnet',
            reliability: 0.93,
            avgResponseTime: 2500
        },
        GOOGLE: {
            enabled: !!process.env.GOOGLE_AI_API_KEY,
            apiKey: process.env.GOOGLE_AI_API_KEY,
            models: {
                'gemini-pro': { costPer1kTokens: 0.0005, maxTokens: 2048, contextWindow: 32768 },
                'gemini-pro-vision': { costPer1kTokens: 0.0025, maxTokens: 2048, contextWindow: 16384 }
            },
            defaultModel: 'gemini-pro',
            reliability: 0.90,
            avgResponseTime: 1800
        }
    },
    
    // Processing settings
    CACHE_TTL: 3600, // 1 hour
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    REQUEST_TIMEOUT: 30000,
    
    // Cost management
    DAILY_BUDGET_USD: parseFloat(process.env.DAILY_BUDGET) || 50.0,
    COST_ALERT_THRESHOLD: 0.8, // 80% of budget
    
    // Rate limiting
    RATE_LIMITS: {
        openai: { requestsPerMinute: 3000, tokensPerMinute: 90000 },
        anthropic: { requestsPerMinute: 1000, tokensPerMinute: 40000 },
        google: { requestsPerMinute: 1500, tokensPerMinute: 30000 }
    }
};

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Initialize providers
let providers = {};
let promptTemplates = {};
let cache = new NodeCache({ stdTTL: CONFIG.CACHE_TTL });

// Cost and usage tracking
let usageStats = {
    requests: 0,
    totalCost: 0,
    dailyCost: 0,
    lastResetDate: new Date().toISOString().split('T')[0],
    providerStats: {
        openai: { requests: 0, cost: 0, tokens: 0, avgResponseTime: 0 },
        anthropic: { requests: 0, cost: 0, tokens: 0, avgResponseTime: 0 },
        google: { requests: 0, cost: 0, tokens: 0, avgResponseTime: 0 }
    }
};

class CloudLLMManager {
    constructor() {
        this.initializeProviders();
        this.loadPromptTemplates();
        this.requestQueue = new Map();
        this.abTestConfig = new Map();
    }

    async initializeProviders() {
        // Initialize OpenAI
        if (CONFIG.PROVIDERS.OPENAI.enabled) {
            try {
                providers.openai = new OpenAI({
                    apiKey: CONFIG.PROVIDERS.OPENAI.apiKey,
                    timeout: CONFIG.REQUEST_TIMEOUT
                });
                console.log('✅ OpenAI initialized');
            } catch (error) {
                console.warn('⚠️ OpenAI initialization failed:', error.message);
            }
        }

        // Initialize Anthropic
        if (CONFIG.PROVIDERS.ANTHROPIC.enabled) {
            try {
                providers.anthropic = new Anthropic({
                    apiKey: CONFIG.PROVIDERS.ANTHROPIC.apiKey,
                    timeout: CONFIG.REQUEST_TIMEOUT
                });
                console.log('✅ Anthropic initialized');
            } catch (error) {
                console.warn('⚠️ Anthropic initialization failed:', error.message);
            }
        }

        // Initialize Google AI
        if (CONFIG.PROVIDERS.GOOGLE.enabled) {
            try {
                providers.google = new GoogleGenerativeAI(CONFIG.PROVIDERS.GOOGLE.apiKey);
                console.log('✅ Google AI initialized');
            } catch (error) {
                console.warn('⚠️ Google AI initialization failed:', error.message);
            }
        }
    }

    async loadPromptTemplates() {
        try {
            const templatesPath = path.join(__dirname, 'prompt-templates.json');
            const templatesData = await fs.readFile(templatesPath, 'utf8');
            promptTemplates = JSON.parse(templatesData);
            console.log('✅ Prompt templates loaded');
        } catch (error) {
            console.warn('⚠️ Failed to load prompt templates:', error.message);
            promptTemplates = { templates: {} };
        }
    }

    selectOptimalProvider(task, textLength, priority = 'balanced') {
        const availableProviders = [];
        const estimatedTokens = Math.ceil(textLength / 4); // Rough estimation

        Object.entries(CONFIG.PROVIDERS).forEach(([name, config]) => {
            if (!config.enabled || !providers[name.toLowerCase()]) return;

            const provider = name.toLowerCase();
            const modelInfo = Object.values(config.models)[0]; // Use default model
            
            // Check if provider can handle the request
            if (estimatedTokens > modelInfo.contextWindow) return;

            const estimatedCost = (estimatedTokens / 1000) * modelInfo.costPer1kTokens;
            
            availableProviders.push({
                name: provider,
                cost: estimatedCost,
                reliability: config.reliability,
                responseTime: config.avgResponseTime,
                model: config.defaultModel,
                score: this.calculateProviderScore(config, estimatedCost, priority)
            });
        });

        if (availableProviders.length === 0) {
            throw new Error('No providers available for this request');
        }

        // Sort by score and return best provider
        return availableProviders.sort((a, b) => b.score - a.score)[0];
    }

    calculateProviderScore(config, estimatedCost, priority) {
        const weights = {
            balanced: { cost: 0.3, reliability: 0.4, speed: 0.3 },
            cost: { cost: 0.7, reliability: 0.2, speed: 0.1 },
            speed: { cost: 0.1, reliability: 0.3, speed: 0.6 },
            reliability: { cost: 0.2, reliability: 0.6, speed: 0.2 }
        };

        const w = weights[priority] || weights.balanced;
        
        // Normalize scores (0-1 scale)
        const costScore = 1 - Math.min(estimatedCost / 0.1, 1); // $0.10 as max reference
        const reliabilityScore = config.reliability;
        const speedScore = 1 - Math.min(config.avgResponseTime / 5000, 1); // 5s as max reference

        return (costScore * w.cost) + (reliabilityScore * w.reliability) + (speedScore * w.speed);
    }

    async processWithOpenAI(prompt, options = {}) {
        if (!providers.openai) {
            throw new Error('OpenAI not available');
        }

        const model = options.model || CONFIG.PROVIDERS.OPENAI.defaultModel;
        const startTime = Date.now();

        const response = await providers.openai.chat.completions.create({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: options.maxTokens || 1000,
            temperature: options.temperature || 0.3,
            stream: options.stream || false
        });

        const responseTime = Date.now() - startTime;
        const tokens = response.usage.total_tokens;
        const cost = (tokens / 1000) * CONFIG.PROVIDERS.OPENAI.models[model].costPer1kTokens;

        this.updateUsageStats('openai', cost, tokens, responseTime);

        return {
            text: response.choices[0].message.content,
            tokens: tokens,
            cost: cost,
            provider: 'openai',
            model: model,
            responseTime: responseTime,
            confidence: this.calculateConfidence(response.choices[0])
        };
    }

    async processWithAnthropic(prompt, options = {}) {
        if (!providers.anthropic) {
            throw new Error('Anthropic not available');
        }

        const model = options.model || CONFIG.PROVIDERS.ANTHROPIC.defaultModel;
        const startTime = Date.now();

        const response = await providers.anthropic.messages.create({
            model: model,
            max_tokens: options.maxTokens || 1000,
            temperature: options.temperature || 0.3,
            messages: [{ role: 'user', content: prompt }]
        });

        const responseTime = Date.now() - startTime;
        const tokens = response.usage.input_tokens + response.usage.output_tokens;
        const cost = (tokens / 1000) * CONFIG.PROVIDERS.ANTHROPIC.models[model].costPer1kTokens;

        this.updateUsageStats('anthropic', cost, tokens, responseTime);

        return {
            text: response.content[0].text,
            tokens: tokens,
            cost: cost,
            provider: 'anthropic',
            model: model,
            responseTime: responseTime,
            confidence: 0.9 // Anthropic doesn't provide confidence scores
        };
    }

    async processWithGoogle(prompt, options = {}) {
        if (!providers.google) {
            throw new Error('Google AI not available');
        }

        const model = providers.google.getGenerativeModel({ 
            model: options.model || CONFIG.PROVIDERS.GOOGLE.defaultModel 
        });
        const startTime = Date.now();

        const result = await model.generateContent(prompt);
        const response = await result.response;
        
        const responseTime = Date.now() - startTime;
        const tokens = response.promptFeedback?.tokenCount || 1000; // Estimate if not provided
        const cost = (tokens / 1000) * CONFIG.PROVIDERS.GOOGLE.models['gemini-pro'].costPer1kTokens;

        this.updateUsageStats('google', cost, tokens, responseTime);

        return {
            text: response.text(),
            tokens: tokens,
            cost: cost,
            provider: 'google',
            model: 'gemini-pro',
            responseTime: responseTime,
            confidence: 0.85 // Google doesn't provide confidence scores
        };
    }

    async processRequest(prompt, options = {}) {
        // Check daily budget
        this.checkBudgetLimits();

        // Try caching first
        const cacheKey = this.generateCacheKey(prompt, options);
        const cachedResult = cache.get(cacheKey);
        if (cachedResult && options.useCache !== false) {
            return { ...cachedResult, cached: true };
        }

        // Select optimal provider
        const provider = this.selectOptimalProvider(
            options.task || 'general',
            prompt.length,
            options.priority || 'balanced'
        );

        console.log(`Using provider: ${provider.name} for request`);

        // Process with selected provider
        let result;
        try {
            switch (provider.name) {
                case 'openai':
                    result = await this.processWithOpenAI(prompt, { ...options, model: provider.model });
                    break;
                case 'anthropic':
                    result = await this.processWithAnthropic(prompt, { ...options, model: provider.model });
                    break;
                case 'google':
                    result = await this.processWithGoogle(prompt, { ...options, model: provider.model });
                    break;
                default:
                    throw new Error(`Unknown provider: ${provider.name}`);
            }

            // Cache the result
            if (options.useCache !== false) {
                cache.set(cacheKey, result);
            }

            return result;

        } catch (error) {
            console.warn(`Provider ${provider.name} failed:`, error.message);
            
            // Try fallback providers
            const fallbackProviders = ['openai', 'anthropic', 'google']
                .filter(p => p !== provider.name && providers[p]);

            for (const fallbackProvider of fallbackProviders) {
                try {
                    console.log(`Trying fallback provider: ${fallbackProvider}`);
                    
                    switch (fallbackProvider) {
                        case 'openai':
                            result = await this.processWithOpenAI(prompt, options);
                            break;
                        case 'anthropic':
                            result = await this.processWithAnthropic(prompt, options);
                            break;
                        case 'google':
                            result = await this.processWithGoogle(prompt, options);
                            break;
                    }

                    if (options.useCache !== false) {
                        cache.set(cacheKey, result);
                    }

                    return result;

                } catch (fallbackError) {
                    console.warn(`Fallback provider ${fallbackProvider} failed:`, fallbackError.message);
                    continue;
                }
            }

            throw new Error('All LLM providers failed');
        }
    }

    generateCacheKey(prompt, options) {
        const key = JSON.stringify({
            prompt: prompt.substring(0, 100), // First 100 chars
            temperature: options.temperature,
            maxTokens: options.maxTokens,
            task: options.task
        });
        return require('crypto').createHash('md5').update(key).digest('hex');
    }

    updateUsageStats(provider, cost, tokens, responseTime) {
        const stats = usageStats.providerStats[provider];
        stats.requests++;
        stats.cost += cost;
        stats.tokens += tokens;
        stats.avgResponseTime = (stats.avgResponseTime * (stats.requests - 1) + responseTime) / stats.requests;

        usageStats.requests++;
        usageStats.totalCost += cost;
        usageStats.dailyCost += cost;

        // Reset daily cost if new day
        const today = new Date().toISOString().split('T')[0];
        if (today !== usageStats.lastResetDate) {
            usageStats.dailyCost = cost;
            usageStats.lastResetDate = today;
        }
    }

    checkBudgetLimits() {
        if (usageStats.dailyCost >= CONFIG.DAILY_BUDGET_USD) {
            throw new Error('Daily budget limit exceeded');
        }

        if (usageStats.dailyCost >= CONFIG.DAILY_BUDGET_USD * CONFIG.COST_ALERT_THRESHOLD) {
            console.warn(`⚠️ Cost alert: ${(usageStats.dailyCost / CONFIG.DAILY_BUDGET_USD * 100).toFixed(1)}% of daily budget used`);
        }
    }

    calculateConfidence(choice) {
        // Simple confidence calculation based on various factors
        if (choice.finish_reason === 'stop') {
            return 0.9;
        } else if (choice.finish_reason === 'length') {
            return 0.7;
        }
        return 0.5;
    }

    getPromptTemplate(category, type, complexity = 'simple') {
        try {
            return promptTemplates.templates[category][type][complexity];
        } catch (error) {
            console.warn(`Template not found: ${category}.${type}.${complexity}`);
            return null;
        }
    }

    injectVariables(template, variables) {
        let prompt = template.prompt;
        
        Object.entries(variables).forEach(([key, value]) => {
            const placeholder = `{{${key}}}`;
            prompt = prompt.replace(new RegExp(placeholder, 'g'), value);
        });

        return prompt;
    }

    async analyzeWithTemplate(templatePath, variables, options = {}) {
        const pathParts = templatePath.split('.');
        const template = this.getPromptTemplate(pathParts[0], pathParts[1], pathParts[2] || 'simple');
        
        if (!template) {
            throw new Error(`Template not found: ${templatePath}`);
        }

        const prompt = this.injectVariables(template, variables);
        
        const processingOptions = {
            ...options,
            temperature: template.temperature || options.temperature,
            maxTokens: template.max_tokens || options.maxTokens,
            task: pathParts[0]
        };

        return await this.processRequest(prompt, processingOptions);
    }

    getUsageStats() {
        return usageStats;
    }

    clearCache() {
        cache.flushAll();
        return { status: 'cache_cleared' };
    }
}

// Initialize LLM manager
const llmManager = new CloudLLMManager();

// API Routes
app.get('/health', (req, res) => {
    const availableProviders = Object.keys(providers);
    res.json({
        status: 'healthy',
        providers: availableProviders,
        uptime: process.uptime(),
        dailyCost: usageStats.dailyCost,
        budgetRemaining: CONFIG.DAILY_BUDGET_USD - usageStats.dailyCost
    });
});

app.post('/process', async (req, res) => {
    try {
        const { prompt, options = {} } = req.body;
        
        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        const result = await llmManager.processRequest(prompt, options);
        
        res.json({
            success: true,
            ...result,
            requestId: uuidv4()
        });

    } catch (error) {
        console.error('Processing error:', error);
        
        res.status(500).json({
            error: 'Processing failed',
            message: error.message,
            requestId: uuidv4()
        });
    }
});

app.post('/analyze-template', async (req, res) => {
    try {
        const { templatePath, variables, options = {} } = req.body;
        
        if (!templatePath || !variables) {
            return res.status(400).json({ error: 'templatePath and variables are required' });
        }

        const result = await llmManager.analyzeWithTemplate(templatePath, variables, options);
        
        res.json({
            success: true,
            ...result,
            templateUsed: templatePath,
            requestId: uuidv4()
        });

    } catch (error) {
        console.error('Template analysis error:', error);
        
        res.status(500).json({
            error: 'Template analysis failed',
            message: error.message,
            requestId: uuidv4()
        });
    }
});

app.get('/templates', (req, res) => {
    res.json(promptTemplates);
});

app.get('/providers', (req, res) => {
    const providerInfo = Object.entries(CONFIG.PROVIDERS)
        .filter(([name, config]) => config.enabled && providers[name.toLowerCase()])
        .map(([name, config]) => ({
            name: name.toLowerCase(),
            models: Object.keys(config.models),
            defaultModel: config.defaultModel,
            reliability: config.reliability,
            avgResponseTime: config.avgResponseTime
        }));
    
    res.json({ providers: providerInfo });
});

app.get('/usage', (req, res) => {
    res.json(llmManager.getUsageStats());
});

app.post('/cache/clear', (req, res) => {
    const result = llmManager.clearCache();
    res.json(result);
});

app.get('/cache/stats', (req, res) => {
    res.json({
        keys: cache.keys().length,
        hits: cache.getStats().hits,
        misses: cache.getStats().misses,
        ttl: CONFIG.CACHE_TTL
    });
});

// Batch processing endpoint
app.post('/batch', async (req, res) => {
    try {
        const { requests } = req.body;
        
        if (!Array.isArray(requests)) {
            return res.status(400).json({ error: 'requests must be an array' });
        }

        const results = await Promise.allSettled(
            requests.map(request => 
                llmManager.processRequest(request.prompt, request.options || {})
            )
        );

        const processedResults = results.map((result, index) => ({
            index,
            success: result.status === 'fulfilled',
            data: result.status === 'fulfilled' ? result.value : null,
            error: result.status === 'rejected' ? result.reason.message : null
        }));

        res.json({
            success: true,
            results: processedResults,
            batchId: uuidv4()
        });

    } catch (error) {
        console.error('Batch processing error:', error);
        
        res.status(500).json({
            error: 'Batch processing failed',
            message: error.message
        });
    }
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
    console.log(`🧠 MeetFlow Cloud LLM Service running on port ${CONFIG.PORT}`);
    console.log(`💰 Daily budget: $${CONFIG.DAILY_BUDGET_USD}`);
    console.log(`📊 Available providers: ${Object.keys(providers).join(', ')}`);
});

module.exports = app;

*/ // END CLOUD SERVICE TEMPORARILY DISABLED

// Placeholder service for local-only deployment
const express = require('express');
const app = express();

app.use(express.json());

// Return error for all cloud LLM requests when disabled
app.all('*', (req, res) => {
    res.status(503).json({
        error: 'Cloud LLM service temporarily disabled',
        message: 'System is running in local-only mode. To enable cloud services, uncomment the cloud-llm.js file.',
        provider: 'disabled'
    });
});

const PORT = process.env.CLOUD_LLM_PORT || 8004;
app.listen(PORT, () => {
    console.log(`🚫 Cloud LLM Service DISABLED - placeholder running on port ${PORT}`);
    console.log(`💡 To enable: Uncomment the main service code in cloud-llm.js`);
});

module.exports = app;