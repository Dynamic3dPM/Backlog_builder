const aiAnalysisService = require('./aiAnalysis');

class TicketGeneratorService {
    constructor() {
        this.defaultPriorities = ['Low', 'Medium', 'High', 'Critical'];
        this.defaultTypes = ['Task', 'Bug', 'Feature', 'Epic', 'Story'];
        this.defaultStatuses = ['To Do', 'In Progress', 'Review', 'Done'];
    }

    /**
     * Generate tickets from meeting analysis
     * @param {Object} analysisResult - Results from AI analysis
     * @param {Object} options - Ticket generation options
     * @returns {Object} Generated tickets with metadata
     */
    async generateTickets(analysisResult, options = {}) {
        const {
            includeActionItems = true,
            includeDecisions = true,
            includeSummaryTicket = true,
            projectId = null,
            assigneeMapping = {},
            priorityRules = {},
            typeMapping = {},
            epicMapping = {},
            templateId = null
        } = options;

        try {
            const tickets = [];

            // Generate summary ticket if requested
            if (includeSummaryTicket && analysisResult.summary) {
                const summaryTicket = await this.generateSummaryTicket(
                    analysisResult,
                    { projectId, templateId }
                );
                tickets.push(summaryTicket);
            }

            // Generate tickets from action items
            if (includeActionItems && analysisResult.actionItems?.length > 0) {
                const actionTickets = await this.generateActionItemTickets(
                    analysisResult.actionItems,
                    {
                        projectId,
                        assigneeMapping,
                        priorityRules,
                        typeMapping,
                        epicMapping
                    }
                );
                tickets.push(...actionTickets);
            }

            // Generate tickets from decisions
            if (includeDecisions && analysisResult.decisions?.length > 0) {
                const decisionTickets = await this.generateDecisionTickets(
                    analysisResult.decisions,
                    {
                        projectId,
                        assigneeMapping,
                        priorityRules,
                        typeMapping
                    }
                );
                tickets.push(...decisionTickets);
            }

            // Generate epic tickets for complex topics
            const epicTickets = await this.generateEpicTickets(
                analysisResult,
                { projectId, epicMapping }
            );
            tickets.push(...epicTickets);

            // Post-process tickets
            const processedTickets = await this.postProcessTickets(tickets, options);

            return {
                success: true,
                tickets: processedTickets,
                summary: {
                    totalTickets: processedTickets.length,
                    ticketsByType: this.groupTicketsByType(processedTickets),
                    ticketsByPriority: this.groupTicketsByPriority(processedTickets),
                    estimatedEffort: this.calculateEstimatedEffort(processedTickets)
                },
                metadata: {
                    sourceAnalysis: {
                        actionItemCount: analysisResult.actionItems?.length || 0,
                        decisionCount: analysisResult.decisions?.length || 0,
                        participantCount: analysisResult.participants?.length || 0
                    },
                    generationOptions: options,
                    generatedAt: new Date().toISOString()
                }
            };

        } catch (error) {
            console.error('Ticket generation failed:', error);
            throw new Error(`Failed to generate tickets: ${error.message}`);
        }
    }

    /**
     * Generate a summary ticket for the meeting
     */
    async generateSummaryTicket(analysisResult, options = {}) {
        const { projectId, templateId } = options;

        const ticket = {
            id: this.generateTicketId(),
            type: 'Epic',
            title: this.generateMeetingTitle(analysisResult),
            description: this.formatSummaryDescription(analysisResult),
            priority: 'Medium',
            status: 'To Do',
            labels: ['meeting-summary', 'auto-generated'],
            projectId,
            templateId,
            metadata: {
                source: 'meeting-summary',
                meetingData: {
                    participants: analysisResult.participants || [],
                    topics: analysisResult.topics || [],
                    sentiment: analysisResult.sentiment,
                    duration: analysisResult.metadata?.duration
                }
            },
            createdAt: new Date().toISOString()
        };

        return ticket;
    }

    /**
     * Generate tickets from action items
     */
    async generateActionItemTickets(actionItems, options = {}) {
        const {
            projectId,
            assigneeMapping = {},
            priorityRules = {},
            typeMapping = {},
            epicMapping = {}
        } = options;

        const tickets = [];

        for (const [index, actionItem] of actionItems.entries()) {
            const ticket = {
                id: this.generateTicketId(),
                type: this.determineTicketType(actionItem, typeMapping),
                title: this.formatActionItemTitle(actionItem),
                description: this.formatActionItemDescription(actionItem),
                priority: this.determinePriority(actionItem, priorityRules),
                status: 'To Do',
                assignee: this.determineAssignee(actionItem, assigneeMapping),
                labels: this.generateLabels(actionItem, 'action-item'),
                dueDate: this.extractDueDate(actionItem),
                estimatedHours: this.estimateEffort(actionItem),
                projectId,
                epic: this.determineEpic(actionItem, epicMapping),
                metadata: {
                    source: 'action-item',
                    originalText: actionItem.text || actionItem.description,
                    confidence: actionItem.confidence,
                    extractedFrom: actionItem.context,
                    index
                },
                createdAt: new Date().toISOString()
            };

            tickets.push(ticket);
        }

        return tickets;
    }

    /**
     * Generate tickets from decisions
     */
    async generateDecisionTickets(decisions, options = {}) {
        const {
            projectId,
            assigneeMapping = {},
            priorityRules = {},
            typeMapping = {}
        } = options;

        const tickets = [];

        for (const [index, decision] of decisions.entries()) {
            const ticket = {
                id: this.generateTicketId(),
                type: 'Task',
                title: this.formatDecisionTitle(decision),
                description: this.formatDecisionDescription(decision),
                priority: this.determinePriority(decision, priorityRules),
                status: 'To Do',
                assignee: this.determineAssignee(decision, assigneeMapping),
                labels: this.generateLabels(decision, 'decision'),
                projectId,
                metadata: {
                    source: 'decision',
                    originalText: decision.text || decision.description,
                    confidence: decision.confidence,
                    impact: decision.impact,
                    participants: decision.participants,
                    index
                },
                createdAt: new Date().toISOString()
            };

            tickets.push(ticket);
        }

        return tickets;
    }

    /**
     * Generate epic tickets for complex topics
     */
    async generateEpicTickets(analysisResult, options = {}) {
        const { projectId, epicMapping = {} } = options;
        const tickets = [];

        if (!analysisResult.topics || analysisResult.topics.length === 0) {
            return tickets;
        }

        // Group related action items and decisions by topic
        const topicGroups = this.groupByTopics(analysisResult);

        for (const [topic, items] of Object.entries(topicGroups)) {
            if (items.length >= 3) { // Only create epics for topics with multiple items
                const epic = {
                    id: this.generateTicketId(),
                    type: 'Epic',
                    title: this.formatTopicTitle(topic),
                    description: this.formatTopicDescription(topic, items, analysisResult),
                    priority: this.determineTopicPriority(items),
                    status: 'To Do',
                    labels: ['epic', 'auto-generated', 'topic-based'],
                    projectId,
                    metadata: {
                        source: 'topic-analysis',
                        topic,
                        relatedItems: items.length,
                        complexity: this.assessTopicComplexity(items)
                    },
                    createdAt: new Date().toISOString()
                };

                tickets.push(epic);
            }
        }

        return tickets;
    }

    /**
     * Post-process tickets for consistency and optimization
     */
    async postProcessTickets(tickets, options = {}) {
        const processed = tickets.map(ticket => {
            // Ensure all required fields
            ticket.priority = ticket.priority || 'Medium';
            ticket.status = ticket.status || 'To Do';
            ticket.labels = ticket.labels || [];
            
            // Add auto-generated label
            if (!ticket.labels.includes('auto-generated')) {
                ticket.labels.push('auto-generated');
            }

            // Validate and clean up
            ticket.title = this.cleanTitle(ticket.title);
            ticket.description = this.cleanDescription(ticket.description);

            return ticket;
        });

        // Remove duplicates
        const deduped = this.removeDuplicateTickets(processed);

        // Link related tickets
        const linked = this.linkRelatedTickets(deduped);

        return linked;
    }

    /**
     * Helper methods for ticket generation
     */
    generateTicketId() {
        return 'TICKET-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }

    generateMeetingTitle(analysisResult) {
        const date = new Date().toLocaleDateString();
        const participants = analysisResult.participants?.length || 0;
        return `Meeting Summary - ${date} (${participants} participants)`;
    }

    formatSummaryDescription(analysisResult) {
        let description = '## Meeting Summary\n\n';
        
        if (analysisResult.summary) {
            description += `${analysisResult.summary}\n\n`;
        }

        if (analysisResult.keyPoints?.length > 0) {
            description += '## Key Points\n';
            analysisResult.keyPoints.forEach(point => {
                description += `- ${point}\n`;
            });
            description += '\n';
        }

        if (analysisResult.participants?.length > 0) {
            description += '## Participants\n';
            description += analysisResult.participants.join(', ') + '\n\n';
        }

        if (analysisResult.actionItems?.length > 0) {
            description += `## Generated ${analysisResult.actionItems.length} action item tickets\n\n`;
        }

        if (analysisResult.decisions?.length > 0) {
            description += `## Generated ${analysisResult.decisions.length} decision tickets\n\n`;
        }

        return description;
    }

    formatActionItemTitle(actionItem) {
        let title = actionItem.title || actionItem.action || actionItem.text || 'Action Item';
        
        // Truncate if too long
        if (title.length > 100) {
            title = title.substring(0, 97) + '...';
        }

        return title;
    }

    formatActionItemDescription(actionItem) {
        let description = '';

        if (actionItem.description || actionItem.details) {
            description += `${actionItem.description || actionItem.details}\n\n`;
        }

        if (actionItem.assignee) {
            description += `**Assigned to:** ${actionItem.assignee}\n`;
        }

        if (actionItem.dueDate) {
            description += `**Due Date:** ${actionItem.dueDate}\n`;
        }

        if (actionItem.context) {
            description += `**Context:** ${actionItem.context}\n`;
        }

        return description;
    }

    formatDecisionTitle(decision) {
        let title = decision.title || decision.decision || decision.text || 'Decision';
        
        if (title.length > 100) {
            title = title.substring(0, 97) + '...';
        }

        return `Decision: ${title}`;
    }

    formatDecisionDescription(decision) {
        let description = '';

        if (decision.description || decision.details) {
            description += `${decision.description || decision.details}\n\n`;
        }

        if (decision.rationale) {
            description += `**Rationale:** ${decision.rationale}\n\n`;
        }

        if (decision.impact) {
            description += `**Impact:** ${decision.impact}\n\n`;
        }

        if (decision.participants?.length > 0) {
            description += `**Decision Makers:** ${decision.participants.join(', ')}\n\n`;
        }

        return description;
    }

    determineTicketType(item, typeMapping = {}) {
        // Custom type mapping
        if (typeMapping[item.type]) {
            return typeMapping[item.type];
        }

        // Intelligent type detection
        const text = (item.text || item.description || '').toLowerCase();
        
        if (text.includes('bug') || text.includes('fix') || text.includes('error')) {
            return 'Bug';
        }
        
        if (text.includes('feature') || text.includes('new') || text.includes('add')) {
            return 'Feature';
        }
        
        if (text.includes('research') || text.includes('investigate') || text.includes('analyze')) {
            return 'Task';
        }

        return 'Task'; // Default
    }

    determinePriority(item, priorityRules = {}) {
        // Custom priority rules
        if (priorityRules[item.priority]) {
            return priorityRules[item.priority];
        }

        // Extract priority from text
        const text = (item.text || item.description || '').toLowerCase();
        
        if (text.includes('urgent') || text.includes('critical') || text.includes('asap')) {
            return 'Critical';
        }
        
        if (text.includes('high') || text.includes('important') || text.includes('priority')) {
            return 'High';
        }
        
        if (text.includes('low') || text.includes('when possible') || text.includes('eventually')) {
            return 'Low';
        }

        return 'Medium'; // Default
    }

    determineAssignee(item, assigneeMapping = {}) {
        if (item.assignee && assigneeMapping[item.assignee]) {
            return assigneeMapping[item.assignee];
        }
        
        return item.assignee || null;
    }

    generateLabels(item, source) {
        const labels = [source];
        
        const text = (item.text || item.description || '').toLowerCase();
        
        if (text.includes('urgent')) labels.push('urgent');
        if (text.includes('research')) labels.push('research');
        if (text.includes('testing')) labels.push('testing');
        if (text.includes('documentation')) labels.push('docs');
        if (text.includes('frontend')) labels.push('frontend');
        if (text.includes('backend')) labels.push('backend');
        if (text.includes('api')) labels.push('api');
        if (text.includes('ui') || text.includes('ux')) labels.push('ui-ux');
        
        return labels;
    }

    extractDueDate(item) {
        if (item.dueDate) {
            return item.dueDate;
        }

        // Try to extract date from text
        const text = item.text || item.description || '';
        const datePatterns = [
            /by (\w+ \d{1,2})/i,
            /due (\w+ \d{1,2})/i,
            /(\d{1,2}\/\d{1,2}\/\d{4})/,
            /(\d{4}-\d{2}-\d{2})/
        ];

        for (const pattern of datePatterns) {
            const match = text.match(pattern);
            if (match) {
                return match[1];
            }
        }

        return null;
    }

    estimateEffort(item) {
        const text = (item.text || item.description || '').toLowerCase();
        
        // Look for explicit time estimates
        if (text.includes('hour')) {
            const hourMatch = text.match(/(\d+)\s*hour/);
            if (hourMatch) return parseInt(hourMatch[1]);
        }
        
        if (text.includes('day')) {
            const dayMatch = text.match(/(\d+)\s*day/);
            if (dayMatch) return parseInt(dayMatch[1]) * 8;
        }

        // Estimate based on complexity indicators
        if (text.includes('simple') || text.includes('quick')) return 2;
        if (text.includes('complex') || text.includes('research')) return 16;
        if (text.includes('implement') || text.includes('develop')) return 8;
        
        return 4; // Default estimate
    }

    groupTicketsByType(tickets) {
        const groups = {};
        tickets.forEach(ticket => {
            groups[ticket.type] = (groups[ticket.type] || 0) + 1;
        });
        return groups;
    }

    groupTicketsByPriority(tickets) {
        const groups = {};
        tickets.forEach(ticket => {
            groups[ticket.priority] = (groups[ticket.priority] || 0) + 1;
        });
        return groups;
    }

    calculateEstimatedEffort(tickets) {
        return tickets.reduce((total, ticket) => {
            return total + (ticket.estimatedHours || 0);
        }, 0);
    }

    groupByTopics(analysisResult) {
        const groups = {};
        
        // Group action items by topic
        (analysisResult.actionItems || []).forEach(item => {
            const topic = this.extractTopic(item, analysisResult.topics);
            if (topic) {
                groups[topic] = groups[topic] || [];
                groups[topic].push({ ...item, type: 'action' });
            }
        });

        // Group decisions by topic
        (analysisResult.decisions || []).forEach(item => {
            const topic = this.extractTopic(item, analysisResult.topics);
            if (topic) {
                groups[topic] = groups[topic] || [];
                groups[topic].push({ ...item, type: 'decision' });
            }
        });

        return groups;
    }

    extractTopic(item, topics) {
        const text = (item.text || item.description || '').toLowerCase();
        return topics?.find(topic => text.includes(topic.toLowerCase()));
    }

    removeDuplicateTickets(tickets) {
        const seen = new Set();
        return tickets.filter(ticket => {
            const key = `${ticket.title.toLowerCase()}-${ticket.type}`;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }

    linkRelatedTickets(tickets) {
        // Add relationships between tickets
        return tickets.map(ticket => {
            const related = tickets
                .filter(other => other.id !== ticket.id)
                .filter(other => this.areTicketsRelated(ticket, other))
                .map(other => other.id);

            if (related.length > 0) {
                ticket.relatedTickets = related;
            }

            return ticket;
        });
    }

    areTicketsRelated(ticket1, ticket2) {
        const text1 = (ticket1.title + ' ' + ticket1.description).toLowerCase();
        const text2 = (ticket2.title + ' ' + ticket2.description).toLowerCase();
        
        // Check for common keywords
        const words1 = new Set(text1.split(/\s+/));
        const words2 = new Set(text2.split(/\s+/));
        const intersection = new Set([...words1].filter(x => words2.has(x)));
        
        return intersection.size >= 3; // At least 3 common words
    }

    cleanTitle(title) {
        return title
            .replace(/^\s+|\s+$/g, '') // Trim
            .replace(/\s+/g, ' ') // Normalize whitespace
            .replace(/[^\w\s-]/g, '') // Remove special chars except hyphens
            .substring(0, 100); // Limit length
    }

    cleanDescription(description) {
        return description
            .replace(/^\s+|\s+$/g, '') // Trim
            .replace(/\n{3,}/g, '\n\n') // Normalize line breaks
            .substring(0, 5000); // Limit length
    }

    formatTopicTitle(topic) {
        return `Epic: ${topic.charAt(0).toUpperCase() + topic.slice(1)}`;
    }

    formatTopicDescription(topic, items, analysisResult) {
        let description = `## Epic: ${topic}\n\n`;
        description += `This epic encompasses ${items.length} related items from the meeting analysis.\n\n`;
        
        description += '### Related Items:\n';
        items.forEach((item, index) => {
            description += `${index + 1}. ${item.text || item.description}\n`;
        });

        return description;
    }

    determineTopicPriority(items) {
        const priorities = items.map(item => item.priority).filter(Boolean);
        if (priorities.includes('Critical')) return 'Critical';
        if (priorities.includes('High')) return 'High';
        if (priorities.includes('Medium')) return 'Medium';
        return 'Low';
    }

    assessTopicComplexity(items) {
        if (items.length >= 5) return 'high';
        if (items.length >= 3) return 'medium';
        return 'low';
    }

    determineEpic(item, epicMapping = {}) {
        const text = (item.text || item.description || '').toLowerCase();
        
        for (const [keyword, epic] of Object.entries(epicMapping)) {
            if (text.includes(keyword.toLowerCase())) {
                return epic;
            }
        }

        return null;
    }

    /**
     * Generate tickets from action items array (simplified interface)
     * @param {Array} actionItems - Array of action item strings
     * @param {Object} options - Options for ticket generation
     * @returns {Object} Result with tickets and metadata
     */
    async generateTicketsFromActionItems(actionItems, options = {}) {
        try {
            if (!Array.isArray(actionItems) || actionItems.length === 0) {
                return {
                    success: false,
                    error: 'Action items array is required and cannot be empty',
                    tickets: []
                };
            }

            const { projectId, context, template } = options;
            
            // Convert simple action items to enriched format
            const enrichedActionItems = actionItems.map((item, index) => ({
                content: typeof item === 'string' ? item : item.content || item.description || item,
                assignee: typeof item === 'object' ? item.assignee : null,
                priority: typeof item === 'object' ? item.priority : 'Medium',
                dueDate: typeof item === 'object' ? item.dueDate : null,
                context: context || '',
                confidence: typeof item === 'object' ? item.confidence : 0.8
            }));

            // Generate tickets using the main method
            const analysisResult = {
                actionItems: enrichedActionItems,
                summary: context ? { executiveSummary: context } : null,
                decisions: [],
                metadata: {
                    projectId,
                    template
                }
            };

            const result = await this.generateTickets(analysisResult, {
                includeActionItems: true,
                includeDecisions: false,
                includeSummaryTicket: false,
                projectId
            });

            return {
                success: true,
                tickets: result.tickets,
                summary: result.summary,
                metadata: result.metadata,
                processing_time: result.processing_time
            };

        } catch (error) {
            console.error('Error generating tickets from action items:', error);
            return {
                success: false,
                error: error.message,
                tickets: []
            };
        }
    }
}

module.exports = new TicketGeneratorService();