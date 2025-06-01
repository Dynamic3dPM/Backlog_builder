const { Octokit } = require('@octokit/rest');
const { graphql } = require('@octokit/graphql');

class GitHubService {
    constructor() {
        this.token = process.env.GITHUB_TOKEN;
        this.owner = process.env.GITHUB_OWNER;
        this.repo = process.env.GITHUB_REPO;
        this.projectNumber = process.env.GITHUB_PROJECT_NUMBER;
        
        if (!this.token) {
            console.warn('GitHub token not configured. GitHub integration will be disabled.');
            return;
        }

        this.octokit = new Octokit({
            auth: this.token,
        });

        this.graphqlWithAuth = graphql.defaults({
            headers: {
                authorization: `token ${this.token}`,
            },
        });
    }

    /**
     * Test GitHub API connectivity
     */
    async testConnection() {
        try {
            if (!this.token) {
                throw new Error('GitHub token not configured');
            }

            const { data } = await this.octokit.rest.users.getAuthenticated();
            return {
                success: true,
                user: data.login,
                scopes: data.scopes || []
            };
        } catch (error) {
            console.error('GitHub connection test failed:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Create a GitHub issue from ticket data
     */
    async createIssue(ticket, repository = null, projectId = null) {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            if (!repository) {
                throw new Error('Repository name is required');
            }

            // Format the issue body
            const body = this.formatIssueBody(ticket);
            
            // Create labels based on ticket properties
            const labels = this.getLabelsFromTicket(ticket);

            // Create the issue
            const { data: issue } = await this.octokit.rest.issues.create({
                owner: this.owner,
                repo: repository,
                title: ticket.title,
                body: body,
                labels: labels,
                assignees: ticket.assignee ? [ticket.assignee] : []
            });

            console.log(`Created issue #${issue.number}: ${issue.title}`);

            // Add to GitHub Project if projectId is provided
            if (projectId) {
                try {
                    await this.addIssueToProject(issue.node_id, projectId);
                    console.log(`Added issue #${issue.number} to project ${projectId}`);
                } catch (projectError) {
                    console.warn(`Failed to add issue to project: ${projectError.message}`);
                    // Don't fail the entire operation if project addition fails
                }
            }

            return {
                success: true,
                issue: {
                    id: issue.id,
                    number: issue.number,
                    url: issue.html_url,
                    node_id: issue.node_id
                }
            };
        } catch (error) {
            console.error('Failed to create GitHub issue:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Add issue to GitHub Project
     */
    async addIssueToProject(issueNodeId, projectId) {
        try {
            // Add item to project using the project ID directly
            const addItemMutation = `
                mutation($projectId: ID!, $contentId: ID!) {
                    addProjectV2ItemById(input: {
                        projectId: $projectId
                        contentId: $contentId
                    }) {
                        item {
                            id
                        }
                    }
                }
            `;

            const result = await this.graphqlWithAuth(addItemMutation, {
                projectId: projectId,
                contentId: issueNodeId
            });

            if (result.addProjectV2ItemById?.item?.id) {
                return {
                    success: true,
                    itemId: result.addProjectV2ItemById.item.id
                };
            } else {
                throw new Error('Failed to add item to project');
            }
        } catch (error) {
            console.error('Failed to add issue to project:', error.message);
            throw error;
        }
    }

    /**
     * Format ticket data into GitHub issue body
     */
    formatIssueBody(ticket) {
        let body = '';

        if (ticket.description) {
            body += `## Description\n${ticket.description}\n\n`;
        }

        if (ticket.acceptanceCriteria && ticket.acceptanceCriteria.length > 0) {
            body += `## Acceptance Criteria\n`;
            ticket.acceptanceCriteria.forEach((criteria, index) => {
                body += `- [ ] ${criteria}\n`;
            });
            body += '\n';
        }

        if (ticket.tasks && ticket.tasks.length > 0) {
            body += `## Tasks\n`;
            ticket.tasks.forEach((task, index) => {
                body += `- [ ] ${task}\n`;
            });
            body += '\n';
        }

        if (ticket.priority) {
            body += `**Priority:** ${ticket.priority}\n`;
        }

        if (ticket.complexity) {
            body += `**Complexity:** ${ticket.complexity}\n`;
        }

        if (ticket.estimatedHours) {
            body += `**Estimated Hours:** ${ticket.estimatedHours}\n`;
        }

        if (ticket.dependencies && ticket.dependencies.length > 0) {
            body += `\n## Dependencies\n`;
            ticket.dependencies.forEach(dep => {
                body += `- ${dep}\n`;
            });
        }

        body += `\n---\n*Generated by Backlog Builder*`;

        return body;
    }

    /**
     * Get GitHub labels from ticket properties
     */
    getLabelsFromTicket(ticket) {
        const labels = [];

        if (ticket.type) {
            labels.push(ticket.type);
        }

        if (ticket.priority) {
            labels.push(`priority:${ticket.priority.toLowerCase()}`);
        }

        if (ticket.complexity) {
            labels.push(`complexity:${ticket.complexity.toLowerCase()}`);
        }

        // Add default label
        labels.push('backlog-builder');

        return labels;
    }

    /**
     * Create multiple issues from tickets array
     */
    async createIssuesFromTickets(tickets, repository = null, projectId = null) {
        const results = [];

        for (const ticket of tickets) {
            const result = await this.createIssue(ticket, repository, projectId);
            results.push({
                ticket: ticket.title,
                ...result
            });

            // Add delay to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        return {
            success: true,
            results: results,
            summary: {
                total: tickets.length,
                successful: results.filter(r => r.success).length,
                failed: results.filter(r => !r.success).length
            }
        };
    }

    /**
     * Get repository information
     */
    async getRepositoryInfo(repository = null) {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            if (!repository) {
                throw new Error('Repository name is required');
            }

            const { data } = await this.octokit.rest.repos.get({
                owner: this.owner,
                repo: repository
            });

            return {
                success: true,
                repository: {
                    name: data.name,
                    full_name: data.full_name,
                    url: data.html_url,
                    private: data.private,
                    default_branch: data.default_branch
                }
            };
        } catch (error) {
            console.error('Failed to get repository info:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get project information
     */
    async getProjectInfo(projectId) {
        try {
            if (!projectId) {
                return {
                    success: false,
                    error: 'Project number not configured'
                };
            }

            const projectQuery = `
                query($owner: String!, $number: Int!) {
                    user(login: $owner) {
                        projectV2(number: $number) {
                            id
                            title
                            url
                            shortDescription
                        }
                    }
                    organization(login: $owner) {
                        projectV2(number: $number) {
                            id
                            title
                            url
                            shortDescription
                        }
                    }
                }
            `;

            const result = await this.graphqlWithAuth(projectQuery, {
                owner: this.owner,
                number: parseInt(projectId)
            });

            const project = result.user?.projectV2 || result.organization?.projectV2;

            if (!project) {
                throw new Error(`Project ${projectId} not found`);
            }

            return {
                success: true,
                project: {
                    id: project.id,
                    title: project.title,
                    url: project.url,
                    description: project.shortDescription
                }
            };
        } catch (error) {
            console.error('Failed to get project info:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get all projects for the user/organization
     */
    async getAllProjects() {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            let allProjects = [];
            let accountType = 'unknown';

            // First try to get organization projects
            try {
                const orgQuery = `
                    query($owner: String!, $first: Int!) {
                        organization(login: $owner) {
                            projectsV2(first: $first) {
                                nodes {
                                    id
                                    number
                                    title
                                    url
                                    shortDescription
                                    public
                                    closed
                                    createdAt
                                    updatedAt
                                }
                                pageInfo {
                                    hasNextPage
                                    endCursor
                                }
                            }
                        }
                    }
                `;
                
                const orgResult = await this.graphqlWithAuth(orgQuery, {
                    owner: this.owner,
                    first: 50
                });
                
                if (orgResult.organization) {
                    accountType = 'organization';
                    allProjects = orgResult.organization.projectsV2.nodes || [];
                    console.log(`Found ${allProjects.length} organization projects for ${this.owner}`);
                }
            } catch (orgError) {
                console.log(`${this.owner} is not an organization or no access to org projects:`, orgError.message);
                
                // If organization query fails, try user projects
                try {
                    const userQuery = `
                        query($owner: String!, $first: Int!) {
                            user(login: $owner) {
                                projectsV2(first: $first) {
                                    nodes {
                                        id
                                        number
                                        title
                                        url
                                        shortDescription
                                        public
                                        closed
                                        createdAt
                                        updatedAt
                                    }
                                    pageInfo {
                                        hasNextPage
                                        endCursor
                                    }
                                }
                            }
                        }
                    `;

                    const userResult = await this.graphqlWithAuth(userQuery, {
                        owner: this.owner,
                        first: 50
                    });

                    if (userResult.user) {
                        accountType = 'user';
                        allProjects = userResult.user.projectsV2.nodes || [];
                        console.log(`Found ${allProjects.length} user projects for ${this.owner}`);
                    }
                } catch (userError) {
                    throw new Error(`Failed to access projects for ${this.owner}: ${userError.message}`);
                }
            }

            return {
                success: true,
                accountType: accountType,
                projects: allProjects.map(project => ({
                    id: project.id,
                    number: project.number,
                    title: project.title,
                    url: project.url,
                    description: project.shortDescription,
                    public: project.public,
                    closed: project.closed,
                    createdAt: project.createdAt,
                    updatedAt: project.updatedAt
                })),
                total: allProjects.length
            };
        } catch (error) {
            console.error('Failed to get all projects:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get repositories for a specific project
     */
    async getProjectRepositories(projectId) {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            if (!projectId) {
                throw new Error('Project ID is required');
            }

            // First get the project to check if it's an organization or user project
            const projectQuery = `
                query($projectId: ID!) {
                    node(id: $projectId) {
                        ... on ProjectV2 {
                            id
                            title
                            owner {
                                ... on Organization {
                                    id
                                    login
                                }
                                ... on User {
                                    id
                                    login
                                }
                            }
                        }
                    }
                }
            `;

            const projectResult = await this.graphqlWithAuth(projectQuery, {
                projectId: projectId
            });

            const project = projectResult.node;
            if (!project) {
                throw new Error('Project not found');
            }

            // Get all repositories for the owner (user or organization)
            const owner = project.owner.login;
            const isOrg = project.owner.__typename === 'Organization';

            let repos = [];
            let hasNextPage = true;
            let endCursor = null;

            while (hasNextPage) {
                const query = `
                    query($owner: String!, $after: String) {
                        ${isOrg ? 'organization' : 'user'}(login: $owner) {
                            repositories(first: 100, after: $after, isArchived: false) {
                                nodes {
                                    id
                                    name
                                    full_name: nameWithOwner
                                    description
                                    url
                                    isPrivate
                                    isArchived
                                    isDisabled
                                    isTemplate
                                    hasIssuesEnabled
                                }
                                pageInfo {
                                    endCursor
                                    hasNextPage
                                }
                            }
                        }
                    }
                `;

                const result = await this.graphqlWithAuth(query, {
                    owner: owner,
                    after: endCursor
                });

                const ownerData = result[isOrg ? 'organization' : 'user'];
                const pageRepos = ownerData?.repositories?.nodes || [];
                
                // Filter out archived, disabled, and template repositories
                const filteredRepos = pageRepos.filter(repo => 
                    !repo.isArchived && 
                    !repo.isDisabled && 
                    !repo.isTemplate &&
                    repo.hasIssuesEnabled
                );

                repos = [...repos, ...filteredRepos];
                hasNextPage = ownerData?.repositories?.pageInfo?.hasNextPage || false;
                endCursor = ownerData?.repositories?.pageInfo?.endCursor;
            }

            return {
                success: true,
                repositories: repos.map(repo => ({
                    id: repo.id,
                    name: repo.name,
                    full_name: repo.full_name,
                    description: repo.description,
                    url: repo.url,
                    private: repo.isPrivate
                }))
            };
        } catch (error) {
            console.error('Failed to get project repositories:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get repositories for the user/organization
     */
    async getAllRepositories() {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            // Get repositories for user/organization
            const { data } = await this.octokit.rest.repos.listForUser({
                username: this.owner,
                type: 'all',
                sort: 'updated',
                per_page: 100
            });

            return {
                success: true,
                repositories: data.map(repo => ({
                    id: repo.id,
                    name: repo.name,
                    full_name: repo.full_name,
                    url: repo.html_url,
                    description: repo.description,
                    private: repo.private,
                    default_branch: repo.default_branch,
                    language: repo.language,
                    stars: repo.stargazers_count,
                    forks: repo.forks_count,
                    updated_at: repo.updated_at
                })),
                total: data.length
            };
        } catch (error) {
            console.error('Failed to get repositories:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get project fields and configuration
     */
    async getProjectFields(projectId) {
        try {
            if (!projectId) {
                throw new Error('Project ID is required');
            }

            const fieldsQuery = `
                query($projectId: ID!) {
                    node(id: $projectId) {
                        ... on ProjectV2 {
                            fields(first: 20) {
                                nodes {
                                    ... on ProjectV2Field {
                                        id
                                        name
                                        dataType
                                    }
                                    ... on ProjectV2SingleSelectField {
                                        id
                                        name
                                        dataType
                                        options {
                                            id
                                            name
                                            color
                                        }
                                    }
                                    ... on ProjectV2IterationField {
                                        id
                                        name
                                        dataType
                                        configuration {
                                            iterations {
                                                id
                                                title
                                                startDate
                                                duration
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            `;

            const result = await this.graphqlWithAuth(fieldsQuery, {
                projectId: projectId
            });

            const fields = result.node?.fields?.nodes || [];

            return {
                success: true,
                fields: fields.map(field => ({
                    id: field.id,
                    name: field.name,
                    dataType: field.dataType,
                    options: field.options || null,
                    configuration: field.configuration || null
                }))
            };
        } catch (error) {
            console.error('Failed to get project fields:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get project items (issues/PRs in the project)
     */
    async getProjectItems(projectId, limit = 50) {
        try {
            if (!projectId) {
                throw new Error('Project ID is required');
            }

            const itemsQuery = `
                query($projectId: ID!, $first: Int!) {
                    node(id: $projectId) {
                        ... on ProjectV2 {
                            items(first: $first) {
                                nodes {
                                    id
                                    content {
                                        ... on Issue {
                                            id
                                            number
                                            title
                                            url
                                            state
                                            createdAt
                                            updatedAt
                                            assignees(first: 5) {
                                                nodes {
                                                    login
                                                    name
                                                }
                                            }
                                            labels(first: 10) {
                                                nodes {
                                                    name
                                                    color
                                                }
                                            }
                                        }
                                        ... on PullRequest {
                                            id
                                            number
                                            title
                                            url
                                            state
                                            createdAt
                                            updatedAt
                                        }
                                    }
                                    fieldValues(first: 8) {
                                        nodes {
                                            ... on ProjectV2ItemFieldTextValue {
                                                text
                                                field {
                                                    ... on ProjectV2FieldCommon {
                                                        name
                                                    }
                                                }
                                            }
                                            ... on ProjectV2ItemFieldSingleSelectValue {
                                                name
                                                field {
                                                    ... on ProjectV2FieldCommon {
                                                        name
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                pageInfo {
                                    hasNextPage
                                    endCursor
                                }
                            }
                        }
                    }
                }
            `;

            const result = await this.graphqlWithAuth(itemsQuery, {
                projectId: projectId,
                first: limit
            });

            const items = result.node?.items?.nodes || [];

            return {
                success: true,
                items: items.map(item => ({
                    id: item.id,
                    content: item.content,
                    fieldValues: item.fieldValues?.nodes || []
                })),
                hasNextPage: result.node?.items?.pageInfo?.hasNextPage || false
            };
        } catch (error) {
            console.error('Failed to get project items:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Check account type and permissions
     */
    async getAccountInfo() {
        try {
            if (!this.token || !this.owner) {
                throw new Error('GitHub configuration incomplete');
            }

            const accountQuery = `
                query($owner: String!) {
                    user(login: $owner) {
                        id
                        login
                        name
                        email
                        bio
                        company
                        location
                        createdAt
                    }
                    organization(login: $owner) {
                        id
                        login
                        name
                        email
                        description
                        location
                        createdAt
                        viewerCanAdminister
                        viewerIsAMember
                    }
                }
            `;

            const result = await this.graphqlWithAuth(accountQuery, {
                owner: this.owner
            });

            let accountInfo = {
                type: 'unknown',
                details: null,
                permissions: null
            };

            if (result.organization) {
                accountInfo = {
                    type: 'organization',
                    details: {
                        id: result.organization.id,
                        login: result.organization.login,
                        name: result.organization.name,
                        email: result.organization.email,
                        description: result.organization.description,
                        location: result.organization.location,
                        createdAt: result.organization.createdAt
                    },
                    permissions: {
                        canAdminister: result.organization.viewerCanAdminister,
                        isMember: result.organization.viewerIsAMember
                    }
                };
            } else if (result.user) {
                accountInfo = {
                    type: 'user',
                    details: {
                        id: result.user.id,
                        login: result.user.login,
                        name: result.user.name,
                        email: result.user.email,
                        bio: result.user.bio,
                        company: result.user.company,
                        location: result.user.location,
                        createdAt: result.user.createdAt
                    },
                    permissions: {
                        isOwner: true
                    }
                };
            }

            return {
                success: true,
                account: accountInfo
            };
        } catch (error) {
            console.error('Failed to get account info:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }
}

module.exports = GitHubService;