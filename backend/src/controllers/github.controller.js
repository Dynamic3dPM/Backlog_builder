const GitHubService = require('../services/githubService');

class GitHubController {
    constructor() {
        this.githubService = new GitHubService();
    }

    /**
     * Test GitHub connection
     */
    async testConnection(req, res) {
        try {
            const result = await this.githubService.testConnection();
            
            if (result.success) {
                res.json({
                    success: true,
                    message: 'GitHub connection successful',
                    user: result.user,
                    scopes: result.scopes
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('GitHub connection test error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Create GitHub issue from ticket
     */
    async createIssue(req, res) {
        try {
            const { ticket, repository, projectId } = req.body;

            if (!ticket) {
                return res.status(400).json({
                    success: false,
                    error: 'Ticket data is required'
                });
            }

            if (!repository) {
                return res.status(400).json({
                    success: false,
                    error: 'Repository name is required'
                });
            }

            const result = await this.githubService.createIssue(ticket, repository, projectId);

            if (result.success) {
                res.json({
                    success: true,
                    message: 'GitHub issue created successfully',
                    issue: result.issue
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Create GitHub issue error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Create multiple GitHub issues from tickets
     */
    async createIssues(req, res) {
        try {
            const { tickets, repository, projectId } = req.body;

            if (!tickets || !Array.isArray(tickets) || tickets.length === 0) {
                return res.status(400).json({
                    success: false,
                    error: 'Tickets array is required'
                });
            }

            if (!repository) {
                return res.status(400).json({
                    success: false,
                    error: 'Repository name is required'
                });
            }

            const results = await this.githubService.createIssuesFromTickets(tickets, repository, projectId);

            res.json({
                success: true,
                message: `Created ${results.summary.successful} of ${results.summary.total} GitHub issues`,
                results: results.results,
                summary: results.summary
            });
        } catch (error) {
            console.error('Create GitHub issues error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Get repository information
     */
    async getRepositoryInfo(req, res) {
        try {
            const result = await this.githubService.getRepositoryInfo();

            if (result.success) {
                res.json({
                    success: true,
                    repository: result.repository
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Get repository info error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Get project information
     */
    async getProjectInfo(req, res) {
        try {
            const result = await this.githubService.getProjectInfo();

            if (result.success) {
                res.json({
                    success: true,
                    project: result.project
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Get project info error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Get GitHub integration status
     */
    async getStatus(req, res) {
        try {
            const connectionResult = await this.githubService.testConnection();
            
            const status = {
                connection: connectionResult.success,
                configuration: {
                    token_configured: !!process.env.GITHUB_TOKEN,
                    owner_configured: !!process.env.GITHUB_OWNER,
                    dynamic_repo_and_project: true // Repositories and projects are now dynamic
                }
            };

            if (connectionResult.success) {
                status.user = connectionResult.user;
            }

            res.json({
                success: true,
                status: status
            });
        } catch (error) {
            console.error('GitHub status error:', error);
            res.status(500).json({
                success: false,
                error: 'Internal server error'
            });
        }
    }

    /**
     * Get all GitHub projects
     */
    async getAllProjects(req, res) {
        try {
            const result = await this.githubService.getAllProjects();
            
            if (result.success) {
                res.json({
                    success: true,
                    projects: result.projects,
                    total: result.total
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting all projects:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get projects'
            });
        }
    }

    /**
     * Get repositories for a specific project
     */
    async getProjectRepositories(req, res) {
        try {
            const { projectId } = req.params;
            
            if (!projectId) {
                return res.status(400).json({
                    success: false,
                    error: 'Project ID is required'
                });
            }

            const result = await this.githubService.getProjectRepositories(projectId);
            
            if (result.success) {
                res.json({
                    success: true,
                    repositories: result.repositories,
                    total: result.repositories?.length || 0
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting project repositories:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get project repositories'
            });
        }
    }

    /**
     * Get all repositories
     */
    async getAllRepositories(req, res) {
        try {
            const result = await this.githubService.getAllRepositories();
            
            if (result.success) {
                res.json({
                    success: true,
                    repositories: result.repositories,
                    total: result.total
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting all repositories:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get repositories'
            });
        }
    }

    /**
     * Get project fields
     */
    async getProjectFields(req, res) {
        try {
            const { projectId } = req.params;
            
            if (!projectId) {
                return res.status(400).json({
                    success: false,
                    error: 'Project ID is required'
                });
            }

            const result = await this.githubService.getProjectFields(projectId);
            
            if (result.success) {
                res.json({
                    success: true,
                    fields: result.fields
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting project fields:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get project fields'
            });
        }
    }

    /**
     * Get project items
     */
    async getProjectItems(req, res) {
        try {
            const { projectId } = req.params;
            const { limit = 50 } = req.query;
            
            if (!projectId) {
                return res.status(400).json({
                    success: false,
                    error: 'Project ID is required'
                });
            }

            const result = await this.githubService.getProjectItems(projectId, parseInt(limit));
            
            if (result.success) {
                res.json({
                    success: true,
                    items: result.items,
                    hasNextPage: result.hasNextPage
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting project items:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get project items'
            });
        }
    }

    /**
     * Get account information and permissions
     */
    async getAccountInfo(req, res) {
        try {
            const result = await this.githubService.getAccountInfo();
            
            if (result.success) {
                res.json({
                    success: true,
                    account: result.account
                });
            } else {
                res.status(400).json({
                    success: false,
                    error: result.error
                });
            }
        } catch (error) {
            console.error('Error getting account info:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to get account info'
            });
        }
    }
}

module.exports = new GitHubController();