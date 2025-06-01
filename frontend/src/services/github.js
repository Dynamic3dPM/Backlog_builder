import axios from 'axios';

const API_BASE_URL = process.env.VUE_APP_API_URL || 'http://localhost:3000/api';

const githubService = {
    /**
     * Get all GitHub projects
     * @returns {Promise<Array>} List of projects
     */
    async getProjects() {
        try {
            const response = await axios.get(`${API_BASE_URL}/github/projects`);
            return response.data.projects || [];
        } catch (error) {
            console.error('Error fetching projects:', error);
            throw error;
        }
    },

    /**
     * Get all repositories for a specific project
     * @param {string} projectId - The ID of the project
     * @returns {Promise<Array>} List of repositories
     */
    async getProjectRepositories(projectId) {
        try {
            const response = await axios.get(`${API_BASE_URL}/github/projects/${projectId}/repositories`);
            return response.data.repositories || [];
        } catch (error) {
            console.error('Error fetching project repositories:', error);
            throw error;
        }
    },

    /**
     * Create a new issue in a repository
     * @param {Object} issue - The issue data
     * @param {string} repository - The repository name
     * @param {string} projectId - The project ID (optional)
     * @returns {Promise<Object>} The created issue
     */
    async createIssue(issue, repository, projectId = null) {
        try {
            const payload = {
                ticket: issue,
                repository,
                projectId
            };

            const response = await axios.post(`${API_BASE_URL}/github/create-issue`, payload);
            return response.data;
        } catch (error) {
            console.error('Error creating issue:', error);
            throw error;
        }
    }
};

export default githubService;