import axios from 'axios';

// Adjust the URL format to match what we're providing in the environment variable
const API_URL = process.env.VUE_APP_API_URL || 'http://localhost:3001';
const API_BASE_URL = `${API_URL}/api`;

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
     * @param {string} projectId - The project ID
     * @returns {Promise<Array>} List of repositories
     */
    async getProjectRepositories(projectId) {
        try {
            const response = await axios.get(`${API_BASE_URL}/github/repositories?projectId=${projectId}`);
            return response.data.repositories || [];
        } catch (error) {
            console.error('Error fetching repositories:', error);
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
    async createIssue(ticket, repository, projectId = null) {
        try {
            const response = await axios.post(`${API_BASE_URL}/github/create-issue`, {
                ticket,  // Changed from 'issue' to 'ticket' to match backend expectation
                repository,
                projectId
            });
            return response.data;
        } catch (error) {
            console.error('Error creating issue:', error);
            throw error;
        }
    }
};

export default githubService;
