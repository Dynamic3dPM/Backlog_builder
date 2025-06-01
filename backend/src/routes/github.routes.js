const express = require('express');
const GitHubController = require('../controllers/github.controller');

const router = express.Router();

/**
 * @route GET /api/github/status
 * @desc Get GitHub integration status
 * @access Private
 */
router.get('/status', GitHubController.getStatus.bind(GitHubController));

/**
 * @route GET /api/github/test-connection
 * @desc Test GitHub API connection
 * @access Private
 */
router.get('/test-connection', GitHubController.testConnection.bind(GitHubController));

/**
 * @route GET /api/github/repository
 * @desc Get repository information
 * @access Private
 */
router.get('/repository', GitHubController.getRepositoryInfo.bind(GitHubController));

/**
 * @route GET /api/github/project
 * @desc Get project information
 * @access Private
 */
router.get('/project', GitHubController.getProjectInfo.bind(GitHubController));

/**
 * @route GET /api/github/account
 * @desc Get account information and permissions
 */
router.get('/account', GitHubController.getAccountInfo.bind(GitHubController));

// Project management routes
router.get('/projects', GitHubController.getAllProjects.bind(GitHubController));
router.get('/repositories', GitHubController.getAllRepositories.bind(GitHubController));
router.get('/projects/:projectId/repositories', GitHubController.getProjectRepositories.bind(GitHubController));
router.get('/projects/:projectId/fields', GitHubController.getProjectFields.bind(GitHubController));
router.get('/projects/:projectId/items', GitHubController.getProjectItems.bind(GitHubController));

// Issue creation routes
/**
 * @route POST /api/github/create-issue
 * @desc Create a single GitHub issue from ticket
 * @access Private
 */
router.post('/create-issue', GitHubController.createIssue.bind(GitHubController));

/**
 * @route POST /api/github/create-issues
 * @desc Create multiple GitHub issues from tickets
 * @access Private
 */
router.post('/create-issues', GitHubController.createIssues.bind(GitHubController));

module.exports = router;