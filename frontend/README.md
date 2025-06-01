# Backlog Builder - Frontend

This is the frontend for the Backlog Builder application, which includes GitHub integration for managing issues and projects.

## Features

- View and select GitHub projects
- Dynamic repository selection based on the selected project
- Create issues in the selected repository
- Add issues to the selected project

## Prerequisites

- Node.js (v14+)
- npm or yarn
- Backend server running (see the backend README for setup instructions)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and update the API URL if needed:
   ```bash
   cp .env.example .env
   ```
4. Start the development server:
   ```bash
   npm run serve
   ```

## Usage

1. Navigate to the GitHub Integration page
2. Select a project from the dropdown
3. Select a repository from the repository dropdown (filtered based on the selected project)
4. Click "Create Sample Issue" to test the integration

## Project Structure

- `src/components/` - Reusable Vue components
  - `ProjectSelector.vue` - Component for selecting projects and repositories
- `src/views/` - Page components
  - `GitHubIntegration.vue` - GitHub integration page
- `src/services/` - API services
  - `github.js` - Service for interacting with the GitHub API
- `src/router.js` - Vue Router configuration
- `src/App.vue` - Main application component
- `src/main.js` - Application entry point

## Dependencies

- Vue 3
- Vue Router
- Axios for HTTP requests
- Bootstrap 5 for styling
- vue3-toastify for toast notifications

## License

MIT
