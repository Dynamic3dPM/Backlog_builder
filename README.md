# Backlog Builder with GitHub Projects Integration

A powerful tool for converting product requirements into well-structured development tickets and managing them in GitHub Projects.

## Features

- Multi-step workflow for backlog building and ticket management
- Upload audio/video for transcription with Whisper STT
- AI-powered ticket generation from meeting transcripts (pattern-based and LLM analysis)
- Edit and review generated tickets, including AI "magic rewrite"
- Direct integration with GitHub Projects (dynamic repository and project selection)
- Bulk upload of tickets as GitHub Issues to selected project
- Modern, responsive UI with breadcrumb navigation
- Dockerized for easy local and production deployment
- API-first design for extensibility

---

## Backend Functions & Features

The backend is implemented in Node.js/Express and orchestrates all core workflow logic, AI analysis, and integration with GitHub. Key modules and features include:

### Main Modules
- **Controllers**: Route and orchestrate requests for AI, GitHub, audio, and ticket workflows.
  - `ai.controller.js`: Handles transcription, AI analysis (pattern-based & LLM), and ticket generation endpoints.
  - `github.controller.js`: Manages GitHub authentication, repository/project listing, and issue creation.
  - `upload.controller.js` / `processing.controller.js`: Handle file uploads and processing.
- **Services**: Core business logic for each workflow step.
  - `aiAnalysis.js`: Calls local (and optionally cloud) LLMs for meeting analysis, using pattern-based and LangChain-enhanced pipelines.
  - `ticketGenerator.js`: Converts analysis results into structured tickets, deduplicates, assigns types/priorities, and formats for GitHub.
  - `githubService.js`: Integrates with GitHub's REST and GraphQL APIs for repo/project/issue management.
  - `speechToText.js`: Handles audio/video transcription using Whisper STT.

### AI & LangChain Integration
- **LangChain** is used for advanced LLM orchestration, memory, and prompt engineering in ticket generation and meeting analysis.
- Both classic pattern-based and LangChain-enhanced analysis are supported; LangChain can be toggled via environment/config.

### Key Endpoints
- `POST /api/ai/analyze-conversation-smart`: Pattern-based conversation analysis
- `POST /api/ai/generate-tickets`: Generate tickets from analysis results
- `GET /api/github/repositories`: List available GitHub repositories
- `GET /api/github/projects`: List available GitHub Projects
- `POST /api/github/create-issues`: Bulk upload tickets as issues to GitHub
- `POST /transcribe`: Audio/video transcription to text

### Developer Notes
- All backend services are containerized and communicate via internal Docker networking.
- Environment variables control service URLs, tokens, and feature toggles (see `.env` and `docker-compose.yml`).
- API is designed for easy extension—add new controllers/services for additional integrations or workflows.
- Extensive error logging and validation are implemented for robust workflow automation.

---

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)
- Docker and Docker Compose (for containerized deployment)
- GitHub account with repository access
- GitHub Personal Access Token with appropriate permissions

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Dynamic3dPM/Backlog_builder.git
cd Backlog_builder
```

### 2. Install Dependencies

```bash
# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install

# Return to project root
cd ..
```

### 3. Configure Environment

Copy the example environment file and update with your configuration:

```bash
cp .env.example .env
```

Edit the `.env` file with your GitHub credentials and project details:

```
# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repository_name
GITHUB_PROJECT_NUMBER=1  # The project number from GitHub Projects

# Backend Configuration
PORT=3000
NODE_ENV=development

# Frontend Configuration
VITE_API_BASE_URL=http://localhost:3000
```

### 4. Start the Application

#### Development Mode

```bash
# Start backend
cd backend
npm run dev

# In a new terminal, start frontend
cd ../frontend
npm run dev
```

#### Production Mode with Docker

```bash
docker-compose up --build
```

## Usage

1. Access the web interface at `http://localhost:8085/workflow` (or your configured frontend port)
2. Follow the multi-step workflow:
   - **Upload Audio**: Upload an audio/video file for transcription or enter text manually
   - **Review Text**: Edit and approve the transcript
   - **Create Tickets**: Use AI-powered analysis to generate actionable tickets
   - **Edit Tickets**: Review, edit, or "magic rewrite" tickets for clarity
   - **Upload to Project**: Select a GitHub repository and project, preview, and upload tickets as issues
3. Use the top navigation to access the Workflow (Upload and GitHub links have been removed for simplicity)
4. All API endpoints are proxied through the frontend for seamless container-to-container communication

## API Endpoints

- `POST /api/ai/analyze-conversation-smart`: Pattern-based conversation analysis
- `POST /api/ai/generate-tickets`: Generate tickets from analysis results
- `GET /api/github/repositories`: List available GitHub repositories
- `GET /api/github/projects`: List available GitHub Projects
- `POST /api/github/create-issues`: Bulk upload tickets as issues to GitHub
- `POST /transcribe`: Audio/video transcription to text
- API documentation is available at `/api-docs` when running the backend server.

## Testing

Run the test suite:

```bash
# Run backend tests
cd backend
npm test

# Run frontend tests
cd ../frontend
npm test
```

## Deployment

For local or production deployment, use Docker Compose:

```bash
docker compose up --build -d
```

If you encounter issues with code changes not appearing, try rebuilding containers:

```bash
docker compose down
rm -rf frontend/node_modules frontend/dist
docker compose build --no-cache
docker compose up -d
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository.

## Acknowledgments

- Built with Node.js, Express, and React
- Uses GitHub's REST API for project management
- Inspired by modern agile development practices