# Backlog Builder with GitHub Projects Integration

A powerful tool for converting product requirements into well-structured development tickets and managing them in GitHub Projects.

## Features

- Convert natural language requirements into structured development tickets
- Seamless integration with GitHub Projects
- Automated ticket creation and organization
- Customizable ticket templates
- Support for different project management workflows
- API-first design for easy integration with other tools

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

1. Access the web interface at `http://localhost:5173` (or your configured port)
2. Enter your project requirements or upload a document
3. Review the generated tickets
4. Push tickets directly to your GitHub Project

## API Documentation

API documentation is available at `/api-docs` when running the backend server.

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

For production deployment, use the provided Docker setup:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
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