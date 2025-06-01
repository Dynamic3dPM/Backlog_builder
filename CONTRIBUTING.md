# Contributing to Backlog Builder

Thank you for your interest in contributing to Backlog Builder! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top-right corner of the repository page
   - Clone your forked repository locally:
     ```bash
     git clone https://github.com/your-username/Backlog_builder.git
     cd Backlog_builder
     ```

2. **Set Up Remote Upstream**
   ```bash
   git remote add upstream https://github.com/Dynamic3dPM/Backlog_builder.git
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

## Development Environment

### Prerequisites

- Node.js 14.x or higher
- npm 6.x or higher
- Docker and Docker Compose (for containerized development)

### Setup

1. Install dependencies:
   ```bash
   # Install backend dependencies
   cd backend
   npm install
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the development servers:
   ```bash
   # In one terminal
   cd backend
   npm run dev
   
   # In another terminal
   cd frontend
   npm run dev
   ```

## Making Changes

1. **Sync with Upstream**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests** (see [Testing](#testing))

4. **Commit your changes** with a descriptive commit message:
   ```
   feat: add new feature
   fix: fix issue #123
   docs: update documentation
   style: format code
   refactor: improve code structure
   test: add tests
   chore: update dependencies
   ```

5. **Push your changes**
   ```bash
   git push origin your-branch-name
   ```

## Pull Request Process

1. Open a Pull Request (PR) from your forked repository to the main repository's `main` branch
2. Ensure your PR description clearly explains the problem and solution
3. Include relevant issue numbers if applicable
4. Ensure all tests pass
5. Update documentation as needed
6. Request review from maintainers

## Reporting Bugs

Create a new issue with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected vs actual behavior
4. Environment details (OS, Node.js version, etc.)
5. Screenshots or logs if applicable

## Feature Requests

1. Check existing issues to avoid duplicates
2. Describe the feature and why it's valuable
3. Include any relevant use cases or examples

## Code Style

### JavaScript/TypeScript

- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use ES6+ features
- Use async/await instead of callbacks where possible
- Use destructuring for objects and arrays

### Naming Conventions

- Use camelCase for variables and functions
- Use PascalCase for classes and components
- Use UPPER_CASE for constants
- Prefix boolean variables with is/has/should (e.g., `isLoading`)

## Testing

### Running Tests

```bash
# Run all tests
npm test

# Run backend tests
cd backend
npm test

# Run frontend tests
cd frontend
npm test

# Run specific test file
npm test -- path/to/test/file.test.js

# Run with coverage
npm run test:coverage
```

### Writing Tests

- Write unit tests for all new features
- Test edge cases and error conditions
- Mock external dependencies
- Use descriptive test names

## Documentation

- Update README.md for significant changes
- Add JSDoc comments for new functions and classes
- Update API documentation when endpoints change
- Keep CHANGELOG.md updated

## License

By contributing, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
