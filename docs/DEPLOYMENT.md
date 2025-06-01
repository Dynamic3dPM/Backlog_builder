# Backlog Builder Deployment Guide

This guide covers the deployment of the Backlog Builder application in different environments.

## Prerequisites

- Docker 20.10.0 or higher
- Docker Compose 1.29.0 or higher
- Node.js 14.x or higher (for development)
- npm 6.x or higher (for development)
- GitHub account with repository access
- GitHub Personal Access Token with `repo` and `project` scopes

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Application
NODE_ENV=production
PORT=3000

# GitHub Integration
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repository_name
GITHUB_PROJECT_NUMBER=1

# Frontend
VITE_API_BASE_URL=/api

# Database (if applicable)
DB_HOST=db
DB_PORT=5432
DB_NAME=backlog_builder
DB_USER=postgres
DB_PASSWORD=secure_password

# JWT (generate with `openssl rand -hex 32`)
JWT_SECRET=your_jwt_secret
```

## Deployment Options

### 1. Development Mode

```bash
# Install dependencies
npm install

# Start development servers
npm run dev:backend
# In another terminal
npm run dev:frontend
```

### 2. Docker Compose (Production)

```bash
# Build and start containers
docker-compose -f docker-compose.prod.yml up --build -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### 3. Kubernetes (Production)

1. Create a Kubernetes cluster
2. Install the NGINX Ingress Controller
3. Create a `backlog-builder` namespace
4. Deploy the application:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## GitHub Integration Setup

1. Create a new GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate a new token with `repo` and `project` scopes
   - Copy the token and add it to your `.env` file as `GITHUB_TOKEN`

2. Get your GitHub Project number:
   - Go to your GitHub Project
   - The project number is in the URL: `https://github.com/orgs/your-org/projects/NUMBER`
   - Add this number to your `.env` as `GITHUB_PROJECT_NUMBER`

## Health Checks

- API Health: `GET /health`
- Frontend: `http://localhost:5173` (development) or your domain (production)
- API Documentation: `http://localhost:3000/api-docs` (when running locally)

## Monitoring

- **Logs**: View container logs with `docker-compose logs -f`
- **Metrics**: Prometheus metrics available at `/metrics`
- **Tracing**: Distributed tracing with Jaeger

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker exec -t postgres pg_dump -U postgres backlog_builder > backup_$(date +%Y-%m-%d).sql

# Restore from backup
cat backup_2023-01-01.sql | docker exec -i postgres psql -U postgres backlog_builder
```

### Configuration Backup

Backup the following:
- `.env` file
- Docker volumes
- Any custom configurations

## Scaling

### Horizontal Scaling

For high traffic, you can scale the backend service:

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3
```

### Load Balancing

The Docker Compose setup includes an NGINX load balancer. For production, consider using:
- AWS ALB/ELB
- Google Cloud Load Balancer
- Azure Load Balancer

## Security Considerations

1. **Secrets Management**:
   - Never commit `.env` files
   - Use Docker secrets or Kubernetes secrets in production
   - Rotate tokens and passwords regularly

2. **Network Security**:
   - Enable HTTPS with valid certificates
   - Use network policies to restrict container communication
   - Implement rate limiting

3. **Updates**:
   - Regularly update dependencies
   - Keep the host system patched
   - Monitor for security advisories

## Troubleshooting

### Common Issues

1. **GitHub API Rate Limits**
   - Check your rate limit status at `https://api.github.com/rate_limit`
   - Consider using a GitHub App for higher rate limits

2. **Docker Build Failures**
   - Ensure Docker has enough resources (CPU, memory)
   - Clear Docker cache: `docker system prune -a`

3. **Database Connection Issues**
   - Verify database credentials in `.env`
   - Check if the database container is running
   - Check database logs: `docker-compose logs db`

For additional help, please open an issue in the repository.